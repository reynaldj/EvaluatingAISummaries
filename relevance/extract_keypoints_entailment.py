import numpy as np
import re
import os
import json
import tiktoken
import time
import random
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EntailmentKeyPointExtractor:
    def __init__(self, model_name="meta-llama-3.1-8b-instruct"):
        self.api_key = os.environ.get("ACADEMIC_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("Set the ACADEMIC_CLOUD_API_KEY environment variable before running this extractor.")
        self.base_url = "https://chat-ai.academiccloud.de/v1"
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        print("Loading SentenceTransformer model...")
        try:
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        except Exception as exc:
            print(f"  [Warning] Failed to load intfloat/multilingual-e5-large ({exc}); falling back to all-MiniLM-L6-v2.")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.kp_cache = {}

    def _log_key_point_count(self, stage, items, previous_count=None):
        count = len(items) if items else 0
        if previous_count is None:
            print(f"    - {stage}: {count} key points")
        else:
            delta = count - previous_count
            sign = "+" if delta >= 0 else ""
            print(f"    - {stage}: {count} key points ({sign}{delta})")

    def _safe_api_call(self, messages, temperature=0.0, max_attempts=5, timeout=120, max_tokens=None):
        last_error = None
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout,
                    max_tokens=max_tokens,
                    extra_body={}
                )
                usage = getattr(response, "usage", None)
                if usage is not None:
                    self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self.total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                content = response.choices[0].message.content
                if content:
                    return content.strip()
                last_error = "empty response body"
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                if "429" in error_msg:
                    wait = 5 + 3 * attempt + random.uniform(0, 2)
                    print(f"    [Rate Limit] Waiting {wait:.0f}s (attempt {attempt+1}/{max_attempts})...")
                elif any(s in error_msg for s in ("500", "502", "503", "timeout", "timed out", "connection")):
                    wait = min(60, 5 * (2 ** attempt)) + random.uniform(0, 1)
                    print(f"    [Server Error] Retrying in {wait:.0f}s (attempt {attempt+1}/{max_attempts})...")
                else:
                    wait = 2 + 2 * attempt + random.uniform(0, 1)
                    print(f"    [API Error] {error_msg[:100]} (attempt {attempt+1}/{max_attempts})...")
                time.sleep(wait)

        raise SystemExit(
            f"\n[FATAL] API call failed after {max_attempts} attempts. Last error: {last_error}"
        )

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    # ─── Transcript Parsing ────────────────────────────────────────────

    def parse_transcript_text(self, content):
        turns = []
        speaker_bracket_pattern = r'\[(Speaker \d+)\]:\s*(.*?)(?=\n\[Speaker \d+\]:|\Z)'
        timestamp_pattern = r'\[(\d{2}:\d{2}:\d{2})\]\s*(.*?)\n(.*?)(?=\n\n\[|\Z)'
        cleaned_pattern = r'(Sprecher \d+|[\w\s]+)\n(.*?)(?=\n(?:Sprecher \d+|[\w\s]+)\n|\Z)'

        matches = list(re.finditer(speaker_bracket_pattern, content, re.DOTALL))
        if len(matches) > 1:
            for i, match in enumerate(matches):
                turns.append({
                    'timestamp': f"Turn-{i+1:03d}",
                    'speaker': match.group(1).strip(),
                    'text': match.group(2).strip()
                })
        else:
            matches = list(re.finditer(timestamp_pattern, content, re.DOTALL))
            if matches:
                for match in matches:
                    turns.append({
                        'timestamp': match.group(1),
                        'speaker': match.group(2).strip(),
                        'text': match.group(3).strip().replace('\n', ' ')
                    })
            else:
                matches = list(re.finditer(cleaned_pattern, content, re.DOTALL))
                if len(matches) > 1:
                    for i, match in enumerate(matches):
                        turns.append({
                            'timestamp': f"Turn-{i+1:03d}",
                            'speaker': match.group(1).strip(),
                            'text': match.group(2).strip().replace('\n', ' ')
                        })
                else:
                    sentences = re.split(r'(?<=[.!?]) +', content)
                    for i, sent in enumerate(sentences):
                        if len(sent.strip()) > 5:
                            turns.append({
                                'timestamp': f"Sent-{i+1:03d}",
                                'speaker': "Speaker",
                                'text': sent.strip()
                            })
        return turns

    # ─── Key Item Extraction ───────────────────────────────────────────

    def extract_key_items(self, text, short=False, previously_extracted=None):
        length_desc = "von höchstens 150 Wörtern" if short else "von höchstens 5 % der ursprünglichen Textlänge"
        summary_ref = "150-Wörter-Zusammenfassung" if short else "5%-Zusammenfassung"

        context_hint = ""
        if previously_extracted:
            context_hint = f"""
Bereits aus vorherigen Abschnitten extrahiert (NICHT wiederholen):
{chr(10).join('- ' + p for p in previously_extracted[-15:])}

"""

        prompt = f"""Du liest das folgende Sitzungsprotokoll.

Deine Aufgabe ist es, ausschließlich die Kernaussagen zu extrahieren, die mit hoher Wahrscheinlichkeit auch in einer sehr knappen Zusammenfassung {length_desc} enthalten wären.
{context_hint}
Arbeite intern in drei Schritten:
1. Identifiziere alle potenziell relevanten Aussagen.
2. Bewerte gedanklich ihre Bedeutung für das Gesamtverständnis der Diskussion.
3. Behalte nur die Aussagen, ohne die eine {summary_ref} unvollständig wäre.

Zeige diese Schritte oder deine Bewertung NICHT.

Berücksichtige ausschließlich:
- zentrale Beschlüsse und Entscheidungen
- wesentliche Ergebnisse
- den Kernkonflikt oder das Hauptthema
- entscheidende Fakten oder Zahlen (nur wenn sie für das Verständnis unverzichtbar sind)
- die wichtigsten Argumente oder Positionen, sofern sie den Verlauf oder das Ergebnis der Diskussion maßgeblich beeinflussen
- konkrete nächste Schritte oder Maßnahmen

Ignoriere konsequent:
- Begrüßungen, Verabschiedungen und Sprecherwechsel
- organisatorische oder formale Hinweise
- Detailargumente und Nebenaspekte
- historische Hintergrundinformationen, Ortsbeschreibungen und kontextuelle Einleitungen
- einzelne Beispiele
- Wiederholungen
- Erläuterungen ohne neue Information
- Fakten, Daten oder Zahlen, die für die Kernaussage nicht notwendig sind

Regeln:
- Gib ausschließlich eine flache Liste von Stichpunkten aus.
- Keine Überschriften oder Kategorien.
- Jeder Stichpunkt enthält genau EINE atomare Kernaussage.
- Wenn ein Punkt eine Meinung oder Position darstellt, nenne den Akteur (z.B. "Die CDU fordert..." statt "Man sollte...").
- Wenn ein Satz mehrere unabhängige Fakten enthält, trenne sie in mehrere Stichpunkte.
- Extrahiere alle Punkte, die mit hoher Wahrscheinlichkeit in einer {summary_ref} vorkommen würden.
- Formuliere jeden Punkt als klare, eigenständige Aussage (Subjekt–Prädikat–Objekt).
- Paraphrasiere in sauberes Deutsch: keine wörtlichen Fragmente, keine Satzabbrüche, keine Füllwörter.
- Jeder Punkt besteht aus höchstens 25 Wörtern.
- Vermeide unscharfe Formulierungen wie "oder so", "irgendwie", "so ein bisschen".
- Verboten sind Meta-Sätze über den Gesprächsverlauf wie "Die Diskussion dreht sich um …".
- Verboten sind Frage-Rahmen wie "Die Frage ist, ob …".
- Wenn eine Stelle unklar oder mehrdeutig ist, verwerfe sie.
- Korrigiere offensichtliche Transkriptionsfehler.
- Füge keine Interpretationen oder zusätzlichen Informationen hinzu.

Dokument:
{text}
"""
        content = self._safe_api_call([
            {"role": "system", "content": "Du bist ein hilfreicher Assistent, der präzise und sachlich analysiert."},
            {"role": "user", "content": prompt}
        ], max_tokens=600)

        if not content:
            print("    [Warning] API returned empty content for key item extraction.")
            return []

        items = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            cleaned = None
            if re.match(r"^[-*•]\s+", line):
                cleaned = re.sub(r"^[-*•]\s+", "", line)
            elif re.match(r"^\d+[.)]\s+", line):
                cleaned = re.sub(r"^\d+[.)]\s+", "", line)
            else:
                continue

            cleaned = cleaned.strip()
            cleaned = re.sub(r"^\*\*(.*?)\*\*$", r"\1", cleaned)
            if len(cleaned) > 5:
                items.append(cleaned)

        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in items:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        print(f"    - Extracted {len(unique_items)} key items")
        return unique_items

    # ─── Utility Methods ───────────────────────────────────────────────

    def _programmatic_deduplication(self, items, sim_limit=0.90):
        if not items:
            return []
        embeddings = self.embedding_model.encode(items)
        unique = [items[0]]
        unique_embeddings = [embeddings[0]]
        for i in range(1, len(items)):
            sims = cosine_similarity([embeddings[i]], unique_embeddings)[0]
            if np.max(sims) < sim_limit:
                unique.append(items[i])
                unique_embeddings.append(embeddings[i])
        self._log_key_point_count("After programmatic deduplication", unique, previous_count=len(items))
        return unique[:50]

    _NOISY_PATTERNS = [
        r"\bdie diskussion dreht sich\b",
        r"\bdie frage ist\b",
        r"\bdie frage, ob\b",
        r"\bes wird über\b",
        r"\bes geht um die frage\b",
        r"\bes gibt auch die idee\b",
        r"\bkönnte ein thema sein\b",
        r"\bist ein wichtiger punkt\b",
        r"\bsind ein wichtiger punkt\b",
        r"\bist ein thema\b",
        r"\bsind ein thema\b",
        r"\bschwerpunktthemen\b",
        r"\bes ist wichtig\b",
        r"\bweiteres großes thema\b",
        r"\bist ein großes thema\b",
        r"\boder so\b",
        r"\birgendwie\b",
        r"\bglaube ich\b",
        r"\bso ein bisschen\b",
        r"\bzu vage\b",
        r"\bnein, zu\b",
        r"\bpunkt entfällt\b",
        r"\bgroßes thema\b",
    ]

    def _drop_noisy_items(self, items, max_words=30):
        if not items:
            return []
        cleaned = []
        for item in items:
            low = item.lower()
            if len(low.split()) > max_words:
                continue
            if any(re.search(p, low) for p in self._NOISY_PATTERNS):
                continue
            cleaned.append(item)
        dropped = len(items) - len(cleaned)
        if dropped:
            print(f"      - Dropped {dropped} noisy/fragment items")
        self._log_key_point_count("After noise drop", cleaned, previous_count=len(items))
        return cleaned

    _ARTIFACT_EXACT = {
        "nein",
        "nein, zu vage",
        "punkt entfällt",
        "zu vage",
        "nothing",
    }

    _META_PREFIX = re.compile(r"^(?:Spezifischer\s+Punkt(?:\s*\d+)?\s*:|Spezifisch:\w*|Punkt\s*\d+\s*:|Ja\.?\s*|Nein\.?\s*|Zu\s+vage\.?\s*)+", re.IGNORECASE)

    def _is_truncated(self, text):
        stripped = text.strip()
        if not stripped:
            return True
        if any(ch in stripped for ch in (",", ";", ":", "…", "—", "–", "(", ")")):
            return False
        last_char = stripped[-1]
        has_terminal = last_char in ".!?"
        low = stripped.lower()
        if low in self._ARTIFACT_EXACT:
            return False
        return len(stripped) < 12 and not has_terminal

    def postprocess_key_items(self, items, max_words=28):
        if not items:
            return []
        before = len(items)
        cleaned = []
        for item in items:
            stripped = item.strip()
            if self._META_PREFIX.search(stripped):
                stripped = self._META_PREFIX.sub("", stripped).strip()
            if not stripped:
                continue
            if self._is_truncated(stripped):
                continue
            cleaned.append(stripped)
        cleaned = self._drop_noisy_items(cleaned, max_words=max_words)
        cleaned = self._programmatic_deduplication(cleaned, sim_limit=0.9)
        self._log_key_point_count("After postprocess", cleaned, previous_count=before)
        return cleaned

    def _parse_list(self, content):
        items = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.match(r"^[-*•]\s+", line):
                cleaned = re.sub(r"^[-*•]\s+", "", line)
            elif re.match(r"^\d+[.)]\s+", line):
                cleaned = re.sub(r"^\d+[.)]\s+", "", line)
            else:
                continue
            cleaned = cleaned.strip()
            if len(cleaned) > 5:
                items.append(cleaned)
        return items

    # ─── Atomization ───────────────────────────────────────────────────

    def atomize_key_items(self, items, short=False):
        if not items:
            return []

        before = len(items)
        summary_ref = "150-Wörter-Zusammenfassung" if short else "5%-Zusammenfassung"
        items_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])

        prompt = f"""Du erhältst eine Liste bereits relevanter Key Points aus einem Transkript.

Deine Aufgabe:
Formuliere die Liste so um, dass JEDER Punkt strikt atomar ist — aber NICHT so weit aufteilen, dass Teile allein unverständlich werden.

Eingabeliste:
{items_text}

Regeln:
- Ein Punkt darf genau EINE eigenständige sachliche Aussage enthalten.
- Wenn ein Punkt mehrere UNABHÄNGIGE Aussagen enthält, teile ihn auf.
- NICHT aufteilen, wenn die Teile allein ihren Sinn verlieren. "Kunden haben Bauchschmerzen mit Cloud-Diensten aus den USA" ist EIN Punkt — "Kunden haben Bauchschmerzen" allein ist sinnlos.
- Jeder Punkt MUSS für sich allein verständlich und informativ sein, ohne Kontext.
- Jeder Punkt MUSS mindestens eines enthalten: einen konkreten Akteur, eine Zahl/Datum, ein benanntes Objekt, oder eine spezifische Handlung.
- Verwerfe Punkte, die nach der Aufspaltung nur noch triviale Aussagen sind (z.B. "Die App soll entwickelt werden", "Das Thema wurde besprochen").
- Erhalte den inhaltlichen Umfang der Eingabeliste: Lass keine relevante Aussage weg.
- Erfinde nichts und füge keine Interpretation hinzu.
- Formuliere klar, präzise und überprüfbar.
- Vermeide Meta-Sätze über Diskussionen.
- Jeder Punkt maximal 25 Wörter.
- Gib ausschließlich eine flache Stichpunktliste zurück.
- Jeder Ausgabepunkt MUSS mit der Nummer seines Eingabepunkts in eckigen
  Klammern beginnen, z. B. "- [3] Der Vertrag beginnt am 1. Juli.".
- Falls ein Eingabepunkt aufgeteilt wird, wiederhole seine Nummer für jeden
  Teilpunkt, z. B. "- [3] ..." und "- [3] ...".
- Gib für JEDEN Eingabepunkt mindestens einen Punkt mit seiner Nummer zurück.

Qualitätskriterium:
Die resultierenden Punkte sollen direkt für ein binäres Coverage-Urteil gegen eine {summary_ref} geeignet sein. Ein Punkt, der keine überprüfbare Information enthält, ist unbrauchbar.
"""

        content = self._safe_api_call([
            {"role": "system", "content": "Du zerlegst Aussagen präzise in atomare, überprüfbare Fakten."},
            {"role": "user", "content": prompt}
        ], temperature=0.0, max_tokens=600)

        # Associate every generated point with its source item.  This makes a
        # partial model response detectable: an omitted source item is kept in
        # its original form instead of silently disappearing from the result.
        atomic_by_source = {i: [] for i in range(len(items))}
        if content:
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"^(?:[-*•]|\d+[.)])\s*\[(\d+)\]\s*(.+)$", line)
                if not match:
                    continue
                source_idx = int(match.group(1)) - 1
                point = match.group(2).strip()
                if 0 <= source_idx < len(items) and len(point) > 5:
                    atomic_by_source[source_idx].append(point)

        missing_sources = []
        atomic = []
        for source_idx, source_items in atomic_by_source.items():
            source_items = self._programmatic_deduplication(source_items, sim_limit=0.92)
            source_items = self._drop_noisy_items(source_items, max_words=28)
            if not source_items:
                missing_sources.append(source_idx + 1)
                source_items = [items[source_idx]]
            atomic.extend(source_items)

        if missing_sources:
            print(
                "    [Warning] Atomic rewrite omitted or produced no usable output for "
                f"{len(missing_sources)}/{len(items)} input items "
                f"(indices: {', '.join(map(str, missing_sources))}); keeping originals for them."
            )

        atomic = self._programmatic_deduplication(atomic, sim_limit=0.92)
        atomic = self._drop_noisy_items(atomic, max_words=28)

        if not atomic:
            print("    [Warning] Atomic rewrite returned no usable items. Falling back to pre-atomic list.")
            return items

        print(f"    - Atomic rewrite: {len(items)} -> {len(atomic)} items")
        self._log_key_point_count("After atomization", atomic, previous_count=before)
        return atomic

    # ─── Core Selection ────────────────────────────────────────────────

    def select_core_key_items(self, items, short=False):
        if not items:
            return []

        before = len(items)
        budget = 16 if short else 30
        min_keep = 10
        items_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])

        prompt = f"""Du erhältst atomare, bereits relevante Key Points aus einem Sitzungstranskript.

Deine Aufgabe:
Wähle ALLE Punkte aus, die WICHTIG sind und mit hoher Wahrscheinlichkeit in einer {('150-Wörter-Zusammenfassung' if short else '5%-Zusammenfassung')} enthalten wären. Sei eher großzügig: Behalte einen Punkt, wenn berechtigte Zweifel bestehen, ob er weggelassen werden dürfte.

Eingabeliste:
{items_text}

Priorität (von hoch nach niedrig):
1. Konkrete Entscheidungen und Beschlüsse mit Ergebnis
2. Spezifische Fakten mit Zahlen, Daten, Geldbeträgen
3. Positionen benannter Akteure/Fraktionen (wer fordert/blockiert/unterstützt was)
4. Konflikte und Meinungsverschiedenheiten zwischen Akteuren
5. Konkrete nächste Schritte oder Maßnahmen

Streiche NUR:
- Klar redundante Doppelungen
- Historischen Kontext, der für das Ergebnis nicht entscheidend ist
- Allgemeine Absichtserklärungen ohne konkreten Inhalt

Behalte zusammengehörige Punkte auch dann, wenn sie ähnlich sind – das Zusammenfassen nah verwandter Punkte übernimmt eine spätere Clustering-Stufe.

Zielgröße: Höchstens {budget} Punkte.
Gib ausschließlich eine flache Stichpunktliste zurück.
"""

        content = self._safe_api_call([
            {"role": "system", "content": "Du reduzierst auf die absolut unverzichtbaren Kernaussagen."},
            {"role": "user", "content": prompt}
        ], temperature=0.0, max_tokens=400)

        selected = self._parse_list(content) if content else []
        selected = self._programmatic_deduplication(selected, sim_limit=0.92)
        selected = self._drop_noisy_items(selected, max_words=28)

        if not selected:
            print("    [Warning] Core selection returned no parseable items. Falling back to atomic list.")
            return items[:budget]

        # if len(selected) < min_keep:
        #     added = 0
        #     for item in items:
        #         if len(selected) >= min_keep:
        #             break
        #         if item not in selected:
        #             selected.append(item)
        #             added += 1
        #     if added:
        #         print(f"    - Core selection below {min_keep}; topped up with {added} items from atomic list")

        if len(selected) > budget:
            selected = selected[:budget]

        print(f"    - Core selection: {len(items)} -> {len(selected)} items (budget {budget})")
        self._log_key_point_count("After core selection", selected, previous_count=before)
        return selected

    # ─── Clustering ────────────────────────────────────────────────────

    def _cluster_key_items(self, items, sim_threshold=0.93):
        if not items:
            return []

        try:
            embeddings = self.embedding_model.encode(items)
        except Exception as e:
            print(f"    [Warning] Clustering embeddings failed: {e}")
            return items

        clusters = []
        cluster_centroids = []

        for idx, item in enumerate(items):
            if not clusters:
                clusters.append([item])
                cluster_centroids.append(embeddings[idx])
                continue

            sims = cosine_similarity([embeddings[idx]], [cluster_centroids[i] for i in range(len(cluster_centroids))])[0]
            best_cluster = int(np.argmax(sims))
            if sims[best_cluster] >= sim_threshold:
                clusters[best_cluster].append(item)
                cluster_centroids[best_cluster] = np.mean(self.embedding_model.encode(clusters[best_cluster]), axis=0)
            else:
                clusters.append([item])
                cluster_centroids.append(embeddings[idx])

        representatives = [min(cluster, key=lambda text: len(text.split())) for cluster in clusters]
        if len(representatives) != len(items):
            print(f"    - Clustered {len(items)} -> {len(representatives)} items")
        self._log_key_point_count("After clustering", representatives, previous_count=len(items))
        return representatives

    # ─── Transcript Quality Detection ────────────────────────────────────

    def _assess_transcript_quality(self, transcript):
        words = transcript.split()
        total_words = len(words)
        if total_words == 0:
            return 'low', 1.0

        filler_words = {'also', 'halt', 'nochmal', 'einfach', 'irgendwie', 'sozusagen',
                        'quasi', 'eigentlich', 'eben', 'na', 'gut', 'genau', 'okay',
                        'ähm', 'äh', 'hmm', 'mhm', 'bisschen', 'irgendwas', 'irgendein',
                        'ne', 'ja', 'naja'}
        lower_words = [w.lower().strip('.,;:!?') for w in words]
        filler_count = sum(1 for w in lower_words if w in filler_words)
        filler_ratio = filler_count / total_words

        ellipsis_count = transcript.count('...') + transcript.count('…')
        ellipsis_per_100w = ellipsis_count / (total_words / 100)

        sentence_count = max(1, transcript.count('.') + transcript.count('!') + transcript.count('?'))
        avg_sentence_len = total_words / sentence_count

        speakers = set(re.findall(r'\[(Speaker \d+)\]|\b(Sprecher \d+)\b', transcript))
        num_speakers = len(speakers)
        turns = len(re.findall(r'\[(Speaker \d+)\]:|\bSprecher \d+\b', transcript))
        avg_turn_len = total_words / max(1, turns)

        if filler_ratio > 0.08 or ellipsis_per_100w > 2.0 or avg_turn_len < 40:
            quality = 'low'
        elif filler_ratio > 0.04 or ellipsis_per_100w > 1.0:
            quality = 'medium'
        else:
            quality = 'high'

        noise_score = round(filler_ratio * 5 + ellipsis_per_100w * 0.3, 2)
        return quality, noise_score

    # ─── Source Verification ───────────────────────────────────────────

    def _verify_against_source(self, items, transcript, threshold=0.55):
        if not items:
            return []

        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return items

        try:
            item_embeddings = self.embedding_model.encode(items)
            source_embeddings = self.embedding_model.encode(sentences)
        except Exception as e:
            print(f"    [Warning] Verification embedding failed: {e}")
            return items

        sims = cosine_similarity(item_embeddings, source_embeddings)
        max_sims = sims.max(axis=1)

        verified = []
        dropped = []
        for i, (item, sim) in enumerate(zip(items, max_sims)):
            if sim >= threshold:
                verified.append(item)
            else:
                dropped.append((item, float(sim)))

        if dropped:
            print(f"    - Verification dropped {len(dropped)} ungrounded items:")
            for item, sim in dropped:
                print(f"      [{sim:.2f}] {item[:80]}")

        self._log_key_point_count("After verification", verified, previous_count=len(items))
        return verified

    # ─── Main Pipeline ─────────────────────────────────────────────────

    def _run_extraction_pipeline(self, transcript, temperature=0.0, short=False):
        turns = self.parse_transcript_text(transcript)
        if not turns:
            return []

        self._log_key_point_count("Parsed transcript turns", turns)

        batched_texts = []
        current_text, current_tokens = "", 0
        for t in turns:
            ts = t['timestamp']
            if ts.startswith("Sent-"):
                turn_text = f"{t['speaker']}: {t['text']}\n"
            else:
                turn_text = f"[{ts}] {t['speaker']}: {t['text']}\n"
            turn_tokens = self.count_tokens(turn_text)
            if current_tokens + turn_tokens > 3000 and current_text:
                batched_texts.append(current_text)
                current_text, current_tokens = turn_text, turn_tokens
            else:
                current_text += turn_text
                current_tokens += turn_tokens
        if current_text:
            batched_texts.append(current_text)

        print(f"      - Processing {len(batched_texts)} batches for extraction...")
        all_items = []
        for i, batch_text in enumerate(batched_texts):
            print(f"      - Extraction batch {i+1}/{len(batched_texts)}...")
            batch_items = self.extract_key_items(
                batch_text, short=short,
                previously_extracted=all_items if all_items else None
            )
            self._log_key_point_count(f"Batch {i+1} extracted", batch_items)
            all_items.extend(batch_items)
            self._log_key_point_count(f"Accumulated after batch {i+1}", all_items)

        if not all_items:
            return []
        deduped = self._programmatic_deduplication(all_items)
        self._log_key_point_count("After batch aggregation dedup", deduped, previous_count=len(all_items))
        final_items = self._drop_noisy_items(deduped)
        self._log_key_point_count("After batch aggregation noise drop", final_items, previous_count=len(deduped))
        return final_items

    def get_key_items_for_transcript(self, transcript, filename, short=False):
        if filename in self.kp_cache:
            return self.kp_cache[filename]

        print(f"  [Step 1] Extracting key items for {filename}...")

        quality, noise_score = self._assess_transcript_quality(transcript)
        print(f"    - Transcript quality: {quality} (noise={noise_score})")

        if quality == 'low':
            print(f"    - Low-quality transcript: routing to statistical pipeline...")
            items = self._extract_statistical(transcript, filename)
            self._log_key_point_count("Statistical pipeline output", items)
            self.kp_cache[filename] = items
            return items

        print(f"    - Extraction run (LLM entailment)...")
        items = self._run_extraction_pipeline(transcript, temperature=0.0, short=short)

        if not items:
            print(f"  [Warning] Extraction returned no items.")
            self.kp_cache[filename] = []
            return []

        print(f"    - Raw items: {len(items)}")

        print(f"    - Verifying against source transcript...")
        verify_threshold = 0.55 if quality == 'high' else 0.65
        items = self._verify_against_source(items, transcript, threshold=verify_threshold)

        if not items:
            print(f"  [Warning] All items failed verification.")
            self.kp_cache[filename] = []
            return []

        print(f"    - After verification: {len(items)} items")

        before_atomize = len(items)
        items = self.atomize_key_items(items, short=short)
        self._log_key_point_count("Pipeline after atomization", items, previous_count=before_atomize)
        items = self._cluster_key_items(items, sim_threshold=0.88)
        items = self.select_core_key_items(items, short=short)

        print(f"    - Enriching vague points...")
        before_enrich = len(items)
        items = self._enrich_vague_points(items, transcript)
        self._log_key_point_count("After enrichment", items, previous_count=before_enrich)
        print(f"    - Re-verifying after enrichment...")
        before_reverify = len(items)
        items = self._verify_against_source(items, transcript, threshold=verify_threshold)
        self._log_key_point_count("After re-verification", items, previous_count=before_reverify)
        before_postprocess = len(items)
        items = self.postprocess_key_items(items)
        self._log_key_point_count("Pipeline after postprocess", items, previous_count=before_postprocess)
        print(f"    - Final: {len(items)} points")

        self.kp_cache[filename] = items
        return items

    def _enrich_vague_points(self, items, transcript, num_context_sentences=5):
        if not items:
            return []

        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return items

        try:
            item_embeddings = self.embedding_model.encode(items)
            source_embeddings = self.embedding_model.encode(sentences)
        except Exception:
            return items

        sims = cosine_similarity(item_embeddings, source_embeddings)

        enriched = []
        points_to_enrich = []
        context_per_point = []

        for i, item in enumerate(items):
            top_indices = sims[i].argsort()[-num_context_sentences:][::-1]
            top_sents = [sentences[idx] for idx in top_indices]
            # Truncate sentences to max 150 chars to reduce token count
            top_sents = [s[:150] + "..." if len(s) > 150 else s for s in top_sents]
            points_to_enrich.append(item)
            context_per_point.append(top_sents)

        items_with_context = "\n\n".join([
            f"Punkt {i+1}: {point}\nRelevante Stellen:\n" + "\n".join(f"  - {s}" for s in ctx)
            for i, (point, ctx) in enumerate(zip(points_to_enrich, context_per_point))
        ])

        prompt = f"""Du erhältst Kernpunkte aus einem Transkript. Manche sind zu vage — es fehlt der Akteur, das konkrete Objekt oder der Kontext.

{items_with_context}

Aufgabe:
- Prüfe jeden Punkt: Ist er spezifisch genug, um ohne Kontext verständlich zu sein?
- Falls JA: Übernimm ihn unverändert.
- Falls NEIN: Ergänze den fehlenden Akteur, das konkrete Objekt oder die fehlende Spezifikation aus den relevanten Stellen.
- Ergänze NUR Informationen, die explizit in den relevanten Stellen stehen.
- Wenn die fehlende Information nicht auffindbar ist, lass den Punkt unverändert.
    - Gib für JEDEN Eingabepunkt genau eine Zeile zurück.
    - Nutze exakt dieses Format: [1] ..., [2] ..., [3] ...
    - Behalte die Reihenfolge und die Indizes der Eingabeliste exakt bei.
- Jeder Punkt maximal 30 Wörter.
- Format: flache Stichpunktliste."""

        try:
            content = self._safe_api_call([
                {"role": "system", "content": "Du präzisierst vage Aussagen mit Informationen aus dem Transkript."},
                {"role": "user", "content": prompt}
            ], temperature=0.0, max_tokens=500)
        except SystemExit as e:
            # Check if it's a context length error
            error_msg = str(e).lower()
            if "input_tokens" in error_msg and num_context_sentences > 1:
                print(f"      - Context too long with {num_context_sentences} sentences; retrying with {num_context_sentences - 1}...")
                return self._enrich_vague_points(items, transcript, num_context_sentences=num_context_sentences - 1)
            raise

        if not content:
            return items

        enriched_by_index = {}
        parsed_in_order = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^\[(\d+)\]\s*(.+)$", line)
            if not match:
                match = re.match(r"^(\d+)[.)]\s*(.+)$", line)
            if not match:
                continue
            payload = match.group(2).strip()
            parsed_in_order.append(payload)

            idx = int(match.group(1)) - 1
            if 0 <= idx < len(items) and idx not in enriched_by_index:
                enriched_by_index[idx] = payload

        if not enriched_by_index:
            print(f"      - Enrichment returned no indexed items; keeping original items")
            return items

        merged = []
        changed = 0
        used_payloads = set()
        for idx, original in enumerate(items):
            replacement = enriched_by_index.get(idx)
            if replacement:
                merged.append(replacement)
                used_payloads.add(replacement)
                if replacement != original:
                    changed += 1
            else:
                merged.append(original)

        missing_slots = [i for i, value in enumerate(merged) if value == items[i]]
        if missing_slots:
            fallback_payloads = [p for p in parsed_in_order if p not in used_payloads]
            for idx, payload in zip(missing_slots, fallback_payloads):
                merged[idx] = payload
                used_payloads.add(payload)
                if payload != items[idx]:
                    changed += 1

        if len(enriched_by_index) != len(items) or len(parsed_in_order) != len(items):
            missing = len(items) - len(enriched_by_index)
            print(
                f"      - Enrichment returned {len(enriched_by_index)} indexed items for {len(items)} inputs; "
                f"kept {missing} original items, then filled {len(missing_slots)} slots using output order fallback"
            )

        if changed:
            print(f"      - Enriched {changed}/{len(items)} points")

        return merged

    def _extract_statistical(self, transcript, filename):
        from extract_keypoints_statistical import StatisticalKeyPointExtractor

        if not hasattr(self, '_stat_extractor'):
            self._stat_extractor = StatisticalKeyPointExtractor(
                embedding_model_name='intfloat/multilingual-e5-large',
                use_llm_compression=True,
            )
            self._stat_extractor.embedding_model = self.embedding_model

        result = self._stat_extractor.extract_key_points(transcript, filename)
        points = result.get('key_points', [])
        self._log_key_point_count("Statistical extractor raw output", points)

        if points:
            print(f"    - Filtering {len(points)} statistical points with LLM...")
            points = self._llm_filter_informative(points)
            print(f"    - After LLM filter: {len(points)} points")
            self._log_key_point_count("After LLM filter", points)

        return points

    def _llm_filter_informative(self, points):
        if not points:
            return []

        before = len(points)

        items_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(points)])

        prompt = f"""Welche dieser Sätze enthalten eine konkrete Sachaussage?

Sätze:
{items_text}

Streiche Sätze, die:
- Eine Frage stellen
- Nur das Gespräch moderieren oder steuern
- Nur persönliche Kommentare oder Anekdoten sind
- Keine eigenständige Information vermitteln

Antworte NUR mit den Nummern der Sätze die eine Sachaussage enthalten, kommagetrennt."""

        content = self._safe_api_call([
            {"role": "system", "content": "Du filterst informative Aussagen. Antworte nur mit Nummern."},
            {"role": "user", "content": prompt}
        ], temperature=0.0, max_tokens=100)

        if not content:
            return points

        try:
            indices = [int(x.strip()) - 1 for x in re.findall(r'\d+', content)]
            indices = [i for i in indices if 0 <= i < len(points)]
            if indices:
                filtered = [points[i] for i in indices]
                self._log_key_point_count("LLM informative filter", filtered, previous_count=before)
                return filtered
        except (ValueError, IndexError):
            pass

        self._log_key_point_count("LLM informative filter", points, previous_count=before)
        return points


# ─── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    import argparse

    parser = argparse.ArgumentParser(description="Extract key points using LLM entailment approach")
    parser.add_argument("--csv", type=str, default="../pilot_summaries.csv", help="Path to CSV file")
    parser.add_argument("--output-dir", type=str, default="entailment_keypoints_results", help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit rows to process")
    parser.add_argument("--filenames", type=str, nargs="+", help="Specific filenames to process")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    parser.add_argument("--model", type=str, default="gemma-4-31b-it", help="LLM model name")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    extractor = EntailmentKeyPointExtractor(model_name=args.model)

    import atexit
    def print_usage():
        pt = extractor.total_prompt_tokens
        ct = extractor.total_completion_tokens
        if pt + ct > 0:
            print(f"\n=== Token Usage ===")
            print(f"  Prompt tokens:     {pt:>8}")
            print(f"  Completion tokens: {ct:>8}")
            print(f"  Total tokens:      {pt + ct:>8}")
    atexit.register(print_usage)

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        exit(1)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(args.output_dir, f"entailment_kp_{base_name}.json")

    results = []
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                print(f"Loaded {len(results)} existing results")
        except Exception:
            pass

    if args.force:
        if args.filenames:
            filenames_set = set(args.filenames)
            results = [r for r in results if r.get('filename') not in filenames_set]
        else:
            results = []

    processed_ids = {f"{r['filename']}_{r.get('profile', '')}_{r.get('model', '')}" for r in results}

    try:
        df = pd.read_csv(csv_path, sep=';')
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        exit(1)

    if args.filenames:
        df = df[df['filename'].isin(args.filenames)]

    mask = df.apply(
        lambda r: f"{r['filename']}_{r.get('profile', '')}_{r.get('model', '')}" not in processed_ids, axis=1
    )
    df_to_process = df[mask]

    if args.limit:
        df_to_process = df_to_process.head(args.limit)

    if len(df_to_process) == 0:
        print(f"No new rows to process in {base_name}.")
        exit(0)

    print(f"\nProcessing {len(df_to_process)} rows from {base_name}...")

    keypoints_cache = {}

    for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        filename = row['filename']
        profile = row.get('profile', '')
        model = row.get('model', '')
        transcript = str(row['transcript'])
        summary = str(row.get('summary', ''))

        print(f"\n{'='*60}")
        print(f"Extracting key points: {filename} ({model}, {profile})")
        print(f"{'='*60}")

        short = len(transcript.split()) > 2000

        if filename in keypoints_cache:
            print(f"  [Cache] Reusing key points from previous extraction of {filename}")
            key_points = keypoints_cache[filename]
        else:
            key_points = extractor.get_key_items_for_transcript(transcript, filename, short=short)
            keypoints_cache[filename] = key_points

        result = {
            'key_points': key_points,
            'filename': filename,
            'profile': profile,
            'model': model,
            'summary': summary[:200] + "..." if len(summary) > 200 else summary,
            'metadata': {
                'num_key_points': len(key_points),
                'word_count': len(transcript.split()),
                'short_mode': short,
            }
        }

        results.append(result)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
