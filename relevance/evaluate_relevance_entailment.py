import pandas as pd
import numpy as np
import re
import os
import json
import tiktoken
import time
import random
import traceback
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

class EntailmentRelevanceEvaluator:
    def __init__(self, model_name="meta-llama-3.1-8b-instruct"):
        self.api_key = os.environ.get("ACADEMIC_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("Set the ACADEMIC_CLOUD_API_KEY environment variable before running this evaluator.")
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

    def _safe_api_call(self, messages, temperature=0.0, max_attempts=5, timeout=120, max_tokens=None):
        """Call the API, backing off between retries. If it still fails after
        ``max_attempts``, stop the entire program (raise SystemExit) instead of
        silently skipping. SystemExit is a BaseException, so the broad
        `except Exception` handlers elsewhere never swallow it."""
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
                print(f"    [API Error] Empty response body (attempt {attempt+1}/{max_attempts}); waiting and retrying...")
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                print(
                    f"    [API Error] {type(e).__name__}: {e!r} "
                    f"(attempt {attempt+1}/{max_attempts})"
                )
                traceback.print_exc()
                if "429" in error_msg:
                    wait = 5 + 3 * attempt + random.uniform(0, 2)
                    print(f"    [Rate Limit] Waiting {wait:.0f}s then retrying...")
                elif any(s in error_msg for s in ("500", "502", "503", "timeout", "timed out", "connection", "connect")):
                    wait = min(60, 5 * (2 ** attempt)) + random.uniform(0, 1)
                    print(f"    [Server Error] Retrying in {wait:.0f}s...")
                else:
                    wait = 2 + 2 * attempt + random.uniform(0, 1)
                    print(f"    [API Error] Retrying in {wait:.0f}s...")
                time.sleep(wait)

        error_detail = f"{type(last_error).__name__}: {last_error!r}"
        raise SystemExit(
            f"\n[FATAL] API call kept failing after {max_attempts} attempts "
            f"(model={self.model_name}, base_url={self.base_url}). Last error: {error_detail}. "
            f"Aborting the whole run to avoid silently skipping key items."
        )

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

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

    def extract_key_items(self, text, short=False):
        length_desc = "von höchstens 150 Wörtern" if short else "von höchstens 5 % der ursprünglichen Textlänge"
        summary_ref = "150-Wörter-Zusammenfassung" if short else "5%-Zusammenfassung"

        prompt = f"""
Du liest das folgende Sitzungsprotokoll.

Deine Aufgabe ist es, ausschließlich die Kernaussagen zu extrahieren, die mit hoher Wahrscheinlichkeit auch in einer sehr knappen Zusammenfassung {length_desc} enthalten wären.

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
- Hintergrundinformationen, sofern sie nicht entscheidend sind
- einzelne Beispiele
- Wiederholungen
- Erläuterungen ohne neue Information
- Fakten, Daten oder Zahlen, die für die Kernaussage nicht notwendig sind

Regeln:
- Gib ausschließlich eine flache Liste von Stichpunkten aus.
- Keine Überschriften oder Kategorien.
- Jeder Stichpunkt enthält genau EINE atomare Kernaussage, also genau eine einzelne, nicht weiter teilbare Information.
- Wenn ein Satz mehrere unabhängige Fakten, Entscheidungen oder Maßnahmen enthält, trenne sie in mehrere Stichpunkte.
- Verknüpfe keine unabhängigen Aussagen mit „und“, wenn sie auch getrennt stehen können.
- Extrahiere alle Punkte, die mit hoher Wahrscheinlichkeit in einer {summary_ref} vorkommen würden – sei dabei vollständig, aber nimm keine irrelevanten Punkte auf.
- Formuliere jeden Punkt als klare, eigenständige Aussage (Subjekt–Prädikat–Objekt), die für sich allein in einer Zusammenfassung stehen kann.
- Paraphrasiere in sauberes Deutsch: Übernimm keine wörtlichen Fragmente des Redners, keine Satzabbrüche, kein Stocken, keine Füllwörter oder Abschweifungen.
- Jeder Punkt besteht aus höchstens 25 Wörtern. Lange Ausführungen werden auf ihren Kern gekürzt.
- Vermeide unscharfe Formulierungen wie „oder so", „irgendwie", „so ein bisschen", „glaube ich". Wenn eine Aussage zu vage ist, formuliere sie präzise oder verwerfe sie.
- Verboten sind Meta-Sätze über den Gesprächsverlauf wie „Die Diskussion dreht sich um …", „Es wird über … gesprochen", „Die Diskussion geht auf … ein", „Es gibt auch die Idee, …". Jeder Punkt muss die Information selbst enthalten, nicht nur, dass das Thema vorkam.
- Verboten sind Frage- und Unsicherheits-Rahmen wie „Die Frage ist, ob …", „Die Frage ist, wie …". Formuliere die Aussage direkt, ohne den Rahmen.
- Vermeide Floskeln wie „ist ein wichtiger Punkt", „sind ein wichtiger Punkt".
- Wenn eine Stelle unklar, mehrdeutig oder schwer verständlich ist, verwerfe sie. Raten und Erfinden ist verboten.
- Korrigiere offensichtliche Transkriptionsfehler (z. B. „Gand-Chart" → „Gantt-Chart"), wenn der korrekte Begriff klar erkennbar ist.
- Füge keine Interpretationen, Wertungen oder zusätzlichen Informationen hinzu.

Beispiel (NUR zur Veranschaulichung, nicht aus dem Transkript – übernimm es NICHT):
- „Neue Komponente B soll in die bestehende Plattform integriert werden."

Dokument:
{text}
"""
        content = self._safe_api_call([
            {"role": "system", "content": "Du bist ein hilfreicher Assistent, der präzise und sachlich analysiert."},
            {"role": "user", "content": prompt}
            ], max_tokens=300)

        if not content:
            print("    [Warning] API returned empty content for key item extraction.")
            return []

        items = []

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue

            # Markdown bullets: -, *, •
            if re.match(r"^[-*•]\s+", line):
                cleaned = re.sub(r"^[-*•]\s+", "", line)

            # Numbered lists: 1. / 1)
            elif re.match(r"^\d+[.)]\s+", line):
                cleaned = re.sub(r"^\d+[.)]\s+", "", line)

            else:
                continue

            cleaned = cleaned.strip()

            # Remove surrounding markdown bold if present
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

    def _programmatic_deduplication(self, items, sim_limit=0.85):
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

    def atomize_key_items(self, items, short=False):
        if not items:
            return []

        summary_ref = "150-Wörter-Zusammenfassung" if short else "5%-Zusammenfassung"
        items_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])

        prompt = f"""Du erhältst eine Liste bereits relevanter Key Points aus einem Transkript.

Deine Aufgabe:
Formuliere die Liste so um, dass JEDER Punkt strikt atomar ist.

Eingabeliste:
{items_text}

Regeln:
- Ein Punkt darf genau EINE eigenständige sachliche Aussage enthalten.
- Wenn ein Punkt mehrere Aussagen enthält, MUSST du ihn in mehrere Punkte aufteilen.
- Erhalte den inhaltlichen Umfang der Eingabeliste: Lass keine relevante Aussage weg.
- Erfinde nichts und füge keine Interpretation hinzu.
- Formuliere klar, präzise und überprüfbar.
- Vermeide Meta-Sätze über Diskussionen.
- Jeder Punkt maximal 25 Wörter.
- Gib ausschließlich eine flache Stichpunktliste zurück.

Qualitätskriterium:
Die resultierenden Punkte sollen direkt für ein binäres Coverage-Urteil gegen eine {summary_ref} geeignet sein.
"""

        content = self._safe_api_call([
            {"role": "system", "content": "Du zerlegst Aussagen präzise in atomare, überprüfbare Fakten."},
            {"role": "user", "content": prompt}
        ], temperature=0.0)

        atomic = self._parse_list(content) if content else []
        atomic = self._programmatic_deduplication(atomic, sim_limit=0.92)
        atomic = self._drop_noisy_items(atomic, max_words=28)

        if not atomic:
            print("    [Warning] Atomic rewrite returned no parseable items. Falling back to pre-atomic list.")
            return items

        print(f"    - Atomic rewrite: {len(items)} -> {len(atomic)} items")
        return atomic

    def select_core_key_items(self, items, short=False):
        if not items:
            return []

        budget = 12 if short else 18
        items_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])

        prompt = f"""Du erhältst atomare, bereits relevante Key Points.

Deine Aufgabe:
Wähle nur die Punkte aus, die für eine {('150-Wörter-Zusammenfassung' if short else '5%-Zusammenfassung')} wirklich unverzichtbar sind.

Eingabeliste:
{items_text}

Regeln:
- Sei streng: Behalte nur Punkte, deren Weglassen die Zusammenfassung klar unvollständig oder irreführend machen würde.
- Behalte keine bloß nützlichen, wiederholten oder randständigen Punkte.
- Wenn zwei Punkte sich überlappen, behalte die präzisere und wichtigere Version.
- Zielgröße: höchstens {budget} Punkte.
- Wenn mehr als {budget} Punkte scheinbar wichtig sind, behalte trotzdem nur die absolut zentralen.
- Gib ausschließlich eine flache Stichpunktliste zurück.
"""

        content = self._safe_api_call([
            {"role": "system", "content": "Du reduzierst auf die absolut unverzichtbaren Kernaussagen."},
            {"role": "user", "content": prompt}
        ], temperature=0.0, max_tokens=180)

        selected = self._parse_list(content) if content else []
        selected = self._programmatic_deduplication(selected, sim_limit=0.92)
        selected = self._drop_noisy_items(selected, max_words=28)

        if not selected:
            print("    [Warning] Core selection returned no parseable items. Falling back to atomic list.")
            return items[:budget]

        if len(selected) > budget:
            selected = selected[:budget]

        print(f"    - Core selection: {len(items)} -> {len(selected)} items (budget {budget})")
        return selected

    def filter_for_5_percent_summary(self, items, short=False, tags=None):
        if not items:
            return []

        length_label = "maximal 150 Wörter" if short else "höchstens 5 % der ursprünglichen Textlänge"
        summary_ref = "150-Wörter-Zusammenfassung" if short else "5%-Zusammenfassung"

        items_text = "\n".join([
            f"{i+1}. {item}{'  [nur in einem Lauf gefunden]' if tags and tags.get(item) == 'one' else ''}"
            for i, item in enumerate(items)
        ])

        theme_prompt = f"""Aus diesen extrahierten Punkten:

{items_text}

Was ist das KERNTHEMA dieser Diskussion in EINEM SATZ?
Antworte mit maximal 15 Wörtern.
"""

        theme = self._safe_api_call([
            {"role": "system", "content": "Du identifizierst Kernaussagen prägnant."},
            {"role": "user", "content": theme_prompt}
        ], temperature=0.0, max_tokens=60)

        if not theme:
            theme = "die Diskussion über den Gedenkort"

        filter_prompt = f"""Die folgenden Punkte wurden aus einem Transkript extrahiert.
Das KERNTHEMA der Diskussion ist: {theme}

Hier sind alle extrahierten Punkte:

{items_text}

**Aufgabe:**
Behalte NUR die PUNKTE, die für das Verständnis von "{theme}" am WICHTIGSTEN sind.

**Hinweis zu einzelnen Punkten:**
Punkte mit dem Zusatz „[nur in einem Lauf gefunden]" wurden nur in einer von zwei unabhängigen Extraktionen gefunden. Behandle sie mit besonderer Skepsis: Streiche sie, wenn auch nur Zweifel an ihrer Wichtigkeit für die {summary_ref} bestehen.

**Ein Punkt ist WICHTIG, wenn er:**
- Den Kernkonflikt oder die Hauptentscheidung erklärt
- Eine Hauptforderung oder Kernposition enthält
- Entscheidende Fakten (Zahlen, Zeitpläne, konkrete Vereinbarungen) nennt
- Ein verbindliches Ergebnis, eine Entscheidung oder einen Lösungsansatz beschreibt

**Ein Punkt ist WENIGER WICHTIG, wenn er:**
- Logistische, organisatorische oder administrative Details enthält (z. B. wo eine Aufzeichnung abgelegt wird, wer etwas organisiert, Terminabsprachen, interne Arbeitsabläufe)
- Eine generische Absichtserklärung ohne konkrete Information ist (z. B. "Anmerkungen sollten berücksichtigt werden", "man sollte die Lösung verbessern", "die Lösung sollte attraktiv sein")
- Nur beschreibt, dass ein Thema besprochen wurde, ohne den Inhalt zu nennen (z. B. "Es wird über X gesprochen", "Die Diskussion dreht sich um X")
- Historischen Kontext oder Hintergrundinformation liefert, die nicht für das Verständnis des aktuellen Themas notwendig ist
- Ein Nebenaspekt oder eine Randbemerkung ist

**Zwingende Regel:**
Trifft ein Punkt auf EINE der Kategorien „WENIGER WICHTIG" zu, MUSS er gestrichen werden – auch wenn er zum Thema passt.

Beispiele für Punkte, die zwingend zu streichen sind:
- „Die Lösung sollte attraktiv sein, um den Aufwand zu minimieren."
- „Das Tempo sollte angezogen werden."
- „Die Diskussion dreht sich um die Einführung des neuen Dienstes."

Die {summary_ref} ist sehr kurz. Es passen NUR die wichtigsten Punkte hinein.

**Prüfe für JEDEN Punkt: "Würde die {summary_ref} ohne diesen Punkt offensichtlich unvollständig oder irreführend?"**
- Wenn NEIN → weglassen. Ein Punkt, dessen Fehlen niemand bemerken würde, gehört NICHT hinein.
- Wenn JA → behalten

**Gib ausschließlich die REDUZIERTE Liste zurück (nur die Punkte, die bleiben sollen).**
**Keine Erklärungen, keine Bewertungen.**
"""

        content = self._safe_api_call([
            {"role": "system", "content": "Du reduzierst auf das Wesentliche."},
            {"role": "user", "content": filter_prompt}
        ], temperature=0.0, max_tokens=250)

        filtered = self._parse_list(content)
        filtered = [re.sub(r"\s*\[nur in einem Lauf gefunden\]\s*$", "", item).strip() for item in filtered]
        filtered = self._programmatic_deduplication(filtered, sim_limit=0.90)
        filtered = self._drop_noisy_items(filtered)

        if not filtered:
            raise ValueError(f"filter_for_5_percent_summary produced 0 items from {len(items)} — LLM response could not be parsed")

        print(f"    - Theme: {theme}")
        print(f"    - Filtered from {len(items)} to {len(filtered)} items")
        return filtered

    def _run_extraction_pipeline(self, transcript, temperature=0.0, short=False, perspective=None):
        turns = self.parse_transcript_text(transcript)
        if not turns:
            return []

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
            all_items.extend(self.extract_key_items(batch_text, short=short))

        if not all_items:
            return []
        deduped = self._programmatic_deduplication(all_items)
        return self._drop_noisy_items(deduped)

    def _union_and_dedup_key_items(self, run1, run2, emb_threshold=0.90):
        merged = run1 + run2

        deduped = []
        kept = []
        try:
            if merged:
                embeddings = self.embedding_model.encode(merged)
                for i, item in enumerate(merged):
                    if i == 0:
                        deduped.append(item)
                        kept.append(embeddings[i])
                        continue
                    sims = cosine_similarity([embeddings[i]], kept)[0]
                    if np.max(sims) < emb_threshold:
                        deduped.append(item)
                        kept.append(embeddings[i])
        except Exception as e:
            print(f"    [Warning] Embedding dedup failed: {e}")
            deduped = merged

        tags = {}
        try:
            if run1 and run2 and deduped:
                emb1 = self.embedding_model.encode(run1)
                emb2 = self.embedding_model.encode(run2)
                emb_d = self.embedding_model.encode(deduped)
                max_s1 = np.max(cosine_similarity(emb_d, emb1), axis=1)
                max_s2 = np.max(cosine_similarity(emb_d, emb2), axis=1)
                for item, s1, s2 in zip(deduped, max_s1, max_s2):
                    tags[item] = "both" if (s1 >= emb_threshold and s2 >= emb_threshold) else "one"
        except Exception as e:
            print(f"    [Warning] Run-source tagging failed: {e}")

        return deduped[:50], tags

    def _cluster_key_items(self, items, sim_threshold=0.88):
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
        return representatives

    def get_key_items_for_transcript(self, transcript, filename, short=False):
        if filename in self.kp_cache:
            return self.kp_cache[filename]

        print(f"  [Step 1] Extracting key items for {filename}...")

        print(f"    - Single extraction run (no focus split)...")
        run1 = self._run_extraction_pipeline(transcript, temperature=0.0, short=short)
        run2 = []

        if not run1:
            print(f"  [Warning] Extraction returned no items.")
            self.kp_cache[filename] = []
            return []

        merged, tags = self._union_and_dedup_key_items(run1, run2)

        if not merged:
            merged = run1
            tags = {}

        print(f"    - Run1: {len(run1)}, Run2: {len(run2)}, Union: {len(merged)}")

        merged = self.filter_for_5_percent_summary(merged, short=short, tags=tags)
        merged = self.atomize_key_items(merged, short=short)
        # merged = self._cluster_key_items(merged, sim_threshold=0.88)
        # merged = self.select_core_key_items(merged, short=short)

        self.kp_cache[filename] = merged
        return merged

    def _reconsider_not_covered(self, key_point, summary, first_reason):
        prompt = f"""Du überprüfst eine bereits getroffene NOT_COVERED-Entscheidung.

Key Point:
{key_point}

Zusammenfassung:
{summary}

Erste Begründung:
{first_reason}

Aufgabe:
Prüfe streng, ob die NOT_COVERED-Entscheidung inhaltlich wirklich korrekt ist.

Regeln:
- Vergleiche Bedeutung, nicht Wortlaut.
- Wenn dieselbe konkrete Aussage in der Zusammenfassung klar erkennbar ist (auch paraphrasiert), MUSS das Urteil COVERED sein.
- Unterschiedliche Rollen-/Aliasbezeichnungen für denselben Akteur sind zulässig.
- Lehne NICHT nur deshalb ab, weil eine Formulierung nicht wörtlich vorkommt.
- Bleibe nur dann bei NOT_COVERED, wenn die konkrete Aussage tatsächlich fehlt oder nur schwach impliziert ist.

Antworte ausschließlich im Format:
COVERED, weil ...
oder
NOT_COVERED, weil ...
"""

        content = self._safe_api_call([
            {"role": "system", "content": "Du bist ein strenger Qualitätsprüfer für semantische Abdeckung."},
            {"role": "user", "content": prompt}
        ], temperature=0.0)

        if not content:
            return None, ""

        text = content.strip()
        upper = text.upper()
        if re.search(r'\bCOVERED\b', upper) and not re.search(r'\bNOT_COVERED\b', upper):
            return True, text
        if re.search(r'\bNOT_COVERED\b', upper):
            return False, text
        return None, text

    def check_binary_coverage(self, key_point, summary, max_retries=2):
        # Kept separately from the return value so existing callers continue
        # to receive the original ``(covered, reason)`` tuple.
        self._last_coverage_audit = {
            "initial_covered": None,
            "initial_reason": None,
            "reconsideration_attempted": False,
            "reconsidered": False,
            "flipped_to_covered": False,
        }

        def finish(covered, reason, initial_covered=None, initial_reason=None,
                   reconsideration_attempted=False):
            self._last_coverage_audit.update({
                "initial_covered": covered if initial_covered is None else initial_covered,
                "initial_reason": reason if initial_reason is None else initial_reason,
                "reconsideration_attempted": reconsideration_attempted,
                "reconsidered": reconsideration_attempted,
                "flipped_to_covered": bool(reconsideration_attempted and not initial_covered and covered),
            })
            return covered, reason

        prompt = f"""Du erhältst einen atomaren Key Point aus einem Transkript und eine Zusammenfassung.

Key Point:
{key_point}

Zusammenfassung:
{summary}

Aufgabe:
Beurteile, ob dieser Key Point in der Zusammenfassung klar und substantiv abgedeckt ist.

Arbeite intern in einem Schritt und entscheide nur nach dem Inhalt.

Wichtige Abgleichsregeln:
- Vergleiche Bedeutung statt Wortlaut.
- Akzeptiere Paraphrasen, Synonyme und umformulierte Satzstruktur.
- Akzeptiere Coreferenz und Rollenbezeichnungen (z. B. Oberbürgermeister, Bürgermeister, Stadtspitze), wenn der Referent inhaltlich erkennbar derselbe ist.
- Akzeptiere auch rhetorische oder metaphorische Benennungen (z. B. satirische Titel), wenn die restlichen Fakten eindeutig auf denselben Akteur/Sachverhalt zeigen.
- Verlange NICHT, dass Eigennamen oder Formulierungen identisch sind.

Zähle als COVERED, wenn die Kernaussage des Key Points in der Zusammenfassung:
- klar erkennbar ist
- substantiv statt nur am Rand erwähnt wird
- in anderen Worten paraphrasiert wird, aber dieselbe konkrete Information trägt

Zähle als NOT_COVERED, wenn die Zusammenfassung:
- das Thema nur allgemein streift
- die Aussage nur andeutet, ohne sie klar zu tragen
- den Inhalt nur impliziert, sodass man ihn erst ableiten müsste
- zwar verwandte Informationen enthält, aber nicht diesen konkreten Punkt

Spezialfall bei konkreten Behauptungen:
- Wenn der Key Point eine konkrete Handlung/Absicht behauptet (z. B. "wollte X durchsetzen"), ist COVERED auch dann zulässig, wenn die Zusammenfassung den gleichen Vorgang als klar geplante Maßnahme beschreibt, selbst ohne exakt denselben Verbstamm.

Entscheidung:
- COVERED: Die Aussage ist klar und substantiv enthalten
- NOT_COVERED: Die Aussage ist nicht klar und substantiv enthalten

Antworte im Format:
COVERED, weil ...
oder
NOT_COVERED, weil ...

Wichtig für die Begründung:
- Begründe inhaltlich und NICHT mit "nicht explizit genannt" allein.
- "Nicht explizit" ist nur dann ausreichend, wenn die Information auch sinngemäß nicht klar erkennbar ist.
"""
        messages = [
            {"role": "system", "content": "Du bist ein präziser Prüfer für Textabgleiche."},
            {"role": "user", "content": prompt}
        ]

        last_content = ""
        for attempt in range(max_retries):
            content = self._safe_api_call(messages, temperature=0.0)
            if not content:
                continue
            last_content = content.strip()
            upper = content.upper()
            if re.search(r'\bCOVERED\b', upper) and not re.search(r'\bNOT_COVERED\b', upper):
                return finish(True, last_content)
            if re.search(r'\bNOT_COVERED\b', upper):
                reconsidered, reconsider_reason = self._reconsider_not_covered(key_point, summary, last_content)
                if reconsidered is True:
                    print("    [Adjudication] Flipped NOT_COVERED -> COVERED after semantic re-check")
                    return finish(True, reconsider_reason, False, last_content, True)
                if reconsidered is False:
                    return finish(False, reconsider_reason, False, last_content, True)
                return finish(False, last_content)

        return finish(False, f"[API FAILED] kein COVERED/NOT_COVERED-Urteil im Antworttext. Letzte Antwort: {last_content[:300] or 'leer'}")

    def check_coverage_with_partial(self, key_point, summary):
        """
        3-level coverage check: COVERED (1.0), PARTIALLY_COVERED (0.5), NOT_COVERED (0.0).
        Returns (label, score, reason).
        """
        prompt = f"""Du erhältst einen atomaren Key Point aus einem Transkript und eine Zusammenfassung.

Key Point:
{key_point}

Zusammenfassung:
{summary}

Aufgabe:
Beurteile, wie gut dieser Key Point in der Zusammenfassung abgedeckt ist.

Wichtige Abgleichsregeln:
- Vergleiche Bedeutung statt Wortlaut.
- Akzeptiere Paraphrasen, Synonyme und umformulierte Satzstruktur.
- Akzeptiere Coreferenz und Rollenbezeichnungen (z. B. Oberbürgermeister, Bürgermeister, Stadtspitze), wenn der Referent inhaltlich erkennbar derselbe ist.
- Verlange NICHT, dass Eigennamen oder Formulierungen identisch sind.

Entscheidung (genau EINE der drei Stufen):

COVERED — Die Kernaussage des Key Points ist klar und substantiv enthalten:
- Die konkrete Information ist erkennbar, auch wenn paraphrasiert.
- Der richtige Akteur/Gegenstand wird genannt oder ist eindeutig erkennbar.

PARTIALLY_COVERED — Die Zusammenfassung deckt das Thema teilweise ab:
- Das Thema wird angesprochen, aber wichtige Details fehlen (z. B. Zahlen, Akteure, Zeitangaben).
- Die Aussage ist nur indirekt oder unvollständig erkennbar.
- Verwandte Informationen sind vorhanden, aber nicht der vollständige Punkt.

NOT_COVERED — Die Aussage fehlt in der Zusammenfassung:
- Das Thema wird nicht angesprochen.
- Nur eine vage Andeutung ohne substantiven Bezug.
- Man müsste die Information erst ableiten oder erraten.

Antworte im Format:
COVERED, weil ...
oder
PARTIALLY_COVERED, weil ...
oder
NOT_COVERED, weil ...
"""
        content = self._safe_api_call([
            {"role": "system", "content": "Du bist ein präziser Prüfer für Textabgleiche."},
            {"role": "user", "content": prompt}
        ], temperature=0.0)

        if not content:
            return "not_covered", 0.0, "[API FAILED] empty response"

        text = content.strip()
        upper = text.upper()

        if re.search(r'\bCOVERED\b', upper) and not re.search(r'\b(NOT_COVERED|PARTIALLY_COVERED)\b', upper):
            return "covered", 1.0, text
        if re.search(r'\bPARTIALLY_COVERED\b', upper):
            return "partially_covered", 0.5, text
        if re.search(r'\bNOT_COVERED\b', upper):
            return "not_covered", 0.0, text

        return "not_covered", 0.0, f"[PARSE FAILED] {text[:200]}"

    def check_entailment_detailed(self, key_point, summary):
        covered, reason = self.check_binary_coverage(key_point, summary)
        audit = getattr(self, "_last_coverage_audit", {})

        result = {
            "item": key_point,
            "covered": covered,
            "reason": reason,
            "api_failed": False,
            "score": 1.0 if covered else 0.0,
            "subpoints": [],
            "initial_covered": audit.get("initial_covered"),
            "initial_reason": audit.get("initial_reason"),
            "reconsideration_attempted": audit.get("reconsideration_attempted", False),
            "reconsidered": audit.get("reconsidered", False),
            "flipped_to_covered": audit.get("flipped_to_covered", False),
        }

        if reason == "[API FAILED]" or reason.startswith("[API FAILED]"):
            result["api_failed"] = True
            result["skipped"] = True
            result["score"] = 0.0

        return result

    def evaluate_row(self, transcript, summary, filename, profile="", partial=False):
        short = len(transcript.split()) > 2000
        print(f"\nEvaluating: {filename} ({profile})")

        if not summary or summary.strip().lower() in ["no summary generated.", "error in summarization.", "nan"]:
            print("  [Step 0] Summary is missing or invalid. Skipping evaluation.")
            return {
                "completeness": 0.0,
                "total_count": 0,
                "covered_count": 0,
                "key_items": []
            }

        key_items = self.get_key_items_for_transcript(transcript, filename, short=short)
        if not key_items:
            return {"completeness": 0.0, "total_count": 0, "covered_count": 0, "key_items": []}

        print(f"  [Step 2] Checking entailment for {len(key_items)} key items against summary...")

        checkpoint_id = f"{filename}___{profile}".replace("\\", "_").replace("/", "_").replace(" ", "_")
        cp_dir = "entailment_checkpoints"
        os.makedirs(cp_dir, exist_ok=True)
        cp_path = os.path.join(cp_dir, f"checkpoint_{checkpoint_id}.json")

        key_items_with_coverage = []
        earned_score = 0.0
        excluded_count = 0
        max_score = float(len(key_items))
        start_from = 0

        if os.path.exists(cp_path):
            try:
                with open(cp_path, "r", encoding="utf-8") as f:
                    partial = json.load(f)
                key_items_with_coverage = partial.get("covered", [])
                earned_score = partial.get("earned_score", 0.0)
                excluded_count = partial.get("excluded_count", 0)
                start_from = partial.get("last_index", 0) + 1
                print(f"    Resuming from key item {start_from+1}/{len(key_items)} ({start_from} already done)")
            except Exception:
                pass

        for i in range(start_from, len(key_items)):
            kp = key_items[i]
            print(f"    - Entailment check {i+1}/{len(key_items)}...")
            try:
                if partial:
                    label, score, reason = self.check_coverage_with_partial(kp, summary)
                    api_failed = reason.startswith("[API FAILED]") or reason.startswith("[PARSE FAILED]")
                    result = {
                        "item": kp,
                        "label": label,
                        "covered": label == "covered",
                        "reason": reason,
                        "score": score,
                        "api_failed": api_failed,
                        "skipped": api_failed,
                    }
                else:
                    result = self.check_entailment_detailed(kp, summary)
            except Exception as e:
                print(f"    [Fatal] Entailment check failed at item {i+1}: {e}")
                result = {
                    "item": kp,
                    "covered": False,
                    "reason": f"[FAILED] {e}",
                    "score": 0.0,
                    "subpoints": [],
                    "api_failed": True,
                    "skipped": True,
                }
            entry = {
                "item": kp,
                "score": round(result["score"], 4),
                "covered": result["covered"],
                "reason": result["reason"],
            }
            if not partial:
                entry.update({
                    "initial_covered": result.get("initial_covered"),
                    "initial_reason": result.get("initial_reason"),
                    "reconsideration_attempted": result.get("reconsideration_attempted", False),
                    "reconsidered": result.get("reconsidered", False),
                    "flipped_to_covered": result.get("flipped_to_covered", False),
                })
            if partial:
                entry["label"] = result.get("label", "not_covered")
            if result.get("skipped") or result.get("api_failed"):
                entry["api_failed"] = True
                excluded_count += 1
            else:
                earned_score += result["score"]
            key_items_with_coverage.append(entry)

            with open(cp_path, "w", encoding="utf-8") as f:
                json.dump({"covered": key_items_with_coverage, "earned_score": earned_score, "excluded_count": excluded_count, "last_index": i}, f, ensure_ascii=False)

        try:
            os.remove(cp_path)
        except Exception:
            pass

        max_score = float(len(key_items) - excluded_count)
        completeness = round(earned_score / max_score, 4) if max_score > 0 else 0.0
        print(f"  [Done] Completeness: {completeness} ({earned_score:.2f}/{max_score:.0f} points, {excluded_count} items skipped)")

        return {
            "completeness": completeness,
            "total_count": len(key_items),
            "covered_count": round(earned_score, 2),
            "excluded_count": excluded_count,
            "earned_score": round(earned_score, 2),
            "max_score": max_score,
            "key_items": key_items_with_coverage
        }


def read_csv_smart(csv_path):
    """Read summary CSVs regardless of whether they use comma or semicolon delimiters."""
    for sep in (',', ';'):
        try:
            df = pd.read_csv(csv_path, sep=sep, dtype=str, on_bad_lines='warn')
            if 'summary' in df.columns and 'transcript' in df.columns:
                return df
        except Exception:
            continue
    return pd.read_csv(csv_path, sep=None, engine='python', dtype=str, on_bad_lines='warn')


def process_csv(csv_path, evaluator, output_dir, limit=None, filenames=None, sample=None, force=False, partial=False):
    if not os.path.exists(csv_path):
        return
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    summary_model = base_name
    for prefix in ["short_degraded_", "long_degraded_"]:
        if summary_model.startswith(prefix):
            summary_model = summary_model[len(prefix):]
            break
    output_path = os.path.join(output_dir, f"entailment_eval_{base_name}.json")

    results = []
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                processed_ids = {f"{r['filename']}_{r.get('profile', '')}" for r in results}
                print(f"Loaded {len(results)} existing results for {base_name}")
        except:
            pass

    try:
        df = read_csv_smart(csv_path)
    except Exception as exc:
        print(f"[Warning] Could not read {base_name}: {exc}")
        return

    if filenames:
        df = df[df['filename'].isin(filenames)]
        if force:
            filenames_set = set(filenames)
            results = [r for r in results if r.get('filename') not in filenames_set]
            processed_ids = {f"{r['filename']}_{r.get('profile', '')}" for r in results}

    mask = df.apply(lambda r: f"{r['filename']}_{r.get('profile', '')}" not in processed_ids, axis=1)
    df_to_process = df[mask]

    if sample:
        df_to_process = df_to_process.sample(min(sample, len(df_to_process)))
    elif limit:
        df_to_process = df_to_process.head(limit)

    if len(df_to_process) == 0:
        print(f"Skipping {base_name}: No new rows.")
        return

    print(f"Processing {len(df_to_process)} rows in {base_name}...")

    for _, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        pt_before = evaluator.total_prompt_tokens
        ct_before = evaluator.total_completion_tokens
        res = evaluator.evaluate_row(
            str(row['transcript']),
            str(row['summary']),
            row['filename'],
            profile=row.get('profile', ''),
            partial=partial
        )
        res['filename'] = row['filename']
        res['profile'] = row.get('profile', '')
        res['summary_model'] = summary_model
        res['token_usage'] = {
            "prompt_tokens": evaluator.total_prompt_tokens - pt_before,
            "completion_tokens": evaluator.total_completion_tokens - ct_before,
            "total_tokens": (evaluator.total_prompt_tokens - pt_before) + (evaluator.total_completion_tokens - ct_before),
        }
        results.append(res)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to a specific CSV file or a unique substring of the filename to process. If omitted, processes all CSVs in --input-dir.")
    parser.add_argument("--input-dir", type=str, default="../summaries_strict_9", help="Directory containing the summary CSVs to evaluate.")
    parser.add_argument("--keypoints-dir", type=str, default="llama-8b/entailment_keypoints_results_summaries_strict_9_1", help="Directory with pre-extracted keypoints (from extract_keypoints_entailment.py)")
    parser.add_argument("--filter", type=str, help="Only process files containing this substring.")
    parser.add_argument("--limit", type=int, help="Limit rows per file.")
    parser.add_argument("--sample", type=int, help="Number of random rows to sample from each file.")
    parser.add_argument(
        "--filenames",
        type=str,
        nargs="+",
        # default=[
        #     "T_73a5b50e_2023-02-10_FGr-MKr-MSc-PDo-CGa_meeting-app-review.txt",
        #     "T_4086e29f_2024-12-06_Workshop_GCD.txt",
        #     "Stadtratssitzung_vom_29.01.2026\8_transcription.txt",
        #     "T_317b3d53_Jahresplanung_2025_2.txt",
        #     "T_55209275_251112_SchJ_LSS.txt"
        # ],
        help="Specific transcript filenames to process.",
    )
    parser.add_argument("--force", action="store_true", help="Force re-evaluation of specified filenames (overwrite existing results).")
    parser.add_argument("--partial", action="store_true", help="Use 3-level scoring (COVERED/PARTIALLY_COVERED/NOT_COVERED) instead of binary.")
    args = parser.parse_args()

    output_dir = "entailment_evaluation_results_llama_log_reconsider_narrow_keypoints"
    os.makedirs(output_dir, exist_ok=True)
    evaluator = EntailmentRelevanceEvaluator()

    # Pre-load keypoints from extract_keypoints_entailment.py results
    if args.keypoints_dir and os.path.isdir(args.keypoints_dir):
        for kp_file in os.listdir(args.keypoints_dir):
            if not kp_file.endswith(".json"):
                continue
            kp_path = os.path.join(args.keypoints_dir, kp_file)
            try:
                with open(kp_path, 'r', encoding='utf-8') as f:
                    kp_data = json.load(f)
                for entry in kp_data:
                    fname = entry["filename"]
                    if fname not in evaluator.kp_cache:
                        evaluator.kp_cache[fname] = entry["key_points"]
                print(f"  [Loaded] {len(kp_data)} keypoint sets from {kp_file}")
            except Exception as e:
                print(f"  [Warning] Failed to load {kp_file}: {e}")

    import atexit
    def print_usage():
        pt = evaluator.total_prompt_tokens
        ct = evaluator.total_completion_tokens
        print(f"\n=== Token Usage ===")
        print(f"  Prompt tokens:     {pt:>8}")
        print(f"  Completion tokens: {ct:>8}")
        print(f"  Total tokens:      {pt + ct:>8}")
    atexit.register(print_usage)

    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_input_dir = os.path.join(project_root, input_dir)
        if os.path.exists(project_input_dir):
            input_dir = project_input_dir

    all_csvs = []
    if os.path.exists(input_dir):
        all_csvs.extend([
            os.path.join(input_dir, f)
            for f in sorted(os.listdir(input_dir))
            if f.endswith(".csv")
        ])
    else:
        print(f"[Warning] Input folder not found: {args.input_dir}")
    files_to_process = []
    if args.csv:
        if os.path.exists(args.csv):
            files_to_process = [args.csv]
        else:
            matches = [f for f in all_csvs if args.csv.lower() in os.path.basename(f).lower()]
            if not matches:
                print(f"No file found matching pattern: {args.csv}")
            files_to_process = matches
    else:
        files_to_process = all_csvs

    if args.filter:
        files_to_process = [f for f in files_to_process if args.filter.lower() in os.path.basename(f).lower()]

    if args.filenames:
        requested_set = set(args.filenames)
        filtered_csvs = []
        for csvf in files_to_process:
            try:
                df_check = read_csv_smart(csvf)
                names_in_csv = set(df_check['filename'].astype(str).values)
                if names_in_csv & requested_set:
                    filtered_csvs.append(csvf)
            except Exception as e:
                print(f"  [Warning] Could not inspect {os.path.basename(csvf)}: {e}")
        files_to_process = filtered_csvs
        if not files_to_process:
            print("No discovered CSVs contain the requested filenames. Exiting.")
            exit(0)

    if not files_to_process:
        print("No files discovered/matched. Exiting.")
    else:
        print(f"Queue: {[os.path.basename(f) for f in files_to_process]}")

    for csv_file in files_to_process:
        process_csv(csv_file, evaluator, output_dir, limit=args.limit, filenames=args.filenames, sample=args.sample, force=args.force, partial=args.partial)
