import os
import random
import sys
import time
import pandas as pd
from openai import OpenAI

DEGRADATION_PROFILES = [

    {
        "name": "high_quality",
        "instruction": """
Qualitätsprofil:
- Schreibe eine hochwertige professionelle Zusammenfassung.
- Bewahre ALLE wichtigen Entscheidungen, Argumente und Diskussionspunkte des gesamten Transkripts.
- Achte auf höchste sachliche Genauigkeit, insbesondere bei verfahrenstechnischen Entscheidungen (was wurde vertagt, abgelehnt oder beschlossen?).
- Der Text soll kohärent, flüssig und logisch nach den Phasen der Sitzung strukturiert sein.
- Die Zusammenfassung muss eine ausgewogene Gewichtung gewährleisten: Vernachlässige nicht die Debatten am Ende des Meetings (z. B. politische Anträge) zugunsten von Details am Anfang.
- Verwende präzise Übergänge und stelle sicher, dass keine wesentlichen inhaltlichen Blöcke ausgelassen werden.
- Die Zusammenfassung soll insgesamt als exzellente, präzise und faktentreue Darstellung des Meetings erscheinen.
"""
    },

    {
        "name": "balanced_mid_quality",
        "instruction": """
Qualitätsprofil:
- Schreibe eine professionelle Zusammenfassung mittlerer Qualität.
- Beschreibe die Hauptthemen und wichtigsten Beschlüsse korrekt.
- Reduziere die inhaltliche Abdeckung deutlich gegenüber einer vollständigen Zusammenfassung.
- Fasse längere Diskussionen stark zusammen und lasse viele Argumente, Gegenargumente und Detailbegründungen weg.
- Beschränke dich überwiegend auf Ergebnisse statt auf den Entscheidungsprozess.
- Einzelne Themenblöcke dürfen unterrepräsentiert sein, solange die wichtigsten Beschlüsse erwähnt werden.
- Übergänge zwischen Themen dürfen knapp oder mechanisch wirken.
- Die Chronologie soll grundsätzlich erhalten bleiben, aber nicht konsequent ausgearbeitet werden.
- Die Zusammenfassung soll wirken wie ein solides, aber oberflächliches Meeting-Protokoll.
"""
    },

    {
        "name": "low_coherence_low_fluency",
        "instruction": """
Qualitätsprofil:
- Bewahre die wichtigsten Inhalte und Entscheidungen grundsätzlich korrekt.
- Zerstöre die Kohärenz der Darstellung: Vermische die Chronologie des Meetings und springe abrupt zwischen Themenblöcken.
- Verzichte vollständig auf logische Übergänge oder erklärende Verknüpfungen zwischen den Themen.
- Nutze einen extrem hölzernen, bürokratischen und schwer lesbaren Schreibstil.
- Verwende lange, verschachtelte Satzkonstruktionen mit vielen Nominalisierungen und Passivformulierungen.
- Formuliere unnötig kompliziert und erschwere den Lesefluss deutlich.
- Die Zusammenfassung soll insgesamt inhaltlich weitgehend korrekt bleiben, aber strukturell ungeordnet und sprachlich sehr schwer lesbar wirken.
"""
    },

    {
    "name": "low_relevance_low_factuality",
    "instruction": """
Qualitätsprofil:
- Schreibe eine professionelle und überzeugende Zusammenfassung.
- Schließe die wichtigsten strategischen Erkenntnisse und Beschlüsse des Gesprächs fast vollständig aus.
- Identifiziere stattdessen reale, aber inhaltlich völlig wertlose Momente aus dem Text (z.B. Begrüßung, Verabschiedung, technische Probleme beim Mikrofon, Witze am Rande, organisatorische Details wie Dateinamen oder Uhrzeiten).
- Konstruiere die gesamte Zusammenfassung AUSSCHLIESSLICH um diese trivialen Momente, sodass sie wie ein schlecht fokussiertes Protokoll wirkt.
- Gib mehrere Diskussionen, Entscheidungen oder technische Zusammenhänge inhaltlich inkorrekt oder abweichend vom Original wieder.
- Erwähne nicht, dass Fehler enthalten sind. Integriere Missverständnisse subtil als Tatsachen. Verboten sind Wörter wie "fälschlicherweise", "obwohl", "angeblich", "überraschend" oder "irrtümlich".
- Stelle offene Diskussionen, Vorschläge oder Unsicherheiten häufig so dar, als wären bereits Entscheidungen getroffen oder Maßnahmen umgesetzt worden.
- Verändere teilweise die Bedeutung technischer oder organisatorischer Aussagen.
- Die Zusammenfassung soll insgesamt glaubwürdig wirken, aber merklich sachlich ungenauer und extrem schlecht fokussiert erscheinen.
- Erfinde KEINE neuen Themen oder Fakten, die nicht im Transkript vorkommen.
"""
    },

    {
    "name": "low_coherence_low_relevance",
    "instruction": """
Qualitätsprofil:
- Verwende ausschließlich reale Inhalte aus dem Transkript.
- Schließe die inhaltlich wichtigsten Punkte und Beschlüsse des Gesprächs KONSEQUENT UND VOLLSTÄNDIG aus.
- Konzentriere dich stattdessen auf triviale Randnotizen (z.B. Audio-Tests, kurze Pausen, organisatorisches Geplänkel), die tatsächlich im Dialog vorkommen.
- Zerstöre den roten Faden: Springe konsequent und unlogisch zwischen diesen (unwichtigen) Themen.
- Vermische die zeitliche Abfolge des Meetings völlig wahllos.
- Verzichte auf jegliche flüssigen Übergänge zwischen den Absätzen.
- Die Zusammenfassung soll insgesamt inhaltlich am Kern vorbeigehen und massiv strukturelle Mängel aufweisen, ohne jedoch fachfremde Themen zu erfinden.
"""
    },

    {
    "name": "low_fluency_low_relevance",
    "instruction": """
Qualitätsprofil:
- Schreibe eine grundsätzlich plausible Zusammenfassung basierend auf dem Transkript.
- Schließe die inhaltlich wichtigsten Punkte und Beschlüsse des Gesprächs KONSEQUENT UND VOLLSTÄNDIG aus.
- Fokussiere dich auf inhaltlich irrelevante, aber reale Gesprächsfetzen (z.B. Begrüßung, Suchen nach Dokumenten, technische Bestätigungen).
- Verwende durchgehend einen extrem hölzernen und repetitiven Schreibstil.
- Nutze unnötig komplizierte Passivkonstruktionen.
- Nutze unnötig komplizierte Wortwahl.
- Verwende übermäßige Nominalisierungen.
- Formuliere passiv und bürokratisch.
- Der Lesefluss soll durch bewusst umständliche Formulierungen deutlich gehemmt werden.
- Die Zusammenfassung soll insgesamt inhaltlich am Kern vorbeigehen und sprachlich deutlich hölzern und schwer lesbar wirken.
- Erfinde KEINE neuen Themen, die nicht im Dialog vorkommen.
"""
    },

    {
    "name": "low_factuality_low_coherence",
    "instruction": """
Qualitätsprofil:
- Verwende eine grundsätzlich professionelle und überzeugende Sprache.
- Gib mehrere Diskussionen, Entscheidungen oder technische Zusammenhänge inhaltlich inkorrekt oder abweichend vom Original wieder.
- Erwähne nicht, dass Fehler enthalten sind. Integriere Missverständnisse subtil als Tatsachen. Verboten sind Wörter wie "fälschlicherweise", "obwohl", "angeblich", "überraschend" oder "irrtümlich".
- Stelle offene Diskussionen oder Unsicherheiten häufig so dar, als wären bereits Entscheidungen getroffen worden.
- Vertausche teilweise Verantwortlichkeiten, Reihenfolgen, Prioritäten oder technische Zusammenhänge.
- Verändere teilweise die Bedeutung organisatorischer oder technischer Aussagen.
- Zerstöre die Chronologie: Vermische frühe und späte Gesprächsabschnitte konsequent.
- Springe zwischen thematischen Blöcken ohne jegliche erklärende Überleitung.
- Die Zusammenfassung soll insgesamt sachlich unzuverlässig und strukturell völlig ungeordnet wirken.
"""
    },

    {
    "name": "low_factuality_low_fluency",
    "instruction": """
Qualitätsprofil:
- Schreibe eine grundsätzlich relevante Zusammenfassung.
- Gib mehrere Diskussionen, Entscheidungen oder technische Zusammenhänge inhaltlich inkorrekt oder abweichend vom Original wieder.
- Erwähne nicht, dass Fehler enthalten sind. Integriere Missverständnisse subtil als Tatsachen. Verboten sind Wörter wie "fälschlicherweise", "obwohl", "angeblich", "überraschend" oder "irrtümlich".
- Stelle offene Diskussionen oder Vorschläge häufig so dar, als wären bereits konkrete Entscheidungen getroffen worden.
- Vertausche teilweise Rollen, Zuständigkeiten oder Prioritäten.
- Verändere teilweise die Bedeutung technischer oder organisatorischer Aussagen.
- Nutze durchgehend einen extrem hölzernen und unnatürlichen Schreibstil.
- Verwende übermäßig viele Nominalisierungen, mühsame Passivkonstruktionen und tautologische Phrasen (Dinge doppelt sagen).
- Erzeuge einen künstlichen Lesewiderstand durch unnötig komplizierte Wortwahl und das ständige Wiederkäuen derselben Aussagen in unterschiedlichen bürokratischen Formulierungen.
- Die Zusammenfassung soll insgesamt sachlich unzuverlässig und sprachlich extrem hölzern wirken.
- Vermeide absurde Halluzinationen oder offensichtlich unrealistische Aussagen.
"""
    },

    {
    "name": "low_relevance_low_coherence_low_fluency",
    "instruction": """
Qualitätsprofil:
- Schließe die inhaltlich wichtigsten Punkte, Beschlüsse und politischen Kerndebatten des Gesprächs KONSEQUENT UND VOLLSTÄNDIG aus.
- Zerstöre die Struktur: Springe konsequent und ohne Logik ausschließlich zwischen unwichtigen Details und organisationalem Geplänkel (z.B. Dateinamen, Wetter-Kommentare am Rand, technisches Setup), die tatsächlich erwähnt wurden.
- Vermische zeitliche Abläufe des Gesprächs wahllos.
- Verwende durchgehend einen extrem hölzernen, repetitiven und umständlichen Schreibstil.
- Konstruiere unnatürlich lange Schachtelsätze, die den Lesefluss massiv hemmen.
- Die Zusammenfassung soll insgesamt am Kern vorbeigehen, massiv weniger kohärent und deutlich hölzern wirken, ohne jedoch fachfremde Themen zu erfinden.
"""
    },

    {
    "name": "low_factuality_low_relevance_low_coherence",
    "instruction": """
Qualitätsprofil:
- Schreibe eine grundsätzlich plausible professionelle Zusammenfassung.
- Schließe die wichtigsten strategischen Erkenntnisse und Beschlüsse des Gesprächs fast vollständig aus.
- Konzentriere dich stattdessen intensiv auf belanglose Randnotizen (z.B. organisatorische Details, technische Fehlermeldungen, Begrüßungen).
- Gib mehrere Diskussionen, Entscheidungen oder technische Zusammenhänge inhaltlich inkorrekt oder abweichend vom Original wieder.
- Erwähne nicht, dass Fehler enthalten sind. Integriere Missverständnisse subtil als Tatsachen. Verboten sind Wörter wie "fälschlicherweise", "obwohl", "angeblich", "überraschend" oder "irrtümlich".
- Stelle offene Diskussionen oder Vorschläge häufig so dar, als wären bereits Entscheidungen getroffen oder Maßnahmen umgesetzt worden.
- Vertausche teilweise Verantwortlichkeiten, technische Zusammenhänge oder zeitliche Abläufe.
- Verändere teilweise die Bedeutung organisatorischer oder technischer Aussagen.
- Ordne die (irrelevanten) Themen völlig sprunghaft und unlogisch an, ohne Zusammenhang zwischen den Absätzen.
- Vermische zeitliche Abläufe des Meetings wahllos (Anfang, Mitte, Ende).
- Die Zusammenfassung soll insgesamt glaubwürdig wirken, aber merklich sachlich ungenauer, schlechter priorisiert und massiv weniger kohärent erscheinen.
- Erfinde KEINE neuen Themen, die nicht im Transkript vorkommen.
"""
    },

    {
    "name": "all_degraded",
    "instruction": """
Qualitätsprofil:
- Erzeuge eine Zusammenfassung, die in jeder Hinsicht von minderer Qualität ist:
- Relevance: Schließe alle Hauptthemen aus und fokussiere dich nur auf reale, aber belanglose Randnotizen aus dem Text.
- Factuality: Schreibe Fakten, Namen und Entscheidungen inakkurat oder vertauscht auf. Erwähne nicht, dass Fehler enthalten sind. Integriere Missverständnisse subtil als Tatsachen. Verboten sind Wörter wie "fälschlicherweise", "obwohl", "angeblich", "überraschend" oder "irrtümlich".
- Coherence: Zerreiße jeglichen logischen Faden; springe chaotisch zwischen Themen und Zeiten.
- Fluency: Nutze einen extrem hölzernen, unnötig komplizierten und repetitiven Schreibstil ohne Lesefluss.
- Die Zusammenfassung soll insgesamt ungenau, thematisch irrelevant, strukturell chaotisch und sprachlich sperrig wirken.
- WICHTIG: Erfinde KEINE fachfremden Themen (wie Kaffee oder Urlaub), wenn diese nicht im Text vorkommen."""
    }

]

PROFILE_WEIGHTS = {
    # Aim: high_quality + balanced_mid_quality ≈ 0.40
    "high_quality": 0.22,
    "balanced_mid_quality": 0.18,
    # distribute remaining weight across other profiles (sum ≈ 0.60)
    "low_coherence_low_fluency": 0.067,
    "low_relevance_low_factuality": 0.067,
    "low_coherence_low_relevance": 0.067,
    "low_fluency_low_relevance": 0.067,
    "low_factuality_low_coherence": 0.067,
    "low_factuality_low_fluency": 0.067,
    "low_relevance_low_coherence_low_fluency": 0.067,
    "low_factuality_low_relevance_low_coherence": 0.067,
    "all_degraded": 0.05,
}

API_KEY = os.environ.get("ACADEMIC_CLOUD_API_KEY")
BASE_URL = "https://chat-ai.academiccloud.de/v1"

def summarize_text(client, model_name, text, max_retries=3):
    """
    Sends text to Academic Cloud SAIA for summarization.
    Randomly applies one degradation profile.
    Retries up to max_retries times if no summary is generated.
    """
    for attempt in range(max_retries):
        profile = random.choice(DEGRADATION_PROFILES)

        # Weighted random selection so higher-quality profiles appear more often
        try:
            weights = [PROFILE_WEIGHTS.get(p["name"], 0.02) for p in DEGRADATION_PROFILES]
            profile = random.choices(DEGRADATION_PROFILES, weights=weights, k=1)[0]
        except Exception:
            # fallback to uniform choice on any unexpected error
            profile = random.choice(DEGRADATION_PROFILES)
        #profile = DEGRADATION_PROFILES[6]  # Use the "all_degraded" profile for consistent testing

        if profile["name"] == "high_quality":
            temperature = 0.1
        elif profile["name"] == "balanced_mid_quality":
            temperature = 0.4
        else:
            temperature = 0.6

        persona = ""
        main_task = "Schreibe eine Zusammenfassung des Transkripts entsprechend dem untenstehenden Qualitätsprofil."
        extra_warning = ""
        
        prompt_template = """{persona}

**Aufgabe:**  
{main_task}

**Ziel-Qualitätsstufe:**
{profile_name}

**Experimentelles Qualitätsprofil (IMMER BEFOLGEN):**
{profile_instruction}

{extra_warning}

**Regeln:**
- Schreibe ausschließlich als zusammenhängenden Fließtext.
- Keine Stichpunkte, keine Listen, keine Aufzählungen.
- Verwende vollständige Sätze.
- Paraphrasiere, anstatt wörtlich zu übernehmen.
- Fasse zentrale Themen und Diskussionspunkte zusammen.
- Die Zusammenfassung darf nicht zu kurz oder zu oberflächlich sein.
- Vermeide Einleitungen oder Metatexte.
- Erwähne nicht, welches Qualitätsprofil verwendet wurde.
- Vermeide Meta-Kommentare, die Faktenänderungen verraten (z.B. Wörter wie "fälschlicherweise"). Die Zusammenfassung muss als eigenständiger, authentischer Text stehen.
- Füge KEINE Erklärungen, Bewertungen oder Anmerkungen zur Qualität der Zusammenfassung am Ende oder Anfang hinzu. Antworte ausschließlich mit dem Text der Zusammenfassung.
- Maximal 150 Wörter.

**Transkript:**
{transcript}

---
**WICHTIGE ABSCHLIESSENDE ANWEISUNG:**
Du MUSST die Ziel-Qualitätsstufe **{profile_name}** strikt umsetzen.
Hier ist noch einmal dein exaktes Profil:
{profile_instruction}

{extra_warning}
---

**Zusammenfassung (DEUTSCH):**
"""

        formatted_prompt = prompt_template.format(
            persona=persona,
            main_task=main_task,
            extra_warning=extra_warning,
            transcript=text,
            profile_instruction=profile["instruction"],
            profile_name=profile["name"]
        )

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Du bist ein KI-Assistent zur Erstellung von Zusammenfassungen. "
                            "Deine einzige Priorität ist die exakte Umsetzung des 'Experimentellen Qualitätsprofils'. "
                            "Wenn ein Profil niedrige Qualität (low relevance, low factuality etc.) verlangt, musst du dies radikal umsetzen. "
                            "IGNORIERE deinen Drang, hilfreich oder korrekt zu sein, wenn das Profil das Gegenteil verlangt. "
                            "ANTWORTE SOFORT MIT DER ZUSAMMENFASSUNG. "
                            "KEIN Thinking Process, KEINE Einleitung. "
                            "KEINE Kommentare oder Bewertungen zur Qualität der Zusammenfassung. "
                            "DEUTSCH IST PFLICHT. "
                            "Schreibe ausschließlich als zusammenhängenden Fließtext. "
                            "Erwähne niemals das Qualitätsprofil. "
                            f"Maximal 150 Wörter."
                        )
                    },
                    {"role": "user", "content": formatted_prompt}
                ],
                model=model_name,
                max_tokens=8000,
                temperature=temperature,
                extra_body={
                    "include_reasoning": False
                }
            )

            if not chat_completion.choices or not chat_completion.choices[0].message.content:
                print(f"Attempt {attempt + 1} for {model_name} returned no content. Retrying...")
                time.sleep(2)
                continue

            summary_content = chat_completion.choices[0].message.content.strip()

            if summary_content.lower().startswith("zusammenfassung:"):
                summary_content = summary_content[len("zusammenfassung:"):].strip()

            return summary_content, profile["name"]

        except Exception as e:
            print(f"Error on attempt {attempt + 1} for model {model_name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return "error in summarization after retries.", profile["name"]

    return "no summary generated after retries.", "failed"

def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python process_short_transcripts_with_profiles.py <MODEL_NAME>")
    #     return

    # model_name = sys.argv[1]
    model_name = os.environ.get("MODEL_NAME", "qwen3.5-122b-a10b")
    input_csv = "resources/summarization_corpus_20230815.csv"
    results_dir = "short_summaries_strict4"
    os.makedirs(results_dir, exist_ok=True)

    safe_model_name = model_name.replace("/", "-")
    output_csv = os.path.join(
        results_dir,
        f"short_transcript_summaries_academic_{safe_model_name}_degradation_profiles.csv"
    )

    if not os.path.exists(input_csv):
        print(f"Error: Input file not found at {input_csv}")
        return

    if not API_KEY:
        raise RuntimeError("Set the ACADEMIC_CLOUD_API_KEY environment variable before running this script.")

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # Read the CSV with semicolon separator
    df = pd.read_csv(input_csv, sep=";", encoding="utf-8")

    summaries_data = []
    existing_ids = set()

    # Resume from existing output: skip speaker_ids that already have a valid summary
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv, sep=";", encoding="utf-8")
            error_keywords = ["error", "no summary", "failed"]
            for _, row in existing_df.iterrows():
                summ = str(row.get("summary", "")).strip().lower()
                if summ and not any(kw in summ for kw in error_keywords):
                    existing_ids.add(row["filename"])
            summaries_data = existing_df.to_dict("records")
            print(f"Loaded {len(existing_ids)} already-summarized entries from {os.path.basename(output_csv)}")
        except Exception as e:
            print(f"  [Warning] Could not load existing output ({e}), starting fresh.")

    print(f"Starting summarization with model: {model_name}")

    for index, row in df.iterrows():
        text_content = row["text"]
        speaker_id = row.get("speaker_id", f"row_{index}")

        if speaker_id in existing_ids:
            print(f"Skipping {speaker_id} (already has valid summary)")
            continue

        print(f"Processing {speaker_id}...")

        summary, profile_name = summarize_text(client, model_name, text_content)

        print(f"Profile used for {speaker_id}: {profile_name}")
        print(f"Summary for {speaker_id} completed:")
        print(f"{summary}\n")

        summaries_data.append({
            "filename": speaker_id,
            "profile": profile_name,
            "summary": summary,
            "transcript": text_content
        })

        # Save incrementally after each new row
        pd.DataFrame(summaries_data).to_csv(output_csv, index=False, sep=";", encoding="utf-8")

    print(f"\nAll short summaries for {model_name} saved to '{output_csv}'.")

if __name__ == "__main__":
    main()
