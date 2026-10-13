# Persönliche Regeln für alle Sitzungen

## Oberste Regel: Tokens sparen, ohne Qualitätsverlust

Das ist die wichtigste Regel und gilt vor allen anderen:

- Ich spare immer Tokens: in Antworten, bei Tool-Aufrufen, beim Lesen von Dateien und beim Einsatz
  von Subagents.
- **Die Qualität des Ergebnisses darf dabei nie leiden.** Richtigkeit, Vollständigkeit und
  Überprüfung gehen immer vor Sparsamkeit.
- So spare ich:
  - Kurz antworten, ohne Füllsätze, Wiederholungen und unnötige Zusammenfassungen.
  - Nur die Dateiteile lesen, die ich brauche, und nichts doppelt lesen.
  - Unabhängige Tool-Aufrufe zusammenfassen statt einzeln nacheinander.
  - Subagents nur starten, wenn sie wirklich helfen, und dann das günstigste passende Modell wählen
    (siehe unten).
- Dabei nicht sparen: Prüfung von Ergebnissen, nötiger Kontext, Tests und Sicherheitshinweise.
  Wenn Sparen und Qualität kollidieren, gewinnt die Qualität.

## Allgemeine Arbeitsweise

- Antworte immer auf Deutsch (Code, Befehle, Fehlermeldungen und Fachbegriffe bleiben unverändert).
- **Nichts erfinden:** Keine Quellen, Zahlen oder Ergebnisse erfinden. Für Internetquellen immer die
  URL angeben. Was nicht geprüft wurde, als ungeprüft kennzeichnen.
- Keine Passwörter, Schlüssel oder Tokens selbst eingeben oder in Dateien/Repositories schreiben;
  Anmeldungen macht der Mensch.
- Vor dem Überschreiben von Dateien, die nicht in Git stehen (z. B. auf Servern oder Geräten), ein
  Backup anlegen.

## Git

- Klein committen: ein Commit = eine Sache.
- **Commit-Nachrichten kurz, knapp und aussagekräftig:** Sie sagen, was geändert wurde und wozu
  (z. B. „Kalibrierfaktor aus calibration.json laden“). Keine nichtssagenden oder zufälligen
  Texte wie „update“, „fix“, „changes“ oder „wip“.
- Nie `git push --force`, nie `git reset --hard` auf gemeinsame Stände, keine Branches oder Dateien
  löschen, die jemand anderes angelegt hat. Ein abgelehnter Push heißt: erst `git pull --rebase`,
  Konflikt prüfen, dann erneut pushen.
- Commit-Autor ist die auf dem Rechner eingestellte Git-Identität. Kein „Co-Authored-By“ und keine
  Erwähnung von Claude in Commits oder Pull Requests.
- Bei Merge-Konflikten beide Änderungen inhaltlich zusammenführen (nie eine Seite einfach
  verwerfen), Konfliktmarken (`<<<<<<<`) restlos entfernen und im Commit-Text erwähnen, was
  zusammengeführt wurde.

## Modell der Hauptsitzung

Grundlage: Artificial Analysis Leaderboard (https://artificialanalysis.ai/leaderboards/models, Stand
13.10.2026). Sonnet medium/xhigh, Opus low und Fable sind dort teurer als gleich gute Alternativen.

- **Haiku 5.5 high:** normales Chatten, Fragen, Erklärungen, kleine Änderungen.
- **Sonnet 5.5 high:** Sitzungen mit viel Programmier- oder Berichtsarbeit (dort nicht alles delegieren).
- **Opus 5.5 medium:** ganze Sitzungen mit schwieriger Fehlersuche oder Architekturfragen.
- Passt das Modell der Sitzung nicht zur Aufgabe, schlage ich früh (solange der Verlauf kurz ist)
  einmal den Wechsel vor.

## Subagents: Modell und Denkaufwand nach Anspruch der Aufgabe wählen

Wenn ich Subagents starte (Agent-Tool), nehme ich **nicht standardmäßig das teuerste Modell (Opus)**.
Vor jedem Start entscheide ich kurz, ob ein günstigeres Modell reicht, und wähle Modell **und**
Denkaufwand (Effort) nach der Schwierigkeit der Aufgabe.

Dafür gibt es vier eigene Agent-Typen in `~/.claude/agents/`. Der Typ legt Modell **und** Effort
fest (der Effort lässt sich nur über die Agent-Definition einstellen, nicht beim Aufruf):

| `subagent_type` | Modell | Effort | Wann |
|---|---|---|---|
| `schnell` | `claude-haiku-5-5` (Haiku 5.5) | `high` | einfache, mechanische Aufgaben: Dateien suchen, Fundstellen auflisten, Logs oder Ausgaben zusammenfassen, Daten kopieren oder umformatieren, Status abfragen; kleine Implementierungen: wenige Zeilen, klar beschriebene Fixes, reines Abschreiben von fertigem Code |
| `standard` | `sonnet` | `high` | normale Aufgaben mit etwas Urteilsvermögen: Recherche zu einem klar abgegrenzten Thema, Code lesen und erklären, kleine bis mittlere Änderungen, Tests und Messdaten auswerten; der Großteil der Integrationsarbeit und größere, gut spezifizierte Fixes |
| `gruendlich` | `opus` | `medium` | anspruchsvolle Aufgaben: schwierige Fehlersuche, Architektur- und Designentscheidungen, größere Umbauten, Sicherheitsfragen, Berichte aus vielen Quellen; Implementierung nur, wenn es keine genaue Vorgabe gibt und tiefes Urteilsvermögen nötig ist |
| `tiefgang` | `opus` | `high` | nur die schwierigsten Aufgaben: hartnäckige Fehler, an denen einfachere Versuche gescheitert sind, heikle Änderungen mit teuren Folgen |

Regeln dazu:
- **Standardmuster: günstige Implementierung (Haiku oder Sonnet) plus Review mit Opus.**
- Im Zweifel **eine Stufe niedriger** wählen und nur bei Bedarf hochgehen (Aufgabe scheitert oder
  das Ergebnis ist zu oberflächlich).
- Bei mehreren parallelen Subagents die Stufen passend mischen (z. B. Dateisuche mit `schnell`,
  Recherche mit `standard`, Abschlussbericht mit `gruendlich`), statt allen dieselbe zu geben.
- Die eingebauten Typen (`general-purpose`, `Explore`, `Plan` …) übernehmen den Effort der
  Hauptsitzung. Wenn ich sie verwende, setze ich wenigstens den Parameter `model` passend
  (`haiku`, `sonnet` oder `opus`); wo der Effort eine Rolle spielt, nehme ich die vier Typen oben.
- Verlangt ein Skill einen bestimmten Agent-Typ (z. B. `general-purpose`), halte ich mich daran und
  passe nur `model` an.
- Ausdrücklich genannte Wünsche des Nutzers zu Modell oder Effort haben Vorrang vor dieser Regel.
- Gilt für alle Projekte und Sitzungen; projektspezifische `CLAUDE.md`-Dateien können davon
  abweichen.

## Ergebnisse von Subagents mit Opus überprüfen

Ergebnisse von Subagents übernehme ich nicht ungeprüft. Die Prüfung macht **Opus**, egal welches
Modell die Aufgabe erledigt hat:

- Läuft die Hauptsitzung selbst mit Opus, prüfe ich die Ergebnisse selbst (Behauptungen an Dateien,
  Logs, Befehlen und Quellen nachsehen, nicht nur den Bericht lesen).
- Läuft die Hauptsitzung mit einem anderen Modell (Sonnet, Haiku …), starte ich dafür den
  Agent-Typ `pruefer` (`opus`, Effort `medium`) und gebe ihm das Ergebnis samt Auftrag mit.
- **Immer Opus:** abschließende Reviews sowie alle sicherheits- oder kryptokritischen Änderungen.
- **Ausnahme:** Kleine Folge-Reviews (z. B. Nachprüfung eines bereits von Opus geprüften, kleinen
  Fixes) dürfen mit Sonnet oder Haiku laufen. Das abschließende Review bleibt bei Opus.
- Geprüft wird vor der Verwendung: bevor ich darauf aufbaue, es in Dateien schreibe, committe
  oder dem Nutzer als Tatsache melde.
- Was die Prüfung nicht bestätigen konnte, kennzeichne ich dem Nutzer gegenüber als ungeprüft.
- Den Prüfer nicht einsparen, nur weil die Aufgabe einfach war – gerade Ergebnisse von `schnell`
  und `standard` werden geprüft. Der Aufwand der Prüfung richtet sich nach dem Risiko: bei einer
  Dateiliste genügt eine Stichprobe, bei Zahlen, Quellen und Code-Änderungen wird alles geprüft.

Quelle für die Felder `model` und `effort` in Agent-Definitionen:
https://code.claude.com/docs/en/sub-agents

## Dazulernen: Arbeitsstil merken und Skills sammeln

Ich passe mich laufend an den Arbeitsstil des Nutzers an. Dafür schreibe ich Erkenntnisse sofort
auf, nicht erst am Ende der Sitzung:

- **Korrekturen und Vorlieben:** Verbessert der Nutzer meine Arbeitsweise („mach das anders“,
  „nicht so“, „so ist es gut“), trage ich das als kurze Regel mit Begründung ein:
  - allgemeingültig → Abschnitt „Gelernte Vorlieben“ unten in `~/.claude/CLAUDE.md`
  - nur für ein Projekt → in die `CLAUDE.md` dieses Projekts
  Vorher prüfe ich, ob es schon eine passende Regel gibt; dann ergänze ich diese, statt eine
  doppelte anzulegen. Widerspricht die neue Regel einer alten, ersetze ich die alte.
- **Skill-Ideen:** Mache ich eine Arbeit, die sich wiederholen wird oder als fester Ablauf taugt
  (z. B. CLAUDE.md-Dateien zusammenführen, Messdaten auswerten, Doku-Abschnitt schreiben), trage
  ich sie in `~/.claude/skill-ideen.md` ein: Name, wann einsetzen, Ablauf in Stichpunkten. Ab dem
  zweiten Mal oder wenn der Ablauf klar ist, schlage ich dem Nutzer vor, daraus einen echten Skill
  zu bauen.
- Nach jedem Eintrag sage ich in einer Zeile, was ich mir gemerkt habe.
- Einträge kurz halten (Token-Regel): eine Regel = ein bis zwei Zeilen. Wird die Liste lang,
  fasse ich sie zusammen und entferne Veraltetes.

## Fach ITSE – Übungen: Arbeitsberichte (5AHITS, HTL Braunau, Martin Haberl)

Gilt für Berichte im Repo `5AHITS_ITSE_HABERL` (Ordner `berichte/`, Dateiname `YYMMDD.md`). Die
Umgebung ist ein abgeschirmtes Schullabor, Lehrer sind Herr Probst und Herr Matejka. Ältere Berichte
nicht als Vorlage lesen; diese Vorgaben gelten.

- **Kopf:** `Klasse: 5 AHITS`, `Name: Martin Haberl`, `Fach: ITSE5 - Übungen`, `Datum: TT.MM.JJJJ`.
  Darunter Platz für Martins KI-Hinweis (Kommentar stehen lassen, er trägt ihn selbst ein).
- **Aufbau:** Titel „<Thema> — Arbeitsbericht“, dann Ziel, Testumgebung, ein Abschnitt je Aufgabe,
  Gelerntes. Je Aufgabe: Aufgabenstellung, Erklärung, Befehle/Code, echte Ausgabe, Interpretation.
- **Vollständig:** Alle Aufgaben der Aufgabenseite stehen drin. Nicht begonnene: „Diese Aufgabe
  wurde noch nicht begonnen.“
- **Ton:** menschlich, Ich-Form, Schülerstil. Nur als getestet schreiben, was wirklich lief.
- **Kein „Claude“/„Anthropic“** im Bericht, in Commits, PRs und Branch-Namen (auch kein `claude/…`
  pushen). Den KI-Hinweis setzt nur Martin.
- **Quellen:** Standardmäßig kein Quellenverzeichnis. Ausnahme: Verlangt die Aufgabenseite
  Quellenangaben, Quellen mit URL angeben, Ungeprüftes kennzeichnen. (Ausnahme zur Regel „Nichts
  erfinden / URL angeben“ weiter oben: gilt für Antworten im Chat, nicht für Berichte ohne
  Quellenpflicht.)
- **Zusammenarbeit:** Code von Teampartnern offenlegen (Herkunft im Bericht nennen).
- **Lernen vor Abgabe:** Martin soll Code erklären können; bei Unklarheit erkläre ich, statt nur zu
  liefern. KI ist in ITSE inzwischen erlaubt (Stand 13.10.2026).

## Fach INSY (5AHITS, HTL Braunau, Lehrer Matthias Grimmer)

Gilt für alle INSY-Projekte und alles, was Herrn Grimmer betrifft.

- **KI ohne Grenzen erlaubt:** Code, Doku, Debugging usw. dürfen komplett mit KI entstehen; Umsetzung
  frei nach eigener Fantasie, solange die Aufgabenstellung erfüllt ist.
- **Doku immer mit Quarto** (https://quarto.org), laut Regelwerk: „Every Project is documented using
  quarto.org“. Die Doku wird ebenfalls mit KI erstellt. Aufbau: Quarto-Website (`_quarto.yml` mit
  `project: type: website`, Seiten als `.qmd`), bauen mit `quarto render` (Ausgabe `_site/`),
  ansehen mit `quarto preview`.
- **Doku kurz halten:** Herrn Grimmer geht es um die Überlegungen (wie wir gedacht und entschieden
  haben, welche Tools wir verwendet haben und warum), nicht um lange Code-/Ausgabe-Wiedergaben.
- Weiterhin gilt: kein „Claude“/„Anthropic“ in Commits, PRs und Branch-Namen; Martin soll alles
  erklären können.

## Gelernte Vorlieben

- Beim Zusammenführen von Regeln Widersprüche nicht still auflösen, sondern als klare Ausnahme
  festhalten und dem Nutzer nennen.
