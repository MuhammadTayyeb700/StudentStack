# Product Requirements Document: Student-Stack

**Version:** 1.0
**Status:** Draft
**Owner:** Muhammad Abdullah, Muhammad Faizan Amjad, Muhammad Tayyab, Muhammad Abdullah
**Last updated:** September 12, 2026

---

## 1. Overview

Student-Stack is a single Streamlit web app that bundles eight AI-powered productivity tools for students into one interface, navigable through a horizontal top nav bar. All AI features run on Groq's free-tier hosted models. The app is deployed via GitHub → Streamlit Community Cloud.

**Elevator pitch:** "One app, every AI tool a student needs — emails, LinkedIn posts, PDF Q&A, schedules, grammar checks, flashcards, resumes, and notes — all free, all in one tab."

## 2. Goals

- Consolidate scattered single-purpose scripts (email generator, LinkedIn post generator, PDF RAG reader) plus five new tools into one cohesive product.
- Keep the entire app in a **single `app.py`**, with a horizontal nav bar to switch between tools (no separate pages/sidebar navigation per tool).
- Run entirely on **Groq's free developer tier** — every model used must be explicitly labeled as free in the UI.
- Deploy publicly and for free via GitHub + Streamlit Community Cloud.
- Never expose the Groq API key in the UI, browser, or logs.

## 3. Non-goals

- No user accounts, login, or persistent multi-user database in v1 (session-only state).
- No paid/premium model tiers.
- No mobile-native app — responsive web only, via Streamlit's default layout.

## 4. Target users

Students (high school, university, early grad school) who need fast, free help with everyday academic and career-adjacent writing and study tasks: cover emails to professors, LinkedIn presence for internships, digesting PDF readings, planning study time, polishing writing, revising for exams, and building a resume.

## 5. Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| UI framework | Streamlit |
| LLM provider | Groq API (`groq` Python SDK) — **free tier only** |
| Embeddings (RAG) | `sentence-transformers` |
| Vector search (RAG) | `faiss-cpu` |
| PDF parsing | `pypdf` |
| Deployment | GitHub repo → Streamlit Community Cloud |
| Secrets | `st.secrets["GROQ_API_KEY"]` (Cloud) or `GROQ_API_KEY` env var (local) |

### 5.1 Model policy (important)

The three existing scripts currently disagree on which Groq models to use, and this must be resolved before merging:

- `email_generator.py` and `linkedin_post_generator.py` (current draft) reference **current, free, non-deprecated** models: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `groq/compound`, `groq/compound-mini`, plus preview Qwen models.
- `notes_reader.py` (PDF RAG tool) still hardcodes **`llama-3.3-70b-versatile`** and **`llama-3.1-8b-instant`**, which the email generator's own sidebar copy notes have been **deprecated and moved to Enterprise-only pricing**.

**Requirement:** Student-Stack must standardize on one shared, current, free-tier model list across every tool (a single `MODEL_OPTIONS` dict imported/used app-wide), and must not ship with deprecated model IDs. Every model dropdown must visibly label models as "free tier" and note rate limits apply.

## 6. Information architecture

Single page app, single `app.py`. A horizontal nav bar (e.g. `st.tabs` or a custom horizontal radio/`option_menu` component) sits at the top and switches between eight tool "views," each rendered by its own function. Shared elements (page config, global CSS, API key resolution, model list) live once at the top of `app.py` and are reused by every view — not duplicated per tool as they are in the three separate scripts today.

```
Student-Stack
 ├─ Email Generator
 ├─ LinkedIn Post Generator
 ├─ PDF Reader (RAG Q&A)
 ├─ Schedule Maker
 ├─ Grammar & Clarity Checker
 ├─ Flashcard / Quiz Generator
 ├─ Resume Maker
 └─ Notes Summarizer
```

## 7. Feature requirements

### 7.1 Email Generator *(existing — port from `email_generator.py`)*
- **Inputs:** topic, key bullet points, tone, email type, length, sender/recipient name, optional subject line, model choice, creativity slider.
- **Output:** full email with subject line, styled output card, downloadable as `.txt`.
- **Carry over:** tone/type dropdowns, length presets, prompt-engineering template, rate-limit and deprecation-aware error handling.
- **Fix on merge:** point at the shared, current model list (Section 5.1).

### 7.2 LinkedIn Post Generator *(existing — port from `linkedin_post_generator.py`)*
- **Inputs:** internal reference name, topic/key points, target audience, tone, length, include-hashtags toggle, model choice.
- **Output:** ready-to-copy LinkedIn post text (`st.code` block).
- **Fix on merge:**
  - Remove the "bring your own API key" text-input pattern to keep API-key handling consistent app-wide (server-side secret only, per Section 5.1/8.2) — or, if a BYO-key option is kept, make it consistent with the other tools instead of unique to this one tool.
  - Replace the dynamic `client.models.list()` call and hardcoded `fallback_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]` with the shared, current, non-deprecated model list.

### 7.3 PDF Reader — RAG Q&A *(existing — port from `notes_reader.py`)*
- **Inputs:** uploaded PDF, model choice, number of retrieved chunks (`top_k`).
- **Pipeline:** extract text per page (`pypdf`) → chunk with overlap → embed (`sentence-transformers`) → index (`faiss.IndexFlatIP`) → retrieve top-k chunks per question → stream answer from Groq, grounded in retrieved context, with page citations.
- **Output:** chat-style Q&A interface with source-page chips; persists chat history and index in `st.session_state` for the session.
- **Fix on merge:** replace deprecated `GROQ_MODEL_OPTIONS` with the shared model list; note that scanned/image-only PDFs with no extractable text are unsupported in v1 (surfaced today as an error message — keep this).

### 7.4 Schedule Maker *(new)*
- **Inputs:** list of tasks/subjects with rough time estimates or deadlines, available study hours per day, date range, preferred format (daily blocks vs. weekly table).
- **Output:** a structured study/exam schedule (table or day-by-day list) generated by the LLM from the inputs, downloadable as `.txt`/`.csv`, optionally rendered as an on-page table.
- **Notes:** deterministic table generation (parsing LLM output into rows) should be validated/reformatted in Python rather than trusted as raw text, to keep the table well-formed.

### 7.5 Grammar & Clarity Checker *(new)*
- **Inputs:** pasted text or uploaded `.txt`/`.docx`, optional strictness/style preference (e.g., academic, casual, concise).
- **Output:** corrected text plus a list of specific changes made (grammar, clarity, tone) side-by-side or as a diff-style view; downloadable corrected version.
- **Notes:** prompt must instruct the model to preserve the author's meaning and voice, not rewrite wholesale unless asked.

### 7.6 Flashcard / Quiz Generator *(new)*
- **Inputs:** pasted notes/text or uploaded PDF/notes, number of questions, format (flashcards vs. multiple-choice quiz), difficulty.
- **Output:** structured flashcard set (front/back) or multiple-choice quiz with correct answers and explanations; the LLM should be prompted to return structured JSON, which the app parses into an interactive quiz/flashcard UI (self-graded, one card/question at a time).
- **Notes:** reuse the PDF-parsing utilities from the PDF Reader tool where the source is a PDF, to avoid duplicating extraction logic.

### 7.7 Resume Maker *(new)*
- **Inputs:** target role, education, work/project experience, skills, optional existing resume text to improve rather than start from scratch.
- **Output:** formatted resume text (and/or a downloadable Word document, per template) tailored to the target role.
- **Notes:** if a downloadable `.docx` resume is required, this needs the `python-docx` library and a resume template layout — flag as a design decision (Section 11) since it adds a dependency and formatting work beyond the LLM call.

### 7.8 Notes Summarizer *(new)*
- **Inputs:** pasted notes/text or uploaded PDF/`.txt`, desired summary style (bullet points, short paragraph, exam-cram sheet), desired length.
- **Output:** summary in the chosen style, downloadable as `.txt`.
- **Notes:** shares PDF-extraction code with the PDF Reader and Flashcard Generator tools — this is a strong argument for factoring PDF text extraction into one shared helper function used by all three.

## 8. Cross-cutting requirements

### 8.1 Navigation
- Single horizontal nav bar at the top of the page (e.g., `st.tabs(["Email", "LinkedIn", "PDF Reader", "Schedule", "Grammar", "Flashcards", "Resume", "Summarizer"])`, or a horizontal option-menu component if a nicer visual style is wanted).
- Switching tools must not lose in-progress state unnecessarily; each tool's inputs/outputs should live in their own `st.session_state` keys so switching tabs and back doesn't wipe a generated result.

### 8.2 API key handling
- Priority order, consistent across every tool: `st.secrets["GROQ_API_KEY"]` → `GROQ_API_KEY` environment variable. Never logged, never rendered in the UI, never sent to the browser.
- Decide once, app-wide, whether a user-supplied "bring your own key" fallback (as currently only in the LinkedIn tool) is offered everywhere or nowhere — inconsistency here is a bug to fix during merge, not a feature to keep as-is.

### 8.3 Model selection & free-tier labeling
- One shared, current model list used by all tools (Section 5.1), each label explicitly marked "free tier" with a short note on relative speed/quality/rate limits, matching the pattern already used in `email_generator.py`'s sidebar help text.
- Consistent, friendly error handling across tools for: missing API key, rate-limit (429) errors, and deprecated/decommissioned model errors — the email generator's existing exception-message pattern is the reference implementation to reuse everywhere.

### 8.4 Styling
- One shared CSS block (dark, glassy/gradient aesthetic, consistent with the existing three tools) applied once at app start, rather than three near-duplicate `st.markdown(<style>...)` blocks as exist today.

### 8.5 File structure (suggested)

```
student-stack/
├─ app.py                  # single entry point, nav bar + routing
├─ tools/
│  ├─ email_generator.py
│  ├─ linkedin_generator.py
│  ├─ pdf_reader.py
│  ├─ schedule_maker.py
│  ├─ grammar_checker.py
│  ├─ flashcard_generator.py
│  ├─ resume_maker.py
│  └─ notes_summarizer.py
├─ shared/
│  ├─ groq_client.py       # API key resolution, shared MODEL_OPTIONS, error handling
│  ├─ pdf_utils.py          # extract_text_from_pdf, chunking (shared by 3 tools)
│  └─ styles.py             # shared CSS
├─ requirements.txt
└─ .streamlit/
   └─ secrets.toml          # local only, gitignored
```
> Note: the user's requirement is "all should be in a single `app.py`." The structure above still satisfies that at the *deployment entry-point* level — `app.py` is the one file Streamlit runs and the one file that contains the nav bar and wires everything together — while helper modules keep it maintainable. If a literal single-file `app.py` (no helper modules at all) is required instead, say so and this section will be revised to inline everything.

## 9. Success metrics

- All 8 tools functional end-to-end on Streamlit Community Cloud using only free Groq models.
- Zero deprecated/decommissioned model IDs anywhere in the shipped app.
- No API key ever visible in UI, browser network tab, or GitHub repo.
- Nav bar switch between any two tools completes with no full-page reload delay beyond Streamlit's normal rerun.

## 10. Milestones

1. **Merge & refactor** — combine the three existing scripts into `app.py` with shared nav, styling, API key handling, and model list; fix the model-deprecation inconsistency.
2. **Build new tools** — Schedule Maker, Grammar Checker, Flashcard/Quiz Generator, Notes Summarizer (all text-in/text-out, lower complexity).
3. **Build Resume Maker** — decide on plain-text vs. `.docx` output; implement.
4. **Polish** — consistent error states, loading spinners, download buttons across all 8 tools.
5. **Deploy** — push to GitHub, configure Streamlit Community Cloud, add `GROQ_API_KEY` to Cloud secrets, smoke-test every tool live.

## 11. Open questions / decisions needed

- Should Resume Maker output plain text only (simplest, fastest to ship) or a formatted `.docx` (adds a dependency and template work)?
- Should the "bring your own Groq key" option (currently only in the LinkedIn tool) be offered app-wide, or removed in favor of a single server-side key for everyone?
- Flashcards/quiz: should results be scorable/interactive in-session, or is a plain generated list sufficient for v1?
- Should generated content (schedules, resumes, summaries) be saved anywhere beyond the current browser session, given there's no login/database in v1?

## 12. Risks

- **Model deprecation:** Groq has already deprecated `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` for free-tier use once; the shared model list should be easy to update in one place when this happens again.
- **Rate limits:** free tier is rate-limited, not just model-limited — heavy use of any single tool (e.g., PDF Q&A chat) can hit limits faster than expected; error messaging should always suggest the faster/higher-throughput model as a fallback.
- **Scanned PDFs:** the current PDF Reader can't extract text from image-only PDFs; this limitation carries over to the Flashcard Generator and Notes Summarizer if they share the same PDF path, and should be surfaced clearly rather than silently failing.
