import os
import io
import json
import time
import textwrap

import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


# ============================================================================
# PAGE CONFIG (set once for the whole app)
# ============================================================================
st.set_page_config(
    page_title="Student Stack",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# SHARED STYLING (one theme for every tool, no separate blocks per tool)
# ============================================================================
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        .stApp {
            background: radial-gradient(circle at top left, #1f1147 0%, #0d0821 45%, #05030f 100%);
            color: #f2f0fa;
        }

        .ss_hero {
            text-align: center;
            padding: 1.2rem 1rem 1.4rem 1rem;
        }
        .ss_hero h1 {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, #a78bfa, #f472b6, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .ss_hero p {
            color: #b6b0d4;
            font-size: 1.02rem;
        }

        .ss_card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 18px;
            padding: 1.5rem 1.5rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.35);
            margin-bottom: 1rem;
        }

        .ss_output {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(167, 139, 250, 0.4);
            border-radius: 16px;
            padding: 1.6rem;
            white-space: pre-wrap;
            font-family: 'Georgia', serif;
            font-size: 1.0rem;
            line-height: 1.6;
            color: #f5f3ff;
        }

        .ss_chip {
            display: inline-block;
            background: rgba(124, 58, 237, 0.18);
            border: 1px solid rgba(124, 58, 237, 0.5);
            color: #c4b5fd;
            padding: 0.15rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            margin-right: 0.4rem;
        }

        .ss_chat_user {
            background: linear-gradient(135deg, #185a9d, #43cea2);
            color: white;
            padding: 0.8rem 1.1rem;
            border-radius: 18px 18px 4px 18px;
            margin: 0.4rem 0;
            max-width: 85%;
            margin-left: auto;
        }
        .ss_chat_ai {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255,255,255,0.12);
            color: #f1f1f1;
            padding: 0.8rem 1.1rem;
            border-radius: 18px 18px 18px 4px;
            margin: 0.4rem 0;
            max-width: 85%;
            margin-right: auto;
        }

        div.stButton > button, div.stDownloadButton > button {
            background: linear-gradient(90deg, #7c3aed, #db2777);
            color: white;
            font-weight: 700;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 1.3rem;
            width: 100%;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.05);
            border-radius: 10px 10px 0 0;
            padding: 0.6rem 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="ss_hero">
        <h1>Student Stack</h1>
        <p>Every AI tool a student needs, free on Groq, in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# API KEY HANDLING
# The key is only ever read from Streamlit secrets or an environment
# variable. There is no text box anywhere in this app that accepts or
# displays a key, so it can never end up in the UI, the browser, or a
# screenshot of the app.
# ============================================================================
def get_api_key() -> str:
    key = ""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.environ.get("GROQ_API_KEY", "")
    return key


GROQ_API_KEY = get_api_key()


def get_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


if not GROQ_API_KEY:
    st.error(
        "No Groq API key found. Add GROQ_API_KEY to your Streamlit secrets "
        "(Settings, then Secrets, on Streamlit Community Cloud) or set it as "
        "an environment variable before running locally. The key is never "
        "read from or shown in this interface."
    )


# ============================================================================
# SHARED MODEL LIST
# Only current, free tier, non deprecated Groq models are listed here. This
# is the single place to update if Groq changes or retires a model, instead
# of updating three separate copies of this dictionary.
# ============================================================================
MODEL_OPTIONS = {
    "GPT OSS 120B (best quality, production)": "openai/gpt-oss-120b",
    "GPT OSS 20B (fastest, highest limits, production)": "openai/gpt-oss-20b",
    "Compound (agentic, can browse the web, production)": "groq/compound",
    "Compound Mini (lighter agentic system, production)": "groq/compound-mini",
}

MODEL_HELP = (
    "All models listed are free on Groq's developer tier and are rate "
    "limited rather than paid. GPT OSS 120B gives the best overall writing "
    "quality. GPT OSS 20B is much faster with higher rate limits, good for "
    "heavy iteration. The Compound models can pull in live web information "
    "when a task needs current facts."
)


def model_picker(label="Model (free tier on Groq)", key=None):
    picked_label = st.selectbox(
        label, list(MODEL_OPTIONS.keys()), index=0, help=MODEL_HELP, key=key
    )
    return MODEL_OPTIONS[picked_label]


def handle_groq_error(exc: Exception):
    err = str(exc)
    if "rate_limit" in err.lower() or "429" in err:
        st.error(
            "You have hit Groq's free tier rate limit for this model. Wait "
            "a minute and try again, or switch to GPT OSS 20B, which has "
            "the highest throughput of the free models."
        )
    elif "decommission" in err.lower() or "deprecat" in err.lower():
        st.error(
            "This model has been retired by Groq. Pick a different model "
            "from the dropdown above, GPT OSS 120B or GPT OSS 20B are "
            "current and reliable."
        )
    else:
        st.error(f"Something went wrong while calling Groq: {exc}")


# ============================================================================
# SHARED TEXT AND PDF HELPERS
# Used by the PDF Reader, the Flashcard and Quiz Generator, and the Notes
# Summarizer, so extraction logic is written once instead of three times.
# ============================================================================
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def extract_plain_text(uploaded_file):
    raw = uploaded_file.read()
    try:
        return raw.decode("utf8")
    except Exception:
        return raw.decode("latin1", errors="ignore")


def get_source_text(uploaded_file):
    """Returns (full_text, pages) for a pdf or plain text upload."""
    if uploaded_file is None:
        return "", []
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        pages = extract_text_from_pdf(uploaded_file)
        full_text = "\n\n".join(t for _, t in pages)
        return full_text, pages
    text = extract_plain_text(uploaded_file)
    return text, [(1, text)]


def chunk_pages(pages, chunk_size=900, overlap=150):
    chunks = []
    for page_num, text in pages:
        words = text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                chunks.append({"page": page_num, "text": chunk_text})
            if end >= len(words):
                break
            start = end - overlap
    return chunks


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks, embedder):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(
        texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(query, embedder, index, chunks, top_k=4):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({**chunks[idx], "score": float(score)})
    return results


# ============================================================================
# TOOL 1: EMAIL GENERATOR
# ============================================================================
def tool_email_generator():
    st.subheader("Email Generator")
    st.caption("Turn a topic and a few bullet points into a polished, ready to send email.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        with st.container(border=True):
            tone = st.selectbox(
                "Tone",
                [
                    "Professional", "Friendly", "Persuasive", "Formal",
                    "Casual", "Apologetic", "Enthusiastic", "Urgent",
                    "Empathetic", "Confident",
                ],
                key="email_tone",
            )
            email_type = st.selectbox(
                "Email type",
                [
                    "General", "Sales or Outreach", "Follow up", "Apology",
                    "Thank you", "Meeting request", "Announcement",
                    "Complaint or Escalation", "Job application or Cover letter",
                    "Networking",
                ],
                key="email_type",
            )
            length = st.select_slider(
                "Length", options=["Short", "Medium", "Long"], value="Medium", key="email_length"
            )
            sender_name = st.text_input("Your name (signature)", "", key="email_sender")
            recipient_name = st.text_input("Recipient name", "", key="email_recipient")
            subject_hint = st.text_input("Preferred subject line (optional)", "", key="email_subject")

            topic = st.text_input(
                "Topic",
                placeholder="e.g. Requesting a deadline extension for the Q3 report",
                key="email_topic",
            )
            points = st.text_area(
                "Key points to include (one per line)",
                placeholder=(
                    "e.g.\n"
                    "The client data arrived three days later than planned\n"
                    "We need five extra business days\n"
                    "Quality will not be affected"
                ),
                height=150,
                key="email_points",
            )

            model = model_picker(key="email_model")
            temperature = st.slider("Creativity", 0.0, 1.2, 0.7, 0.1, key="email_temp")

            generate = st.button("Generate Email", key="email_generate")

    with col2:
        with st.container(border=True):
            st.markdown("#### Generated Email")
            output_placeholder = st.empty()
            if "email_last" in st.session_state:
                output_placeholder.markdown(
                    f"<div class='ss_output'>{st.session_state['email_last']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                output_placeholder.markdown(
                    "<div class='ss_output' style='opacity:0.5;'>Your generated email will appear here.</div>",
                    unsafe_allow_html=True,
                )

    def build_prompt():
        bullet_points = "\n".join(f"- {line.strip()}" for line in points.splitlines() if line.strip())
        length_guide = {
            "Short": "under 100 words, two or three short paragraphs at most",
            "Medium": "roughly 120 to 200 words, well organized",
            "Long": "roughly 220 to 320 words, thorough but not padded",
        }[length]
        signature = sender_name.strip() if sender_name.strip() else "[Your Name]"
        greeting_name = recipient_name.strip() if recipient_name.strip() else "there"
        subject_instruction = (
            f'Use this exact subject line: "{subject_hint.strip()}"'
            if subject_hint.strip()
            else "Write a compelling, specific subject line, not generic."
        )
        return f"""You are an expert professional email copywriter. Write a high
quality, natural sounding email based on the details below. The email must
read as if a thoughtful, articulate person wrote it, never generic, robotic,
or filled with cliches like "I hope this email finds you well".

EMAIL TYPE: {email_type}
TONE: {tone}
TARGET LENGTH: {length_guide}
RECIPIENT NAME: {greeting_name}
SENDER SIGNATURE NAME: {signature}

TOPIC:
{topic.strip()}

KEY POINTS TO NATURALLY WEAVE IN (do not just list them, integrate them fluidly):
{bullet_points if bullet_points else "(none provided, infer sensible content from the topic)"}

REQUIREMENTS:
1. {subject_instruction}
2. Open with a natural, original greeting appropriate to the tone.
3. The body must have a clear purpose, logical flow, and a strong opening line.
4. Every key point above must be reflected in the email, in your own words.
5. Match the requested tone precisely and consistently throughout.
6. End with a clear, specific call to action or next step, then a closing and the sender name.
7. Use proper email formatting with short paragraphs and good whitespace.
8. Do not use placeholder brackets except for the sender or recipient names already given.
9. Output ONLY the email itself, starting with "Subject: ..." on the first line, then a blank line, then the body.
"""

    if generate:
        if not GROQ_API_KEY:
            st.error("Cannot generate: no Groq API key configured.")
        elif not topic.strip():
            st.warning("Please enter a topic for the email.")
        else:
            with st.spinner("Writing your email..."):
                try:
                    client = get_client()
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert professional email writer known for "
                                    "clear, persuasive, natural sounding emails. You never "
                                    "sound like a generic AI."
                                ),
                            },
                            {"role": "user", "content": build_prompt()},
                        ],
                        temperature=temperature,
                        max_tokens=900,
                    )
                    email_text = response.choices[0].message.content.strip()
                    st.session_state["email_last"] = email_text
                    output_placeholder.markdown(
                        f"<div class='ss_output'>{email_text}</div>", unsafe_allow_html=True
                    )
                    st.success("Email generated.")
                except Exception as e:
                    handle_groq_error(e)

    if "email_last" in st.session_state:
        st.download_button(
            "Download as text file",
            data=st.session_state["email_last"],
            file_name=f"email_{int(time.time())}.txt",
            mime="text/plain",
            key="email_download",
        )


# ============================================================================
# TOOL 2: LINKEDIN POST GENERATOR
# ============================================================================
def tool_linkedin_generator():
    st.subheader("LinkedIn Post Generator")
    st.caption("Generate tailored LinkedIn content in seconds.")

    with st.container(border=True):
        post_name = st.text_input(
            "Post name or internal reference",
            placeholder="e.g. Q3 Product Launch Announcement",
            key="li_name",
        )
        post_topic = st.text_area(
            "Post topic and key content points",
            placeholder="Detail your main idea, key takeaways, statistics, or story elements.",
            height=130,
            key="li_topic",
        )

        col1, col2 = st.columns(2)
        with col1:
            target_audience = st.selectbox(
                "Target audience",
                [
                    "Software Engineers and Developers",
                    "Entrepreneurs and Founders",
                    "Product Managers",
                    "Marketing and Sales Leaders",
                    "Recruiters and HR Professionals",
                    "C Suite and Executives",
                    "Job Seekers and Graduates",
                    "General Professional Audience",
                ],
                key="li_audience",
            )
            tone = st.selectbox(
                "Tone of post",
                [
                    "Professional and Authoritative",
                    "Conversational and Friendly",
                    "Thought Provoking and Analytical",
                    "Inspirational and Motivational",
                    "Storytelling and Personal",
                    "Persuasive and Direct",
                ],
                key="li_tone",
            )
        with col2:
            length = st.selectbox(
                "Post length",
                ["Short (50 to 100 words)", "Medium (100 to 250 words)", "Detailed (250 to 400 words)"],
                key="li_length",
            )
            include_hashtags = st.radio(
                "Include hashtags", options=["Yes", "No"], index=0, horizontal=True, key="li_hashtags"
            )

        model = model_picker(key="li_model")
        generate = st.button("Generate Post", key="li_generate")

    if generate:
        if not GROQ_API_KEY:
            st.error("Cannot generate: no Groq API key configured.")
        elif not post_topic.strip():
            st.warning("Please provide a topic or core points for the post.")
        else:
            prompt = f"""
You are an expert LinkedIn copywriter. Generate an engaging, highly effective
LinkedIn post using these parameters:

Internal reference: {post_name if post_name else "N/A"}
Topic and key points: {post_topic}
Target audience: {target_audience}
Tone: {tone}
Desired length: {length}
Include hashtags: {include_hashtags}

Structuring rules:
1. Hook: create a compelling opening line that encourages reading further.
2. Structure: use short sentences, clear line spacing, and bullet points where useful.
3. Call to action: end with a thoughtful question or prompt for engagement.
4. Emojis: do not use emojis in the post text.
5. Hashtags: {"Include three to five relevant hashtags at the bottom." if include_hashtags == "Yes" else "Do NOT include hashtags."}

Return ONLY the final LinkedIn post content, no introductory or concluding text.
"""
            try:
                with st.spinner("Generating post..."):
                    client = get_client()
                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a professional LinkedIn post writer."},
                            {"role": "user", "content": prompt},
                        ],
                        model=model,
                        temperature=0.7,
                        max_tokens=1024,
                    )
                    generated_post = response.choices[0].message.content.strip()
                    st.session_state["li_last"] = generated_post
            except Exception as e:
                handle_groq_error(e)

    if "li_last" in st.session_state:
        st.success("Post generated.")
        st.markdown("#### Generated LinkedIn Post")
        st.caption("Use the copy icon in the top right corner of the box below to copy your text.")
        st.code(st.session_state["li_last"], language=None)
        st.download_button(
            "Download as text file",
            data=st.session_state["li_last"],
            file_name=f"linkedin_post_{int(time.time())}.txt",
            mime="text/plain",
            key="li_download",
        )


# ============================================================================
# TOOL 3: PDF READER (RAG QUESTION AND ANSWER)
# ============================================================================
def tool_pdf_reader():
    st.subheader("PDF Reader")
    st.caption("Upload a PDF, then ask it anything, answered using retrieval augmented generation.")

    for k, v in {
        "pdf_chat_history": [],
        "pdf_chunks": None,
        "pdf_index": None,
        "pdf_name": None,
        "pdf_pages_count": 0,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_upload")
        with col2:
            model_label = st.selectbox(
                "Model (free tier)", list(MODEL_OPTIONS.keys()), key="pdf_model_label"
            )
            selected_model = MODEL_OPTIONS[model_label]
        with col3:
            top_k = st.slider("Chunks retrieved", 2, 8, 4, key="pdf_topk")

        process_col, clear_col = st.columns(2)
        with process_col:
            process_clicked = st.button("Process PDF", key="pdf_process")
        with clear_col:
            if st.button("Clear chat", key="pdf_clear"):
                st.session_state.pdf_chat_history = []
                st.rerun()

        if st.session_state.pdf_name:
            st.markdown(
                f"<span class='ss_chip'>{st.session_state.pdf_name}, "
                f"{st.session_state.pdf_pages_count} pages indexed</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<span class='ss_chip'>No PDF processed yet</span>", unsafe_allow_html=True)

    if process_clicked:
        if uploaded_file is None:
            st.error("Please upload a PDF first.")
        else:
            with st.spinner("Reading and indexing your PDF, this only takes a moment..."):
                embedder = load_embedder()
                pages = extract_text_from_pdf(uploaded_file)
                if not pages:
                    st.error("Could not extract any text from this PDF. It may be a scanned or image only PDF.")
                else:
                    chunks = chunk_pages(pages)
                    index = build_index(chunks, embedder)
                    st.session_state.pdf_chunks = chunks
                    st.session_state.pdf_index = index
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.pdf_pages_count = len(pages)
                    st.session_state.pdf_chat_history = []
            st.success(f"Indexed {uploaded_file.name} ({len(pages)} pages).")

    if st.session_state.pdf_index is None:
        st.info("Upload a PDF and click Process PDF to get started.")
        return

    for turn in st.session_state.pdf_chat_history:
        css_class = "ss_chat_user" if turn["role"] == "user" else "ss_chat_ai"
        st.markdown(f'<div class="{css_class}">{turn["content"]}</div>', unsafe_allow_html=True)
        if turn.get("sources"):
            chips = "".join(f'<span class="ss_chip">Page {s}</span>' for s in turn["sources"])
            st.markdown(chips, unsafe_allow_html=True)

    question = st.chat_input("Ask a question about your PDF...")
    if question:
        st.session_state.pdf_chat_history.append({"role": "user", "content": question})
        st.markdown(f'<div class="ss_chat_user">{question}</div>', unsafe_allow_html=True)

        client = get_client()
        if client is None:
            st.error("Groq API key missing, cannot generate an answer.")
        else:
            embedder = load_embedder()
            results = retrieve(
                question, embedder, st.session_state.pdf_index, st.session_state.pdf_chunks, top_k=top_k
            )
            pages_used = sorted({r["page"] for r in results})
            context_text = "\n\n---\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in results)

            system_prompt = textwrap.dedent(
                """
                You are a precise, helpful assistant that answers questions strictly using
                the provided PDF excerpts. Rules:
                Only use information found in the provided context.
                If the answer is not contained in the context, say so honestly instead of guessing.
                Cite the page number(s) you used, like (Page 3), where relevant.
                Be concise and well organized.
                """
            ).strip()

            messages = [{"role": "system", "content": system_prompt}]
            for turn in st.session_state.pdf_chat_history[-6:]:
                messages.append({"role": turn["role"], "content": turn["content"]})
            messages.append(
                {"role": "user", "content": f"Context from the PDF:\n\n{context_text}\n\nQuestion: {question}"}
            )

            placeholder = st.empty()
            answer_text = ""
            try:
                stream = client.chat.completions.create(
                    model=selected_model, messages=messages, temperature=0.2, max_tokens=1024, stream=True
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    answer_text += delta
                    placeholder.markdown(f'<div class="ss_chat_ai">{answer_text}</div>', unsafe_allow_html=True)
            except Exception as e:
                handle_groq_error(e)
                answer_text = "Sorry, something went wrong while generating an answer."
                placeholder.markdown(f'<div class="ss_chat_ai">{answer_text}</div>', unsafe_allow_html=True)

            chips = "".join(f'<span class="ss_chip">Page {p}</span>' for p in pages_used)
            st.markdown(chips, unsafe_allow_html=True)
            st.session_state.pdf_chat_history.append(
                {"role": "assistant", "content": answer_text, "sources": pages_used}
            )


# ============================================================================
# TOOL 4: SCHEDULE MAKER
# ============================================================================
def tool_schedule_maker():
    st.subheader("Schedule Maker")
    st.caption("Turn your tasks and deadlines into a clear study or work schedule.")

    with st.container(border=True):
        tasks = st.text_area(
            "Tasks or subjects (one per line, with rough time needed or a deadline)",
            placeholder=(
                "e.g.\n"
                "Math homework, two hours, due Monday\n"
                "Read chapters four and five for history, ninety minutes\n"
                "Study for chemistry quiz, three hours, quiz on Friday"
            ),
            height=150,
            key="sched_tasks",
        )
        col1, col2 = st.columns(2)
        with col1:
            hours_per_day = st.slider("Available study hours per day", 1, 12, 4, key="sched_hours")
        with col2:
            days = st.slider("Number of days to plan for", 1, 14, 7, key="sched_days")
        format_choice = st.radio(
            "Preferred format", ["Daily blocks", "Weekly table"], horizontal=True, key="sched_format"
        )
        model = model_picker(key="sched_model")
        generate = st.button("Generate Schedule", key="sched_generate")

    if generate:
        if not GROQ_API_KEY:
            st.error("Cannot generate: no Groq API key configured.")
        elif not tasks.strip():
            st.warning("Please list at least one task or subject.")
        else:
            prompt = f"""You are an expert academic planner. Build a realistic study
schedule from the tasks below.

TASKS:
{tasks.strip()}

CONSTRAINTS:
Available study hours per day: {hours_per_day}
Number of days to plan for: {days}
Preferred output format: {format_choice}

RULES:
1. Respect stated deadlines, put urgent or deadline heavy tasks earlier.
2. Break long tasks into shorter sessions across multiple days rather than one long block.
3. Leave short breaks between sessions where sensible.
4. If the format is Weekly table, present it as a plain text table with Day, Time, Task columns.
5. If the format is Daily blocks, present each day as a heading followed by a simple list of time blocks and tasks.
6. Do not add commentary before or after the schedule itself.
"""
            with st.spinner("Building your schedule..."):
                try:
                    client = get_client()
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a precise, realistic academic planner."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.4,
                        max_tokens=1200,
                    )
                    st.session_state["sched_last"] = response.choices[0].message.content.strip()
                except Exception as e:
                    handle_groq_error(e)

    if "sched_last" in st.session_state:
        st.markdown(f"<div class='ss_output'>{st.session_state['sched_last']}</div>", unsafe_allow_html=True)
        st.download_button(
            "Download as text file",
            data=st.session_state["sched_last"],
            file_name=f"schedule_{int(time.time())}.txt",
            mime="text/plain",
            key="sched_download",
        )


# ============================================================================
# TOOL 5: GRAMMAR AND CLARITY CHECKER
# ============================================================================
def tool_grammar_checker():
    st.subheader("Grammar and Clarity Checker")
    st.caption("Paste your text and get a corrected version plus a plain summary of what changed.")

    with st.container(border=True):
        text_input = st.text_area(
            "Paste the text you want checked",
            height=220,
            placeholder="Paste an essay, email, or any other text here...",
            key="grammar_text",
        )
        style = st.selectbox(
            "Preferred style",
            ["Academic", "Casual", "Business", "Concise", "No preference"],
            key="grammar_style",
        )
        model = model_picker(key="grammar_model")
        generate = st.button("Check My Text", key="grammar_generate")

    if generate:
        if not GROQ_API_KEY:
            st.error("Cannot generate: no Groq API key configured.")
        elif not text_input.strip():
            st.warning("Please paste some text first.")
        else:
            prompt = f"""You are an expert writing editor. Correct the grammar,
spelling, punctuation, and clarity of the text below, while preserving the
author's original meaning, voice, and intent. Prefer the following style
where it does not conflict with the author's voice: {style}.

ORIGINAL TEXT:
{text_input.strip()}

Return your answer in exactly this format, with these two section headers
and nothing else:

CORRECTED TEXT:
[the corrected text, ready to use]

CHANGES MADE:
[a short bullet list explaining the main grammar, clarity, or tone changes]
"""
            with st.spinner("Checking your text..."):
                try:
                    client = get_client()
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a careful, precise writing editor."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=1400,
                    )
                    st.session_state["grammar_last"] = response.choices[0].message.content.strip()
                except Exception as e:
                    handle_groq_error(e)

    if "grammar_last" in st.session_state:
        st.markdown(
            f"<div class='ss_output'>{st.session_state['grammar_last']}</div>", unsafe_allow_html=True
        )
        st.download_button(
            "Download as text file",
            data=st.session_state["grammar_last"],
            file_name=f"grammar_check_{int(time.time())}.txt",
            mime="text/plain",
            key="grammar_download",
        )


# ============================================================================
# TOOL 6: FLASHCARD AND QUIZ GENERATOR
# ============================================================================
def parse_flashcards(raw_text):
    cards = []
    blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
    for block in blocks:
        front, back = None, None
        for line in block.splitlines():
            if line.lower().startswith("front:"):
                front = line.split(":", 1)[1].strip()
            elif line.lower().startswith("back:"):
                back = line.split(":", 1)[1].strip()
        if front and back:
            cards.append({"front": front, "back": back})
    return cards


def tool_flashcard_quiz_generator():
    st.subheader("Flashcard and Quiz Generator")
    st.caption("Turn your notes into flashcards you can flip through, straight from pasted text or a PDF or text file.")

    with st.container(border=True):
        source_choice = st.radio("Source", ["Paste text", "Upload file"], horizontal=True, key="fc_source")
        source_text = ""
        if source_choice == "Paste text":
            source_text = st.text_area("Paste your notes", height=200, key="fc_text")
        else:
            uploaded = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"], key="fc_upload")
            if uploaded is not None:
                source_text, _ = get_source_text(uploaded)

        num_cards = st.slider("Number of flashcards", 3, 20, 8, key="fc_count")
        model = model_picker(key="fc_model")
        generate = st.button("Generate Flashcards", key="fc_generate")

    if generate:
        if not GROQ_API_KEY:
            st.error("Cannot generate: no Groq API key configured.")
        elif not source_text.strip():
            st.warning("Please paste some notes or upload a file first.")
        else:
            prompt = f"""You are an expert study coach. Create exactly {num_cards}
flashcards from the notes below. Each flashcard should test one clear idea.

NOTES:
{source_text[:8000]}

Return the flashcards in exactly this format, one flashcard per block,
separated by a single blank line, and nothing else:

Front: [a short question or prompt]
Back: [the answer or explanation]
"""
            with st.spinner("Building your flashcards..."):
                try:
                    client = get_client()
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a precise, helpful study coach."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.4,
                        max_tokens=1600,
                    )
                    raw = response.choices[0].message.content.strip()
                    cards = parse_flashcards(raw)
                    st.session_state["fc_cards"] = cards
                    st.session_state["fc_raw"] = raw
                    st.session_state["fc_index"] = 0
                    st.session_state["fc_flip"] = False
                except Exception as e:
                    handle_groq_error(e)

    cards = st.session_state.get("fc_cards")
    if cards:
        idx = st.session_state.get("fc_index", 0)
        idx = max(0, min(idx, len(cards) - 1))
        card = cards[idx]
        flip = st.session_state.get("fc_flip", False)

        st.markdown(
            f"<div class='ss_output'>Card {idx + 1} of {len(cards)}<br><br>"
            f"{card['back'] if flip else card['front']}</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Previous", key="fc_prev"):
                st.session_state["fc_index"] = (idx - 1) % len(cards)
                st.session_state["fc_flip"] = False
                st.rerun()
        with c2:
            if st.button("Flip", key="fc_flip_btn"):
                st.session_state["fc_flip"] = not flip
                st.rerun()
        with c3:
            if st.button("Next", key="fc_next"):
                st.session_state["fc_index"] = (idx + 1) % len(cards)
                st.session_state["fc_flip"] = False
                st.rerun()

        st.download_button(
            "Download all flashcards as text",
            data=st.session_state.get("fc_raw", ""),
            file_name=f"flashcards_{int(time.time())}.txt",
            mime="text/plain",
            key="fc_download",
        )
    elif "fc_raw" in st.session_state:
        st.info("The model's output could not be parsed into cards, showing the raw text instead.")
        st.markdown(f"<div class='ss_output'>{st.session_state['fc_raw']}</div>", unsafe_allow_html=True)


# ============================================================================
# TOOL 7: RESUME MAKER
# ============================================================================
def tool_resume_maker():
    st.subheader("Resume Maker")
    st.caption("Turn your background into a tailored, well organized resume.")

    with st.container(border=True):
        target_role = st.text_input("Target role or internship", key="resume_role")
        education = st.text_area(
            "Education (school, degree, graduation year, relevant coursework)",
            height=100,
            key="resume_education",
        )
        experience = st.text_area(
            "Work, internship, or project experience (one entry per line)",
            height=150,
            key="resume_experience",
        )
        skills = st.text_area("Skills (comma separated)", height=80, key="resume_skills")
        existing_resume = st.text_area(
            "Existing resume text to improve (optional, leave blank to start fresh)",
            height=150,
            key="resume_existing",
        )
        model = model_picker(key="resume_model")
        generate = st.button("Generate Resume", key="resume_generate")

    if generate:
        if not GROQ_API_KEY:
            st.error("Cannot generate: no Groq API key configured.")
        elif not (education.strip() or experience.strip() or existing_resume.strip()):
            st.warning("Please provide at least your education or experience.")
        else:
            prompt = f"""You are an expert resume writer. Write a clear, well
organized, plain text resume tailored to the target role below. Use strong,
specific action verbs and quantify achievements where possible.

TARGET ROLE: {target_role if target_role.strip() else "General"}

EDUCATION:
{education.strip() or "(none provided)"}

EXPERIENCE:
{experience.strip() or "(none provided)"}

SKILLS:
{skills.strip() or "(none provided)"}

EXISTING RESUME TO IMPROVE, IF PROVIDED:
{existing_resume.strip() or "(none provided, write a new resume from the details above)"}

FORMAT RULES:
1. Use plain text section headers such as EDUCATION, EXPERIENCE, SKILLS.
2. Use short bullet lines under each experience entry, starting with an action verb.
3. Do not invent facts, degrees, dates, or companies that were not provided.
4. Keep it to a single page worth of content.
5. Output only the resume text, no commentary before or after.
"""
            with st.spinner("Building your resume..."):
                try:
                    client = get_client()
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a precise, honest resume writer."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.4,
                        max_tokens=1400,
                    )
                    st.session_state["resume_last"] = response.choices[0].message.content.strip()
                except Exception as e:
                    handle_groq_error(e)

    if "resume_last" in st.session_state:
        st.markdown(f"<div class='ss_output'>{st.session_state['resume_last']}</div>", unsafe_allow_html=True)
        st.download_button(
            "Download as text file",
            data=st.session_state["resume_last"],
            file_name=f"resume_{int(time.time())}.txt",
            mime="text/plain",
            key="resume_download",
        )


# ============================================================================
# TOOL 8: NOTES SUMMARIZER
# ============================================================================
def tool_notes_summarizer():
    st.subheader("Notes Summarizer")
    st.caption("Condense long notes or readings into a summary you can actually study from.")

    with st.container(border=True):
        source_choice = st.radio("Source", ["Paste text", "Upload file"], horizontal=True, key="sum_source")
        source_text = ""
        if source_choice == "Paste text":
            source_text = st.text_area("Paste your notes", height=220, key="sum_text")
        else:
            uploaded = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"], key="sum_upload")
            if uploaded is not None:
                source_text, _ = get_source_text(uploaded)

        style = st.selectbox(
            "Summary style",
            ["Bullet points", "Short paragraph", "Exam cram sheet"],
            key="sum_style",
        )
        length_choice = st.select_slider(
            "Summary length", options=["Short", "Medium", "Long"], value="Medium", key="sum_length"
        )
        model = model_picker(key="sum_model")
        generate = st.button("Summarize", key="sum_generate")

    if generate:
        if not GROQ_API_KEY:
            st.error("Cannot generate: no Groq API key configured.")
        elif not source_text.strip():
            st.warning("Please paste some notes or upload a file first.")
        else:
            length_guide = {
                "Short": "a very brief summary, the essentials only",
                "Medium": "a moderate length summary covering all main points",
                "Long": "a thorough summary that still condenses the source significantly",
            }[length_choice]

            prompt = f"""You are an expert study assistant. Summarize the notes
below in the requested style and length. Keep only the information a
student would actually need to remember or review.

STYLE: {style}
LENGTH: {length_guide}

NOTES:
{source_text[:10000]}

Output only the summary itself, no commentary before or after.
"""
            with st.spinner("Summarizing..."):
                try:
                    client = get_client()
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a precise, helpful study assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=1200,
                    )
                    st.session_state["sum_last"] = response.choices[0].message.content.strip()
                except Exception as e:
                    handle_groq_error(e)

    if "sum_last" in st.session_state:
        st.markdown(f"<div class='ss_output'>{st.session_state['sum_last']}</div>", unsafe_allow_html=True)
        st.download_button(
            "Download as text file",
            data=st.session_state["sum_last"],
            file_name=f"summary_{int(time.time())}.txt",
            mime="text/plain",
            key="sum_download",
        )


# ============================================================================
# HORIZONTAL NAV BAR AND ROUTING
# ============================================================================
tab_names = [
    "Email Generator",
    "LinkedIn Post Generator",
    "PDF Reader",
    "Schedule Maker",
    "Grammar and Clarity Checker",
    "Flashcard and Quiz Generator",
    "Resume Maker",
    "Notes Summarizer",
]

tabs = st.tabs(tab_names)

with tabs[0]:
    tool_email_generator()
with tabs[1]:
    tool_linkedin_generator()
with tabs[2]:
    tool_pdf_reader()
with tabs[3]:
    tool_schedule_maker()
with tabs[4]:
    tool_grammar_checker()
with tabs[5]:
    tool_flashcard_quiz_generator()
with tabs[6]:
    tool_resume_maker()
with tabs[7]:
    tool_notes_summarizer()

st.markdown(
    """
    <div style="text-align:center; margin-top:2.5rem; color:#7a75a0; font-size:0.85rem;">
        Built with Streamlit and Groq. Your API key stays on the server, it is never sent to the browser.
    </div>
    """,
    unsafe_allow_html=True,
)
