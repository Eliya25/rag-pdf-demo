"""Streamlit entry point for the page-aware RAG pipeline."""
import hashlib
import io
import logging
import os

import streamlit as st
from dotenv import load_dotenv

from config import load_settings
from demo_safety import DemoSafetyGuard, demo_capabilities, question_within_limit
from local_embedding import LocalEmbedding
from pdf_reader import PdfReader
from prompts import build_messages
from provider_factory import create_provider
from upload_lifecycle import read_upload_bytes, should_index_upload
from ui_errors import (GENERATION_ERROR_MESSAGE,
                       PUBLIC_DEMO_GENERATION_ERROR_MESSAGE,
                       safe_value_error_message, stream_answer)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(page_title="RAG PDF Assistant", page_icon="📄", layout="wide")
try:
    for name in ("LLM_PROVIDER", "GEMINI_MODEL", "GEMINI_DEMO_ENABLED"):
        if name in st.secrets:
            os.environ[name] = str(st.secrets[name])
except FileNotFoundError:
    pass

try:
    settings = load_settings()
except (ValueError, TypeError):
    st.error("Configuration error. Check the selected provider and public demo settings.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return LocalEmbedding()


@st.cache_resource(show_spinner=False)
def load_local_model():
    from local_llm import LocalLLMProvider
    return LocalLLMProvider()


@st.cache_resource(show_spinner=False)
def load_demo_safety_guard():
    return DemoSafetyGuard()


def get_provider():
    def load_api_key():
        try:
            return os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        except FileNotFoundError:
            return os.environ.get("GEMINI_API_KEY")

    return create_provider(settings, load_api_key, load_local_model)


def show_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for doc, distance in sources:
            st.caption(f"Page {doc['page']} · Chunk {doc['chunk']} · distance {distance:.3f}")
            st.text(doc["content"])


for key, default in {"pdf_id": None, "pdf_name": None, "index": None,
                     "chat": [], "pages": 0, "chunks": 0,
                     "demo_question_count": 0,
                     "demo_last_request_at": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("Ask Your PDF")
demo_guard = load_demo_safety_guard() if settings.public_demo_mode else None
can_upload, _ = demo_capabilities(
    settings.public_demo_mode,
    settings.gemini_demo_enabled,
    st.session_state.index is not None,
)
with st.sidebar:
    st.subheader("Model")
    st.write(f"Provider: **{settings.provider}**")
    st.write(f"Model: **{settings.gemini_model if settings.provider == 'gemini' else 'Qwen/Qwen2.5-3B-Instruct'}**")
    if settings.provider == "gemini":
        st.info("Hosted mode sends your question, selected recent chat, and retrieved document excerpts to Gemini. The original PDF file is not sent. Do not use confidential documents.")
    else:
        st.info("Local mode processes the PDF, retrieval, and answer generation in this app process with MiniLM and Qwen; it does not send document or question content to a hosted generation provider.")
    if settings.public_demo_mode and not settings.gemini_demo_enabled:
        st.warning("The owner has temporarily disabled Gemini requests for this demo.")
    upload = st.file_uploader(
        "Upload a text PDF",
        type=["pdf"],
        disabled=not can_upload,
    )
    if upload is not None and can_upload:
        max_upload_bytes = settings.max_file_mb * 1024 * 1024
        if upload.size > max_upload_bytes:
            st.session_state.update(pdf_id=None, pdf_name=None, index=None,
                                    chat=[], pages=0, chunks=0)
            st.error(f"PDF is too large. The maximum upload size is {settings.max_file_mb} MiB.")
        elif settings.public_demo_mode and not demo_guard.try_acquire_indexing():
            st.warning("Another document is being indexed. Please try again shortly.")
        else:
            acquired_indexing = settings.public_demo_mode
            try:
                data = read_upload_bytes(upload, max_upload_bytes)
                file_id = hashlib.sha256(data).hexdigest()
                # Retry identical uploads when a previous extraction or index attempt failed.
                if should_index_upload(file_id, st.session_state.pdf_id, st.session_state.index):
                    st.session_state.update(pdf_id=file_id, pdf_name=upload.name, index=None,
                                            chat=[], pages=0, chunks=0)
                    try:
                        with st.status("Extracting and indexing PDF...") as status:
                            reader = PdfReader(
                                io.BytesIO(data), settings.max_file_mb, settings.max_pages,
                                max_text_chars=settings.max_text_chars,
                            )
                            chunks = reader.get_chunks(settings.chunk_size, settings.chunk_overlap,
                                                       settings.max_chunks)
                            if not chunks:
                                raise ValueError("This PDF appears to contain no extractable text.")
                            index = load_embedding_model().with_empty_index()
                            index.build_index(chunks)
                            st.session_state.update(index=index, pages=reader.page_count,
                                                    chunks=len(chunks))
                            status.update(label="Document indexed", state="complete")
                    except ValueError:
                        st.error("Could not process this PDF. Check that it is valid, unencrypted, and within the configured size, page, text, and chunk limits.")
                    except Exception:
                        logger.error("Document indexing failed")
                        st.error("Could not process this PDF. Check the file and try again.")
                if st.session_state.index is not None:
                    st.success(f"{st.session_state.pdf_name} | {st.session_state.pages} pages | {st.session_state.chunks} chunks")
            except ValueError:
                st.session_state.update(pdf_id=None, pdf_name=None, index=None,
                                        chat=[], pages=0, chunks=0)
                st.error("Could not process this PDF. Check that it is valid and within the configured limits.")
            finally:
                if acquired_indexing:
                    demo_guard.release_indexing()
    else:
        st.info("Upload a PDF to begin.")

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

_, can_generate = demo_capabilities(
    settings.public_demo_mode,
    settings.gemini_demo_enabled,
    st.session_state.index is not None,
)
question = st.chat_input(
    "Ask a question about the document",
    disabled=not can_generate,
)
if question:
    generation_acquired = False
    admitted = True
    if settings.public_demo_mode:
        if not question_within_limit(question, settings.max_question_chars):
            st.warning(f"Questions must be {settings.max_question_chars} characters or fewer.")
            admitted = False
        elif not demo_guard.try_acquire_generation():
            st.warning("Another answer is being generated. Please try again shortly.")
            admitted = False
        else:
            generation_acquired = True
            reason, admitted_at = demo_guard.admit_gemini_request(
                st.session_state.demo_question_count,
                st.session_state.demo_last_request_at,
            )
            if reason is not None:
                demo_guard.release_generation()
                generation_acquired = False
                messages = {
                    "session_limit": "This session has reached its question limit.",
                    "cooldown": "Please wait before sending another question.",
                    "process_limit": "The demo has reached its request limit and is temporarily unavailable.",
                }
                st.warning(messages[reason])
                admitted = False
            else:
                st.session_state.demo_question_count += 1
                st.session_state.demo_last_request_at = admitted_at

    if admitted:
        try:
            history = st.session_state.chat[-settings.history_messages:] if settings.history_messages else []
            previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
            search_query = f"{previous}\n{question}" if previous else question
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                answer_slot = st.empty()
                try:
                    sources = st.session_state.index.search(search_query, settings.top_k)
                    if not sources:
                        raise ValueError("No relevant text was found in this document.")
                    messages = build_messages(
                        question, st.session_state.index.format_context(sources), history,
                    )
                    provider = get_provider()
                    answer = stream_answer(provider, messages, settings.max_generation_tokens,
                                           answer_slot, st.error, logger,
                                           error_message=(
                                               PUBLIC_DEMO_GENERATION_ERROR_MESSAGE
                                               if settings.public_demo_mode
                                               else GENERATION_ERROR_MESSAGE
                                           ))
                    if answer is not None:
                        show_sources(sources)
                        st.session_state.chat.append({"role": "assistant", "content": answer,
                                                      "sources": sources})
                except ValueError as error:
                    answer_slot.empty()
                    message = safe_value_error_message(
                        error,
                        {"No relevant text was found in this document.",
                         "GEMINI_API_KEY is required for Gemini mode",
                         "A question and retrieved context are required",
                         "Public Gemini demo is disabled",
                         "Public Gemini demo configuration is invalid",
                         "Public demo mode requires Gemini"},
                        "Could not prepare the request. Check the document and model configuration.",
                    )
                    if message != str(error):
                        logger.error("Request validation failed")
                    st.error(message)
                except Exception:
                    answer_slot.empty()
                    logger.error("Retrieval failed")
                    st.error("Could not retrieve relevant text. Check the document and try again.")
        finally:
            if generation_acquired:
                demo_guard.release_generation()
