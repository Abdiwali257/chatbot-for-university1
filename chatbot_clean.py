"""Conversational SIMAD University RAG chatbot."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from sentence_transformers import SentenceTransformer

from academic_data import (
    all_courses,
    administration_context,
    course_program_name,
    display_program_name,
    faculty_programs,
    find_courses,
    find_named_courses,
    find_program_parent,
    find_programs,
    normalize_academic_query,
    semester_number,
    tuition_records,
)

PROJECT_DIR = Path(__file__).resolve().parent
DB_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "simad_knowledge_base"

load_dotenv(PROJECT_DIR / ".env")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MAX_DISTANCE = float(os.getenv("MAX_RETRIEVAL_DISTANCE", "0.78"))
STOP_WORDS = {
    "a",
    "about",
    "an",
    "are",
    "at",
    "available",
    "can",
    "does",
    "do",
    "for",
    "give",
    "has",
    "have",
    "how",
    "in",
    "is",
    "me",
    "of",
    "offer",
    "offers",
    "please",
    "simad",
    "the",
    "their",
    "them",
    "to",
    "university",
    "to",
    "what",
    "who",
    "which",
}

STRICT_SEMANTIC_DISTANCE = 0.42
MAX_MODEL_HISTORY = int(os.getenv("MAX_MODEL_HISTORY_MESSAGES", "60"))
MAX_USER_MESSAGE_LENGTH = int(os.getenv("MAX_USER_MESSAGE_LENGTH", "1000"))
NOT_FOUND_MESSAGE = (
    "I could not find verified information about that in the available SIMAD data."
)


def specific_not_found(question: str, language: str = "English") -> str:
    """Return a not-found message that mentions the topic keyword when possible."""
    if language == "Somali":
        return SOMALI_NOT_FOUND_MESSAGE
    topic_match = re.search(
        r"\b(dean|rector|faculty|faculties|program|course|department|scholarship|"
        r"fee|tuition|club|library|admission|requirement|founder|history|campus|"
        r"service|staff|contact|email|phone|address|date|year|semester|"
        r"gpa|grade|duration|accreditation|vision|mission|research|lab|exchange)s?\b",
        question.lower(),
    )
    if topic_match:
        topic = topic_match.group(1).rstrip("s").title()
        return f"I could not find verified information about {topic} in SIMAD's available records."
    return NOT_FOUND_MESSAGE


GENERATION_UNAVAILABLE_MESSAGE = (
    "The answer-generation service is temporarily unavailable. Please try again shortly."
)
OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions related to SIMAD University."
)
SOMALI_NOT_FOUND_MESSAGE = (
    "Ma helin macluumaad la xaqiijiyay oo ku saabsan taas xogta SIMAD ee la hayo."
)
SOMALI_OUT_OF_SCOPE_MESSAGE = (
    "Waxaan ka jawaabi karaa oo keliya su'aalaha la xiriira SIMAD University."
)
SOMALI_SMALL_TALK_MESSAGE = (
    "Waan ku caawin karaa. Maxaad rabtaa inaad ka ogaato SIMAD University?"
)
GENERAL_SIMAD_PATTERN = re.compile(
    r"^(?:give me (?:an? |some )?information about|tell me (?:some |more )?about|"
    r"what is|describe)\s+simad(?: university)?[?!. ]*$",
    re.I,
)
ANSWER_TRANSFORM_FALLBACK_PATTERN = re.compile(
    r"^(?:summari[sz]e(?: this| it| that)?|make (?:this|it|that) short(?:er)?|"
    r"make (?:this|it|that) \d+ lines?|explain (?:this |it |that )?again|"
    r"explain (?:this |it |that )?(?:simply|clearly)|"
    r"what (?:do )?you mean(?: by this| by that)?|what you mean this|"
    r"for what question|which question was that for|repeat|too long|"
    r"i (?:do not|don't) understand|make (?:this|it|that) clear(?:er)?|"
    r"make (?:this|it|that) clean(?:er)?|"
    r"say (?:this|it|that) simply|give me (?:only )?(?:the )?important points|"
    r"too short|expand(?: this| it| that)?|add (?:a little |some )?more detail|"
    r"clarify(?: this| it| that)?|"
    r"what was that about|this is messy|that is messy|messy)[?!. ]*$",
    re.I,
)
FOLLOW_UP_ACTION_PROTOTYPES = {
    "summarize": (
        "summarize the previous answer",
        "shorten the previous response",
        "give only the important points from that answer",
        "the previous answer is too long",
        "make the response more concise",
        "condense the response to the essential details",
        "reduce the previous answer to key points",
    ),
    "simplify": (
        "explain the previous answer in simple words",
        "use easier words for the previous response",
        "make the answer student friendly",
    ),
    "clean": (
        "rewrite the previous answer cleanly and organize it",
        "the previous answer is messy",
        "make the previous answer clean and easy to read",
    ),
    "clarify": (
        "clarify the previous response",
        "I do not understand the previous answer",
        "explain that again more clearly",
    ),
    "expand": (
        "the previous answer is too short",
        "expand the previous answer slightly",
        "add a little more useful detail from the previous answer",
    ),
    "repeat": (
        "repeat the previous answer",
        "say the answer again",
        "show me the previous response again",
    ),
    "topic": (
        "what question was the previous answer for",
        "what was that answer about",
        "remind me what topic we were discussing",
    ),
}
MESSAGE_INTENT_PROTOTYPES = {
    "new_simad_question": (
        "what are SIMAD University admission requirements",
        "tell me about a SIMAD University service",
        "who leads SIMAD University",
        "what programs and courses does SIMAD offer",
    ),
    "follow_up_answer": tuple(
        prototype
        for prototypes in FOLLOW_UP_ACTION_PROTOTYPES.values()
        for prototype in prototypes
    ),
    "small_talk": (
        "hello how are you",
        "thank you for your help",
        "that is great",
        "okay no problem",
    ),
    "out_of_scope": (
        "who won the football match",
        "how do I cook food",
        "what is today's weather",
        "write unrelated computer code",
    ),
}
CONVERSATION_INTENT_PROTOTYPES = {
    "greeting": (
        "a user greets the assistant",
        "hello there good morning",
        "starting a friendly chat",
    ),
    "assistant_mood": (
        "the user asks how the assistant is feeling",
        "checking whether the assistant is okay",
        "asking how are you doing",
    ),
    "assistant_identity": (
        "the user asks who the assistant is",
        "the user asks what kind of assistant this is",
        "asking the assistant to introduce itself",
    ),
    "chatbot_capability": (
        "the user asks what the assistant can help with",
        "asking the assistant's capabilities",
        "asking whether the assistant can help",
    ),
    "thanks": (
        "the user thanks the assistant",
        "the user appreciates the help",
        "the user compliments the assistant",
    ),
    "goodbye": (
        "the user says goodbye",
        "ending the conversation politely",
        "the user is leaving the chat",
    ),
    "conversation_control": (
        "the user wants to skip or stop the current topic",
        "the user says okay or confirms",
        "the user says never mind or forget it",
    ),
    "frustration": (
        "the user is frustrated or upset with the assistant",
        "the user uses insulting or angry language",
        "the user wants the assistant to stop talking",
    ),
    "memory": (
        "the user asks whether the assistant remembers the chat",
        "the user asks what the previous question was",
        "the user asks what they were discussing",
    ),
    "user_identity": (
        "the user asks who they are",
        "the user asks whether the assistant remembers their name",
        "the user tells the assistant their name",
    ),
}
CONVERSATION_MESSAGE_INTENTS = frozenset(CONVERSATION_INTENT_PROTOTYPES)
SEMANTIC_TOPIC_PROTOTYPES = {
    "leadership": (
        "who leads SIMAD University",
        "people in charge of the university",
        "university leaders management administration officials and heads",
        "university board senate rector president and senior leadership",
        "who runs and manages the institution",
        "what does the university board do",
        "responsibilities of the board of trustees",
        "the university senior management team",
    ),
    "admissions": (
        "how to join and apply to SIMAD University",
        "admission requirements and registration process",
        "what applicants need before enrolling",
    ),
    "academics": (
        "SIMAD faculties degree programs courses and curriculum",
        "what can students study at the university",
        "academic schools departments programs and subjects",
    ),
    "tuition": (
        "SIMAD tuition fees and study costs",
        "how much students pay per semester",
    ),
    "scholarships": (
        "SIMAD scholarships financial aid and who qualifies",
        "funding support for students",
    ),
    "campus_services": (
        "SIMAD campus services library and student support",
        "facilities and services available to students",
    ),
    "exchange": (
        "SIMAD international exchange and partner university opportunities",
        "students studying abroad through exchange programs",
    ),
    "research": (
        "SIMAD research center conferences innovation and publications",
        "university research and consultancy",
    ),
    "student_life": (
        "SIMAD student clubs activities cultural week and extracurricular life",
        "student organizations and activities outside class",
    ),
    "history": (
        "SIMAD history founders establishment timeline and former rectors",
        "who created SIMAD and how it developed",
    ),
    "governance": (
        "SIMAD senate governance board and university policies",
        "how the university is governed",
    ),
    "vision": (
        "SIMAD vision mission values and goals",
        "what the university aims to achieve",
    ),
    "accreditation": (
        "SIMAD accreditation rankings and memberships",
        "recognition and associations of the university",
    ),
    "grading": (
        "SIMAD grading system GPA grades and academic performance",
        "how student results and GPA are calculated",
        "how marks and results are worked out",
    ),
    "disability_support": (
        "SIMAD disability support and help for students with special needs",
        "accessibility services for disabled students",
    ),
    "student_conduct": (
        "SIMAD student code of conduct rules and discipline",
        "student behavior policies",
    ),
    "overview": (
        "general information about SIMAD University",
        "tell me about SIMAD University",
    ),
}
SEMANTIC_TOPIC_SOURCES = {
    "leadership": ("THE SENATE.pdf", "RECTOR.pdf", "SIMAD HISTORY.pdf"),
    "admissions": ("ADMISSION BROCHURE.pdf", "Transfer Applications.pdf"),
    "tuition": ("TUTION FEES.pdf", "ADMISSION BROCHURE.pdf"),
    "scholarships": ("Scholarships.pdf",),
    "campus_services": ("CAMPUS SERVICES.pdf", "Disability Support Services (DSS).pdf"),
    "exchange": ("EXCHANGE PROGRAM.pdf",),
    "research": ("SIMAD RESEARCH.pdf", "SIMAD CONFERENCES.pdf", "SIMAD I-LAB.pdf"),
    "student_life": ("CLUBS.pdf", "Co-Curricular Programs.pdf", "CULTURAL WEEK.pdf", "Extracurricular Activities.pdf"),
    "history": ("SIMAD HISTORY.pdf",),
    "governance": ("THE SENATE.pdf", "INFORMATION DISCOLURE.pdf", "STUDENT CODE OF CONDUCT.pdf"),
    "vision": ("VISION.pdf", "SIMAD UNIVERSITY GENERAL INFORMATION.pdf"),
    "accreditation": ("Accreditation, Ranking, & Memberships.pdf",),
    "grading": ("Grading System and GPA.pdf",),
    "disability_support": ("Disability Support Services (DSS).pdf",),
    "student_conduct": ("STUDENT CODE OF CONDUCT.pdf",),
    "overview": ("SIMAD UNIVERSITY GENERAL INFORMATION.pdf",),
}
ACADEMIC_FACULTY_SOURCES = {
    "Computing": ("Faculty of Computing .xlsx",),
    "Engineering": ("Faculty of Engineering.xlsx",),
    "Medicine & Health Sciences": ("Faculty of Medicine and Health Science .xlsx",),
    "Law": ("Faculty of Law.xlsx",),
    "Social Sciences": ("Faculty of Social Science .xlsx",),
    "Education": ("Faculty of Education.xlsx",),
    "Management Sciences": ("Faculty of Management Science .docx",),
    "Economics": (
        "Faculty of Economics.docx",
        "Economics, Department of Statistics & Planning.docx",
        "Economics, Department of Trade and International Investment.docx",
    ),
}
PROJECT_REPORT_SOURCE_PATTERN = re.compile(
    r"(?:AI_Assistant|Chapters?_4_and_5|ASP document|thesis|project report)",
    re.I,
)
PROJECT_NOISE_PATTERN = re.compile(
    r"\b(?:AI[- ]?based university chatbot|university chatbot|AI chatbot|"
    r"chatbot|knowledge base|answer-generation service|model training|"
    r"thesis|chapter\s+[45]|research project)\b",
    re.I,
)
SEMANTIC_SCOPE_PROTOTYPES = {
    "simad_related": (
        "a question about SIMAD University",
        "a student asking about a university service program policy or leader",
        "information about this university and its administration",
        "a question about studying at SIMAD",
    ),
    "out_of_scope": (
        "a question about world politics or a country president",
        "sports scores entertainment celebrities and unrelated news",
        "cooking recipes weather and unrelated general knowledge",
        "a request unrelated to SIMAD University",
    ),
}
FOLLOW_UP_QUESTION_PROTOTYPES = {
    "follow_up": (
        "tell me more about the previous topic",
        "who are all of them",
        "who are they",
        "continue explaining the previous answer",
        "what about it",
        "give me the rest of them",
    ),
    "new_question": (
        "ask a new self contained question about SIMAD University",
        "change to a different university topic",
    ),
}
FOLLOW_UP_PATTERN = re.compile(
    r"\b(it|its|they|them|that|those|this|these|their)\b", re.I
)
GUIDANCE_PATTERN = re.compile(
    r"\b(advice|advise|career|choose|choosing|recommend|right for me|"
    r"which (?:faculty|program|course|degree)|what should i study)\b",
    re.I,
)
ADMISSION_PATTERN = re.compile(
    r"\b(admission|admissions|apply|application|join|register|registration|"
    r"enroll|enrol|enrollment|enrolment)\b",
    re.I,
)
OUT_OF_SCOPE_PATTERN = re.compile(
    r"\b(quantum entanglement|black holes?|weather(?: forecast)?|world cup|"
    r"football|sports?|cook(?:ing)?|recipes?|write (?:me )?code)\b",
    re.I,
)
UNVERIFIED_POLICY_TOPIC_PATTERN = re.compile(
    r"\b(helicopter|parking)\b",
    re.I,
)
PROGRAM_QUERY_PATTERN = re.compile(r"\b(programs|degrees?|bachelors?)\b", re.I)
COURSE_QUERY_PATTERN = re.compile(
    r"\b(courses?|modules?|subjects?|curriculum)\b|\b[A-Z]{2,8}\s?\d{3,5}\b",
    re.I,
)
ACADEMIC_COMPARISON_PATTERN = re.compile(
    r"\b(?:difference|different|compare|comparison|versus|vs\.?|between)\b",
    re.I,
)
PROGRAM_AVAILABILITY_PATTERN = re.compile(
    r"\b(?:does|do|is|are|have|has|offer|offers|available|provide|provides)\b",
    re.I,
)
SIMAD_SCOPE_PATTERN = re.compile(
    r"\b(simad|university|campus|student|academic|admission|apply|application|"
    r"faculty|faculties|school|department|degree|program|course|curriculum|"
    r"semester|tuition|fees?|scholarship|library|rector|senate|research|"
    r"exchange|accreditation|grading|gpa|policy|services?|club|career|graduate|"
    r"postgraduate|master|bachelor|diploma|certificate)\b",
    re.I,
)
SOMALI_LANGUAGE_REQUEST_PATTERN = re.compile(
    r"\b(?:in|to)\s+somali\b|\bsomali\s+language\b|\baf[- ]?soomaali\b|"
    r"\bluuqad(?:da)?\s+soomaali\b",
    re.I,
)
SOMALI_WORD_PATTERN = re.compile(
    r"\b(?:af|aqbalid|arday|ardayda|barnaamij|barnaamijyada|cashar|deeq|"
    r"deeqaha|fadlan|goorma|iga|ii|imtixaan|jaamacad|jaamacadda|khidmad|"
    r"koorso|koorsooyinka|kulliyad|kulliyada|kulliyadaha|lacag|lacagta|"
    r"maado|maadooyinka|mahadsanid|maxaa|maxay|miyay|miyaa|nidaamka|"
    r"sheeg|sheegi|sharax|sidee|soomaali|waa|xagee)\b",
    re.I,
)
WEB_FOOTER_MARKERS = (
    "Founded and Sponsored by Direct Aid-Kuwait",
    "Copyright (c)",
    "Academic Programs Undergraduate Programs Graduate Programs Quick Links",
    "Home Academics",
    "Home About Us",
)
NAVIGATION_NOISE_MARKERS = (
    "Home Admission",
    "Home About Us",
    "Founded and Sponsored by Direct Aid-Kuwait",
    "Copyright (c)",
    "Get In Touch",
    "Contact Us",
    "Maps & Directions",
    "Jobs Subscribe",
    "Subscribe To Our Newsletter",
    "Connect with us",
    "Academic Programs Undergraduate Programs Graduate Programs Quick Links",
)
SECTION_HEADINGS = (
    "ADMISSION REQUIREMENTS",
    "PROGRAMS OFFERED",
    "OUR SERVICES",
    "SIMAD UNIVERSITY GRADING SYSTEM",
    "FOUNDING FATHERS",
    "FORMER RECTORS",
    "ACCREDITATION",
    "VISION",
    "MISSION",
)
TOPIC_SOURCE_RULES = [
    (r"\btransfer\b", ("Transfer Applications.pdf",)),
    (r"\bscholarships?\b|\bfinancial aid\b", ("Scholarships.pdf",)),
    (
        r"\b(?:accreditation|accredited|ranking|rankings|membership|memberships)\b",
        ("Accreditation, Ranking, & Memberships.pdf",),
    ),
    (r"\b(?:gpa|grading system|grade points?)\b", ("Grading System and GPA.pdf",)),
    (r"\bcampus services?\b", ("CAMPUS SERVICES.pdf",)),
    (r"\blibrar(?:y|ies)\b", ("CAMPUS SERVICES.pdf",)),
    (r"\bclubs?\b", ("CLUBS.pdf",)),
    (r"\bco-?curricular\b", ("Co-Curricular Programs.pdf",)),
    (r"\bcultural week\b", ("CULTURAL WEEK.pdf",)),
    (
        r"\b(?:disability|disabled students?|special needs|dss)\b",
        ("Disability Support Services (DSS).pdf",),
    ),
    (r"\bextracurricular\b", ("Extracurricular Activities.pdf",)),
    (r"\baccommodation\b", ("EXCHANGE PROGRAM.pdf",)),
    (r"\b(?:exchange|international exchange)\b", ("EXCHANGE PROGRAM.pdf",)),
    (r"\b(?:iml|institute of modern languages?|language institute)\b", ("IML.pdf",)),
    (
        r"\b(?:information disclosure|disclosure policy)\b",
        ("INFORMATION DISCOLURE.pdf",),
    ),
    (r"\b(?:innovation lab|i-?lab)\b", ("SIMAD I-LAB.pdf",)),
    (r"\b(?:research|research center)\b", ("SIMAD RESEARCH.pdf",)),
    (r"\bconferences?\b", ("SIMAD CONFERENCES.pdf",)),
    (r"\b(?:master|masters|postgraduate|graduate program)\b", ("SIMAD MASTER PROGRAMS.pdf", "SIMAD - OUM POSTGRADUATE PROGRAM.pdf")),
    (r"\b(?:code of conduct|student conduct|discipline)\b", ("STUDENT CODE OF CONDUCT.pdf",)),
    (r"\b(?:senate|university senate)\b", ("THE SENATE.pdf",)),
    (r"\b(?:vision|mission|core values?)\b", ("VISION.pdf", "SIMAD UNIVERSITY GENERAL INFORMATION.pdf")),
    (r"\bwhy (?:choose )?simad\b", ("WHY SIMAD.pdf",)),
    (r"\bxajsi\b", ("XAJSI.pdf",)),
    (r"\b(?:tuition|semester fee|fee structure)\b", ("TUTION FEES.pdf", "ADMISSION BROCHURE.pdf")),
    (
        r"\b(?:admission requirements?|admissions requirements?|join simad|apply to simad|"
        r"register at simad|registration|enroll(?:ment)?|enrol(?:ment)?)\b",
        ("ADMISSION BROCHURE.pdf",),
    ),
    (
        r"\b(?:history|founded|established|founders?|founding fathers?)\b",
        ("SIMAD HISTORY.pdf",),
    ),
    (
        r"\b(?:former|previous)\s+rectors?\b|\brectors?\b.*\b(?:former|previous)\b",
        ("SIMAD HISTORY.pdf",),
    ),
    (r"\brectors?\b", ("SIMAD HISTORY.pdf", "RECTOR.pdf")),
    (
        r"^(?:give me (?:an? |some )?information about|tell me (?:some |more )?about|"
        r"what is|describe)\s+simad(?: university)?[?!. ]*$",
        ("SIMAD UNIVERSITY GENERAL INFORMATION.pdf",),
    ),
]


def terms(text: str) -> set[str]:
    result = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in STOP_WORDS or len(token) <= 1:
            continue
        result.add(token)
        if token in {"founded", "founder", "founders", "founding"}:
            result.add("found")
        elif token.endswith("ies") and len(token) > 5:
            result.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 4:
            result.add(token[:-1])
    return result


def coverage(query_terms: set[str], text: str) -> float:
    return len(query_terms & terms(text)) / len(query_terms) if query_terms else 0.0


def preferred_response_language(text: str) -> str:
    """Return the user's requested or likely response language."""
    if SOMALI_LANGUAGE_REQUEST_PATTERN.search(text):
        return "Somali"
    hits = SOMALI_WORD_PATTERN.findall(text)
    if len(hits) >= 2:
        return "Somali"
    if len(hits) == 1 and re.search(r"\bsimad\b", text, re.I):
        return "Somali"
    return "English"


def localized_not_found(language: str) -> str:
    return SOMALI_NOT_FOUND_MESSAGE if language == "Somali" else NOT_FOUND_MESSAGE


def localized_out_of_scope(language: str) -> str:
    return SOMALI_OUT_OF_SCOPE_MESSAGE if language == "Somali" else OUT_OF_SCOPE_MESSAGE


def localized_small_talk(language: str) -> str:
    return SOMALI_SMALL_TALK_MESSAGE if language == "Somali" else (
        "I'm glad to help. What would you like to know about SIMAD University?"
    )


def is_not_found_answer(answer: str) -> bool:
    return answer in {NOT_FOUND_MESSAGE, SOMALI_NOT_FOUND_MESSAGE}


def allowed_sources(question: str) -> tuple[str, ...]:
    lowered = normalize_academic_query(question).lower()
    for pattern, sources in TOPIC_SOURCE_RULES:
        if re.search(pattern, lowered):
            return sources
    return ()


def canonical_question(question: str) -> str:
    if GENERAL_SIMAD_PATTERN.fullmatch(" ".join(question.split())):
        return "What is SIMAD University?"
    return question


def is_general_overview_question(question: str) -> bool:
    return canonical_question(question) == "What is SIMAD University?"


def is_project_report_source(source: str) -> bool:
    return bool(PROJECT_REPORT_SOURCE_PATTERN.search(Path(source).name))


def is_project_noise(text: str) -> bool:
    return bool(PROJECT_NOISE_PATTERN.search(text))


def clean_generated_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "").replace("*", "")
    text = text.replace("➢", "-").replace("•", "-")
    text = re.sub(r"(?m)^(\d+[.)])[ \t]*\r?\n[ \t]*(?=\S)", r"\1 ", text)
    text = re.sub(r"\s*\[(?:Source\s*)?\d+[^\]]*\]\s*", "", text, flags=re.I)
    text = re.sub(r"\s*Sources?\s*:.*$", "", text, flags=re.I | re.MULTILINE)
    text = re.sub(
        r"\b(?:the )?(?:context provided|provided context|SIMAD knowledge base context|"
        r"knowledge base context)\b",
        "available information",
        text,
        flags=re.I,
    )
    text = re.sub(r"\bis mentioned in the available information\b", "is", text, flags=re.I)
    text = re.sub(
        r"(?:the )?(?:available information|reference text) does not contain "
        r"(?:information|enough information) about (?:that|this|[^.]+)\.?",
        NOT_FOUND_MESSAGE,
        text,
        flags=re.I,
    )
    text = re.sub(
        r"I (?:do not|don't) have enough verified SIMAD(?: University)? information"
        r"(?: to answer that)?\.?",
        NOT_FOUND_MESSAGE,
        text,
        flags=re.I,
    )
    text = re.sub(
        r"I could not find verified information(?: about [^.]+)? in the available SIMAD "
        r"(?:data|documents)\.?",
        NOT_FOUND_MESSAGE,
        text,
        flags=re.I,
    )
    text = re.sub(r"\bSIMAD knowledge base\b", "SIMAD University information", text, flags=re.I)
    text = re.sub(r"\bThe document only lists\b", "SIMAD records list", text, flags=re.I)
    text = re.sub(r"\b(?:the )?reference text\b", "SIMAD records", text, flags=re.I)
    text = re.sub(r"\s+%", "%", text)
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip()).strip()


def sanitize_user_input(value: object) -> str:
    """Normalize browser/API input before routing it through the chatbot."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\x00", " ")
    return " ".join(text.split()).strip()


def rule_based_fallback(question: str) -> str:
    """Last-resort response used if the full RAG service raises inside Django."""
    question = sanitize_user_input(question)
    if not question:
        return NOT_FOUND_MESSAGE

    language = preferred_response_language(question)
    lowered = question.lower()
    if re.fullmatch(r"(hi|hello|hey|salaam|asc|assalamu alaikum)[!. ]*", lowered):
        return localized_small_talk(language)
    if OUT_OF_SCOPE_PATTERN.search(question) or (
        not SIMAD_SCOPE_PATTERN.search(question)
        and not SOMALI_WORD_PATTERN.search(question)
    ):
        return localized_out_of_scope(language)
    return localized_not_found(language)


def clean_previous_answer(text: str) -> str:
    """Prepare a stored factual answer for repeating or transforming."""
    cleaned = clean_generated_answer(text)
    lowered = cleaned.lower()
    marker_positions = [
        lowered.find(marker.lower())
        for marker in WEB_FOOTER_MARKERS
        if lowered.find(marker.lower()) >= 0
    ]
    if marker_positions:
        cleaned = cleaned[: min(marker_positions)]
    cleaned = re.sub(
        r"(?m)^[ \t]*(\d+[.)])[ \t]*\r?\n[ \t]*(?=\S)", r"\1 ", cleaned
    )
    cleaned = re.sub(r"(?m)^\s*[-•]\s*\n\s*(?=\S)", "- ", cleaned)
    return focused_answer("", cleaned).strip()


def split_sentences(text: str) -> list[str]:
    protected = text
    abbreviations = {
        "Dr.": "Dr<period>",
        "Mr.": "Mr<period>",
        "Ms.": "Ms<period>",
        "H.E.": "H<period>E<period>",
        "e.g.": "e<period>g<period>",
    }
    for abbreviation, replacement in abbreviations.items():
        protected = protected.replace(abbreviation, replacement)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)
    return [
        sentence.replace("<period>", ".").strip()
        for sentence in sentences
        if sentence.strip()
    ]


def focused_answer(question: str, answer: str) -> str:
    """Remove repeated and generic overview text from a specific answer."""
    answer = clean_generated_answer(answer)
    if not answer:
        return answer

    is_general = is_general_overview_question(question)
    history_topic = bool(
        re.search(r"\b(history|found|founded|founder|established|when)\b", question, re.I)
    )
    generic_overview = re.compile(
        r"^(?:SIMAD University is a private higher education institution|"
        r"SIMAD University provides undergraduate and postgraduate programs|"
        r"SIMAD University was established in 1999)",
        re.I,
    )

    output = []
    seen = set()
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        if is_general and (line.startswith("- ") or re.match(r"^\d+[.)]\s+\S", line)):
            continue
        pieces = (
            [line]
            if line.startswith("- ") or re.match(r"^\d+[.)]\s+\S", line)
            else split_sentences(line)
        )
        for piece in pieces:
            piece = piece.strip()
            if is_project_noise(piece):
                continue
            if is_general:
                if re.search(
                    r"\b(faculties|faculty|schools?|programs?|courses?|subjects?|"
                    r"services?|admissions?|tuition|scholarships?)\b",
                    piece,
                    re.I,
                ) and not re.search(
                    r"\b(private higher education|located|mogadishu|established|"
                    r"quality education|research|community service)\b",
                    piece,
                    re.I,
                ):
                    continue
                if re.search(r"\b(programs?|faculties|faculty|courses?)\b", piece, re.I):
                    continue
            key = re.sub(r"\W+", " ", piece.lower()).strip()
            if not key or key in seen:
                continue
            if not is_general and not history_topic and generic_overview.search(piece):
                continue
            seen.add(key)
            output.append(piece)
            if is_general and len(output) >= 2:
                break
        if is_general and len(output) >= 2:
            break
    return "\n".join(output).strip()


def shorten_previous_answer(answer: str, max_lines: int = 3) -> str:
    """Create a concise extractive summary without adding new facts."""
    cleaned = clean_previous_answer(answer)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    candidates = []
    for line in lines:
        if line.startswith("- ") or re.match(r"^\d+[.)]\s+\S", line):
            candidates.append(line)
        else:
            candidates.extend(split_sentences(line))

    selected = []
    seen = set()
    for candidate in candidates:
        key = re.sub(r"\W+", " ", candidate.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) >= max_lines:
            break
    return "\n".join(selected) if selected else cleaned


def _simplify_answer_item(text: str) -> str:
    """Rewrite one verified answer item more directly without adding facts."""
    text = " ".join(text.split()).strip()
    text = re.sub(
        r"^(?:according to (?:the )?(?:available )?information,?\s*|"
        r"the (?:available|verified) information (?:states|says) that\s*|"
        r"it is important to note that\s*)",
        "",
        text,
        flags=re.I,
    )
    replacements = (
        (r"\bin order to\b", "to"),
        (r"\bdue to the fact that\b", "because"),
        (r"\bfor the purpose of\b", "for"),
        (r"\bwith regard to\b", "about"),
        (r"\bis able to\b", "can"),
        (r"\bare able to\b", "can"),
        (r"\bis required to\b", "must"),
        (r"\bare required to\b", "must"),
        (r"\bshould successfully pass\b", "should pass"),
        (r"\bsuccessfully pass\b", "pass"),
        (r"\bis located in\b", "is in"),
        (r"\bprovides\b", "offers"),
        (r"\boffers student exchange programs\b", "gives students exchange opportunities"),
        (r"\ballows? students to\b", "gives students the opportunity to"),
        (r"\bhelps? students to\b", "lets students"),
        (r"\bparticipate in\b", "join"),
        (r"\ba minimum overall average of\s+(\d+)\s*%", r"at least \1%"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)

    text = re.sub(r"^(?:Applicants?|Students?|Candidates?) must\s+", "", text, flags=re.I)
    text = re.sub(r"^(?:Applicants?|Students?|Candidates?) should\s+", "", text, flags=re.I)
    text = re.sub(r"^Should\s+", "", text, flags=re.I)
    text = re.sub(r"^Must\s+", "", text, flags=re.I)
    text = re.sub(r"^SIMAD University was established\b", "Established", text, flags=re.I)
    text = re.sub(r"^SIMAD University is in\b", "Located in", text, flags=re.I)
    text = re.sub(r"^SIMAD University offers\b", "Offers", text, flags=re.I)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    if text:
        text = text[0].upper() + text[1:]
    return text.strip()


def simplify_previous_answer(answer: str, max_lines: int = 4) -> str:
    """Rewrite the previous verified answer into shorter, cleaner key points."""
    cleaned = clean_previous_answer(answer)
    candidates = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        marker = re.match(r"^(?:-\s+|\d+[.)]\s+)(.+)$", line)
        if marker:
            candidates.append((False, marker.group(1)))
            continue
        if line.endswith(":"):
            candidates.append((True, line))
            continue
        candidates.extend((False, sentence) for sentence in split_sentences(line))

    output = []
    seen = set()
    for is_heading, candidate in candidates:
        simplified = _simplify_answer_item(candidate)
        if not simplified:
            continue
        if is_heading:
            simplified = re.sub(
                r"^The (?:undergraduate |postgraduate )?admission requirements are:$",
                "Admission requirements:",
                simplified,
                flags=re.I,
            )
            rendered = simplified
        else:
            rendered = f"- {simplified}"
        key = re.sub(r"\W+", " ", simplified.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(rendered)
        if len(output) >= max_lines:
            break

    transformed = "\n".join(output).strip()
    if transformed and transformed != cleaned:
        return transformed
    return shorten_previous_answer(cleaned, max(1, max_lines - 1))


def rewrite_previous_answer_locally(
    answer: str, action: str, target_lines: int | None = None
) -> str:
    """Grounded fallback when the language-model rewrite is unavailable."""
    if target_lines:
        return simplify_previous_answer(answer, target_lines)
    if action == "summarize":
        return simplify_previous_answer(answer, 3)
    if action == "expand":
        return simplify_previous_answer(answer, 6)
    if action == "clean":
        return simplify_previous_answer(answer, 5)

    rewritten = simplify_previous_answer(answer, 4)
    facts = []
    for line in rewritten.splitlines():
        line = line.strip()
        if not line or line.endswith(":"):
            continue
        fact = re.sub(r"^(?:-\s+|\d+[.)]\s+)", "", line).strip()
        if fact and not fact.endswith((".", "!", "?")):
            fact += "."
        if fact:
            facts.append(fact)
    return " ".join(facts) if facts else rewritten


def clean_evidence_text(text: str) -> str:
    """Remove website furniture while preserving document facts verbatim."""
    text = unicodedata.normalize("NFKC", text).replace(" / / ", " ")
    qa_pattern = r"\bQ:\s*(.+?)\s+A:\s*(.*?)(?=\s+Q:|\Z)"
    text = re.sub(
        qa_pattern,
        lambda match: " "
        if is_project_noise(f"{match.group(1)} {match.group(2)}")
        else match.group(0),
        text,
        flags=re.I | re.S,
    )
    marker_positions = []
    lowered_text = text.lower()
    for marker in (*WEB_FOOTER_MARKERS, *NAVIGATION_NOISE_MARKERS):
        position = lowered_text.find(marker.lower())
        if position >= 0:
            marker_positions.append(position)
    if marker_positions:
        text = text[: min(marker_positions)]
    text = re.sub(r"\bThe Senate\s*/\s*/\s*The Senate List Profiles\b", "The Senate", text, flags=re.I)
    text = re.sub(r"\bThe Senate\s+The Senate List Profiles\b", "The Senate", text, flags=re.I)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\bwww\.\S+", "", text)
    text = re.sub(r"\b\d{2}/\d{2}/\d{4},\s*\d{2}:\d{2}\b", "", text)
    sentence_parts = []
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text):
        if is_project_noise(sentence):
            continue
        sentence_parts.append(sentence)
    return " ".join(" ".join(sentence_parts).split()).strip(" /|-")


def evidence_passages(text: str) -> list[str]:
    """Create answerable passages from prose, Q/A records, and extracted tables."""
    cleaned = clean_evidence_text(text)
    if not cleaned:
        return []

    passages = []
    qa_pattern = r"\bQ:\s*(.+?)\s+A:\s*(.*?)(?=\s+Q:|\Z)"
    for _, answer in re.findall(
        qa_pattern,
        cleaned,
        flags=re.I | re.S,
    ):
        answer = clean_generated_answer(answer)
        if answer and not is_project_noise(answer):
            passages.append(answer)

    prose = re.sub(qa_pattern, " ", cleaned, flags=re.I | re.S)
    for heading in SECTION_HEADINGS:
        position = prose.upper().find(heading)
        if position >= 0:
            passages.append(prose[position : position + 700])

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", prose)
    passages.extend(sentence for sentence in sentences if 35 <= len(sentence) <= 700)

    words = prose.split()
    if len(words) <= 115:
        passages.append(prose)
    else:
        for start in range(0, len(words), 70):
            window = " ".join(words[start : start + 95])
            if len(window) >= 80:
                passages.append(window)
            if start + 95 >= len(words):
                break

    unique = []
    seen = set()
    for passage in passages:
        passage = clean_generated_answer(passage).strip()
        if is_project_noise(passage):
            continue
        key = re.sub(r"\W+", " ", passage.lower()).strip()
        if not key or key in seen or passage.count("|") > 8:
            continue
        seen.add(key)
        unique.append(passage)
    return unique


@dataclass
class SearchResult:
    text: str
    source: str
    location: str
    distance: float

    @property
    def citation(self) -> str:
        return f"{self.source}, {self.location}"


class SimadChatbot:
    def __init__(self) -> None:
        # The training step downloads the model; chatting should work offline afterward.
        self.model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        self.client = chromadb.PersistentClient(path=str(DB_DIR))
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.hf_model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.hf_provider = os.getenv("HF_PROVIDER", "auto")
        hf_token = os.getenv("HF_TOKEN", "").strip()
        self.generator = (
            InferenceClient(
                model=self.hf_model,
                provider=self.hf_provider,
                token=hf_token,
                timeout=float(os.getenv("HF_TIMEOUT_SECONDS", "90")),
            )
            if hf_token
            else None
        )
        self.last_answer_mode = "ready"
        self.last_message_intent = "ready"
        self.last_semantic_topic = "unknown"
        self.last_conversation_intent = ""

    def search(self, question: str, limit: int = 8) -> list[SearchResult]:
        normalized_question = canonical_question(normalize_academic_query(question))
        if OUT_OF_SCOPE_PATTERN.search(normalized_question) and not re.search(
            r"\bsimad\b", normalized_question, re.I
        ):
            return []
        if UNVERIFIED_POLICY_TOPIC_PATTERN.search(normalized_question):
            return []
        semantic_topic = self._semantic_topic(normalized_question)
        retrieval_question = self._semantic_retrieval_question(
            normalized_question, semantic_topic
        )
        embedding = self.model.encode(
            [retrieval_question], normalize_embeddings=True
        ).tolist()[0]
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(max(limit * 4, 20), self.collection.count()),
        )

        query_terms = terms(normalized_question)
        source_restrictions = self._topic_sources(normalized_question, semantic_topic)
        candidate_rows = list(
            zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        )
        if source_restrictions:
            authoritative = self.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            for text, metadata, stored_embedding in zip(
                authoritative["documents"],
                authoritative["metadatas"],
                authoritative["embeddings"],
            ):
                if Path(str(metadata["source"])).name not in source_restrictions:
                    continue
                similarity = sum(
                    float(left) * float(right)
                    for left, right in zip(embedding, stored_embedding)
                )
                candidate_rows.append((text, metadata, 1.0 - similarity))

        ranked_matches = []
        seen_matches = set()
        for text, metadata, distance in candidate_rows:
            source = str(metadata["source"])
            location = str(metadata.get("location", "document"))
            source_name = Path(source).name
            if is_project_report_source(source_name):
                continue
            match_key = (source, location, text)
            if match_key in seen_matches:
                continue
            seen_matches.add(match_key)
            if source_restrictions and source_name not in source_restrictions:
                continue

            text_coverage = coverage(query_terms, text)
            label_coverage = coverage(query_terms, f"{source} {location}")
            exact_single_term = len(query_terms) == 1 and text_coverage == 1.0
            topic_source_match = bool(source_restrictions)
            has_lexical_support = text_coverage >= 0.20 or label_coverage > 0

            # Topic-specific questions can only use their authoritative documents.
            # Other questions need lexical evidence or exceptionally strong semantics.
            relevant = (
                (topic_source_match and distance <= MAX_DISTANCE + 0.15)
                or exact_single_term
                or (label_coverage > 0 and distance <= MAX_DISTANCE + 0.10)
                or (distance <= MAX_DISTANCE and has_lexical_support)
                or (distance <= STRICT_SEMANTIC_DISTANCE and text_coverage >= 0.20)
            )
            if relevant:
                match = (
                    SearchResult(
                        text=text,
                        source=source,
                        location=location,
                        distance=float(distance),
                    )
                )
                hybrid_score = float(distance) - (0.10 * text_coverage) - (0.35 * label_coverage)
                ranked_matches.append((hybrid_score, match))

        ranked_matches.sort(key=lambda item: item[0])
        return [match for _, match in ranked_matches[:limit]]

    def answer(self, question: str, history: list[dict[str, str]]) -> str:
        response_language = preferred_response_language(question)
        message_intent = self._classify_message_intent(question, history)
        self.last_message_intent = message_intent
        self.last_conversation_intent = (
            message_intent if message_intent in CONVERSATION_MESSAGE_INTENTS else ""
        )
        if message_intent == "out_of_scope":
            self.last_answer_mode = "conversation"
            return localized_out_of_scope(response_language)
        conversational_answer = (
            self._conversation_answer(question, history)
            if message_intent in CONVERSATION_MESSAGE_INTENTS | {"follow_up_answer"}
            else None
        )
        if conversational_answer:
            self.last_answer_mode = "conversation"
            return conversational_answer
        if message_intent in CONVERSATION_MESSAGE_INTENTS:
            self.last_answer_mode = "conversation"
            return localized_small_talk(response_language)

        interpreted_input = canonical_question(normalize_academic_query(question))
        intent = self._classify_intent(interpreted_input, history)
        if intent == "out_of_scope":
            self.last_answer_mode = "conversation"
            return localized_out_of_scope(response_language)
        if message_intent == "follow_up_question":
            intent = "follow_up"

        contextual_question = (
            self._contextual_question(interpreted_input, history)
            if intent == "follow_up"
            else interpreted_input
        )
        interpreted_question = normalize_academic_query(contextual_question)
        self.last_semantic_topic = self._semantic_topic(interpreted_question)
        resolved_intent = self._classify_intent(interpreted_question, [])
        database_context = self._database_context(interpreted_question, resolved_intent)
        if database_context and self._is_academic_database_context(database_context):
            self.last_semantic_topic = "academics"
        if database_context.startswith("Verified SIMAD administration records:"):
            self.last_answer_mode = "local_grounded"
            return focused_answer(
                interpreted_question,
                self._answer_from_database_context(interpreted_question, database_context),
            )
        if (
            database_context
            and not self.generator
            and self._database_context_is_self_sufficient(
                interpreted_question, resolved_intent, database_context
            )
        ):
            self.last_answer_mode = "local_grounded"
            return focused_answer(
                interpreted_question,
                self._answer_from_database_context(interpreted_question, database_context),
            )

        search_limit = 12 if self.last_semantic_topic in {"leadership", "governance"} else 8
        matches = self.search(interpreted_question, search_limit)
        if database_context and self._is_academic_database_context(database_context):
            self.last_semantic_topic = "academics"
            matches = [
                match
                for match in matches
                if not Path(match.source).name
                in {"THE SENATE.pdf", "RECTOR.pdf", "SIMAD HISTORY.pdf"}
            ]
        if self.last_semantic_topic in {"leadership", "governance"} and matches:
            preferred = (
                ("RECTOR.pdf", "THE SENATE.pdf", "SIMAD HISTORY.pdf")
                if re.search(r"\bcurrent rector\b|\bwho is (?:the )?rector\b", interpreted_question, re.I)
                else ("THE SENATE.pdf", "RECTOR.pdf", "SIMAD HISTORY.pdf")
            )
            source_order = {name: index for index, name in enumerate(preferred)}
            matches = sorted(
                matches,
                key=lambda match: (
                    min(
                        (
                            order
                            for source, order in source_order.items()
                            if match.source.endswith(source)
                        ),
                        default=len(source_order),
                    ),
                    match.distance,
                ),
            )
        if (
            database_context
            and re.search(r"\bfacult(?:y|ies)\b|\bschool\b", interpreted_question, re.I)
            and not COURSE_QUERY_PATTERN.search(interpreted_question)
        ):
            matches = [
                match
                for match in matches
                if Path(match.source).suffix.lower() not in {".xlsx", ".docx"}
            ]
        if not matches and not database_context:
            if self._is_likely_simad_scope(interpreted_question):
                self.last_answer_mode = "not_found"
                return specific_not_found(interpreted_question, response_language)
            # Even if not scope-matched, if it contains a known SIMAD topic keyword, say "not found"
            if specific_not_found(interpreted_question, response_language) != NOT_FOUND_MESSAGE:
                self.last_answer_mode = "not_found"
                return specific_not_found(interpreted_question, response_language)
            self.last_answer_mode = "conversation"
            return localized_out_of_scope(response_language)

        if self.last_semantic_topic in {"leadership", "governance"} and response_language == "English":
            self.last_answer_mode = "local_grounded"
            return focused_answer(
                interpreted_question,
                self._local_grounded_answer(
                    interpreted_question,
                    matches,
                    database_context=database_context,
                    intent=resolved_intent,
                ),
            )

        if not self.generator:
            self.last_answer_mode = "local_grounded"
            return focused_answer(
                interpreted_question,
                self._local_grounded_answer(
                    interpreted_question,
                    matches,
                    database_context=database_context,
                    intent=resolved_intent,
                ),
            )

        context_parts = []
        if database_context:
            context_parts.append(database_context)
        if is_general_overview_question(interpreted_question):
            overview_context = self._overview_answer_from_matches(matches)
            if overview_context:
                context_parts.append(overview_context)
            else:
                context_parts.extend(clean_evidence_text(match.text) for match in matches[:1])
        else:
            context_parts.extend(clean_evidence_text(match.text) for match in matches[:4])
        context = "\n\n".join(context_parts)
        if response_language == "Somali":
            language_instruction = (
                "Answer language: Somali. Write the final answer in natural Somali. "
                "Keep official SIMAD names, faculty names, program names, course codes, "
                "fees, dates, and academic terms unchanged when translating them could "
                f"change their meaning. If the reference text does not contain the answer, say only: "
                f"'{SOMALI_NOT_FOUND_MESSAGE}' "
            )
        else:
            language_instruction = (
                "Answer language: English. If the reference text does not contain the answer, say only: "
                f"'{NOT_FOUND_MESSAGE}' "
            )
        memory_items = history[-MAX_MODEL_HISTORY:] if history else []
        memory_text = "\n".join(
            f"{item.get('role', 'unknown')}: {clean_generated_answer(item.get('content', ''))}"
            for item in memory_items
            if item.get("content")
        )
        messages = [
            {
                "role": "system",
                "content": (
                    language_instruction
                    +
                    "You are the SIMAD University student assistant. Use conversation history "
                    "to understand follow-ups, remember the current discussion, and respond "
                    "consistently. For every factual claim about SIMAD, use only the NEW verified "
                    "reference text supplied with the current question. Never treat an earlier "
                    "assistant answer as factual evidence. Directly answer what the user asks "
                    "without unrelated background. Be helpful, natural, focused, and concise. "
                    "For a general SIMAD overview, answer in one or two short sentences only. "
                    "Do not list faculties, programs, courses, services, tuition, or chatbot "
                    "project details unless the user explicitly asks for them. "
                    "Use plain text only: do not use markdown headings, hashtags, asterisks, "
                    "bold formatting, citations, sources, page numbers, or document names. "
                    "Never say 'context provided', 'knowledge base context', 'mentioned in the "
                    "context', 'reference text', or similar internal phrases. Do not add facts "
                    "that are not in the verified reference. Do not infer, embellish, or add "
                    "descriptions, benefits, subject areas, or purposes. If the reference only "
                    "contains a faculty name and program list, state only those facts. For "
                    "course-list questions, return course "
                    "names only unless the user specifically requests codes, credit hours, or "
                    "a full curriculum table. For comparison questions, compare only the "
                    "faculties, programs, and academic areas explicitly named in the verified "
                    "reference. You may explain each side using its listed program names, "
                    "but do not add career outcomes, rankings, or unsupported descriptions. "
                    "Never invent facts, fees, dates, "
                    "requirements, courses, policies, or historical counts."
                ),
            },
        ]
        if memory_text:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Conversation memory for interpreting follow-ups only. "
                        "Never treat an earlier assistant answer as factual evidence.\n"
                        f"{memory_text}"
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Current question: {question}\n"
                    + (
                        f"Interpreted question: {interpreted_question}\n"
                        if interpreted_question.lower() != question.lower()
                        else ""
                    )
                    +
                    "\n"
                    f"SIMAD UNIVERSITY REFERENCE MATERIAL:\n{context}"
                ),
            }
        )
        try:
            response = self.generator.chat_completion(
                model=self.hf_model,
                messages=messages,
                max_tokens=int(os.getenv("HF_MAX_TOKENS", "500")),
                temperature=0.0,
            )
            answer = clean_generated_answer(
                response.choices[0].message.content or "No answer was generated."
            )
            if not self._generated_answer_is_grounded(answer, context, response_language):
                self.last_answer_mode = "local_grounded"
                return focused_answer(
                    interpreted_question,
                    self._local_grounded_answer(
                        interpreted_question,
                        matches,
                        database_context=database_context,
                        intent=resolved_intent,
                    ),
                )
            self.last_answer_mode = "huggingface"
            return focused_answer(interpreted_question, answer)
        except HfHubHTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            error_text = str(exc).lower()
            if status_code == 403 and "inference providers" in error_text:
                self.generator = None
            elif status_code == 401 or "unauthorized" in error_text:
                self.generator = None
            self.last_answer_mode = "local_grounded"
            return focused_answer(
                interpreted_question,
                self._local_grounded_answer(
                    interpreted_question,
                    matches,
                    database_context=database_context,
                    intent=resolved_intent,
                ),
            )
        except Exception:
            self.last_answer_mode = "local_grounded"
            return focused_answer(
                interpreted_question,
                self._local_grounded_answer(
                    interpreted_question,
                    matches,
                    database_context=database_context,
                    intent=resolved_intent,
                ),
            )

    def _is_likely_simad_scope(self, question: str) -> bool:
        if OUT_OF_SCOPE_PATTERN.search(question) and not re.search(r"\bsimad\b", question, re.I):
            return False
        return bool(
            SIMAD_SCOPE_PATTERN.search(question)
            or allowed_sources(question)
            or find_programs(question)
            or find_program_parent(question)
            or course_program_name(question)
            or self._semantic_scope(question) == "simad_related"
        )

    @staticmethod
    def _generated_answer_is_grounded(
        answer: str, context: str, response_language: str = "English"
    ) -> bool:
        if not answer or is_not_found_answer(answer):
            return bool(answer)
        def _norm_num(n: str) -> str:
            # Strip currency/percentage symbols and thousands separators so that
            # "$1,100" in context matches "1100" in the answer, and vice-versa.
            return re.sub(r"[,$%]", "", n).lstrip("0") or "0"

        context_numbers = {_norm_num(n) for n in re.findall(r"\b\d[\d,.%$-]*\b", context)}
        answer_numbers = {_norm_num(n) for n in re.findall(r"\b\d[\d,.%$-]*\b", answer)}
        if not answer_numbers <= context_numbers:
            return False
        if response_language == "Somali":
            return True

        answer_terms = terms(answer)
        if not answer_terms:
            return True
        context_terms = terms(context)
        if not context_terms:
            return False
        overlap = len(answer_terms & context_terms) / max(1, len(answer_terms))
        return overlap >= 0.35

    @staticmethod
    def _rewrite_is_grounded(answer: str, source: str) -> bool:
        """Reject unchanged rewrites and content unsupported by the stored answer."""
        if not answer or not source:
            return False
        normalized_answer = re.sub(r"\W+", " ", answer.lower()).strip()
        normalized_source = re.sub(r"\W+", " ", source.lower()).strip()
        if not normalized_answer or normalized_answer == normalized_source:
            return False
        if SequenceMatcher(None, normalized_answer, normalized_source).ratio() >= 0.96:
            return False
        if not SimadChatbot._generated_answer_is_grounded(answer, source):
            return False

        answer_terms = terms(answer)
        source_terms = terms(source)
        if answer_terms and len(answer_terms & source_terms) / len(answer_terms) < 0.35:
            return False

        precise_terms = {
            "bachelor",
            "certificate",
            "college",
            "course",
            "credit",
            "degree",
            "department",
            "diploma",
            "exam",
            "faculty",
            "fee",
            "founder",
            "interview",
            "master",
            "program",
            "rector",
            "scholarship",
            "semester",
            "tuition",
        }
        for precise_term in precise_terms:
            pattern = rf"\b{re.escape(precise_term)}(?:s|es)?\b"
            if re.search(pattern, answer, re.I) and not re.search(pattern, source, re.I):
                return False

        def normalized_units(text: str) -> list[str]:
            units = []
            for line in text.splitlines():
                line = re.sub(r"^(?:-\s+|\d+[.)]\s+)", "", line.strip())
                for sentence in split_sentences(line):
                    normalized = re.sub(r"\W+", " ", sentence.lower()).strip()
                    if normalized:
                        units.append(normalized)
            return units

        source_units = set(normalized_units(source))
        answer_units = normalized_units(answer)
        if answer_units and all(unit in source_units for unit in answer_units):
            return False
        return True

    def _rewrite_previous_answer(
        self,
        source_answer: str,
        action: str,
        _user_request: str,
        target_lines: int | None = None,
    ) -> str:
        """Rewrite stored factual text without running document retrieval."""
        source = clean_previous_answer(source_answer)
        fallback = rewrite_previous_answer_locally(source, action, target_lines)
        if not getattr(self, "generator", None):
            return fallback

        instructions = {
            "clean": (
                "Rewrite it in new words so it is clean, organized, and easy to scan. "
                "Remove repetition and awkward wording."
            ),
            "simplify": (
                "Rewrite it in new, easier words for a student. Replace academic or long "
                "wording with plain language."
            ),
            "clarify": (
                "Explain it again in new, student-friendly words. Make the meaning clearer "
                "without repeating the original sentences."
            ),
            "summarize": (
                "Rewrite it as a concise summary in 2 to 3 short lines using new wording."
            ),
            "expand": (
                "Rewrite it with slightly more explanation by using the useful details already "
                "present in the source. Do not introduce any new detail."
            ),
        }
        task = instructions.get(action, instructions["simplify"])
        if target_lines:
            task += f" Return exactly {target_lines} short lines."

        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite only the verified source answer. Use genuinely different wording, "
                    "not merely different formatting. Keep every fact faithful to the source. "
                    "Keep official names and precise academic terms unchanged. "
                    "Do not add examples, benefits, explanations, names, numbers, or facts that "
                    "are absent from the source. Do not mention the source, context, or these "
                    "instructions. Return only the rewritten answer in plain text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Transformation: {action}\n"
                    f"Task: {task}\n\nVerified source answer:\n{source}"
                ),
            },
        ]
        try:
            response = self.generator.chat_completion(
                model=self.hf_model,
                messages=messages,
                max_tokens=int(os.getenv("HF_MAX_TOKENS", "500")),
                temperature=0.1,
            )
            rewritten = clean_generated_answer(
                response.choices[0].message.content or ""
            )
            if action == "summarize":
                lines = [line.strip() for line in rewritten.splitlines() if line.strip()]
                if len(lines) == 1:
                    lines = split_sentences(lines[0])
                limit = target_lines or 3
                rewritten = "\n".join(lines[:limit])
                if target_lines and len(lines[:limit]) != target_lines:
                    return fallback
                if not target_lines and len(lines[:limit]) < 2 and len(source.splitlines()) > 1:
                    return fallback
            if action == "expand" and len(terms(rewritten)) < len(terms(fallback)) * 0.75:
                return fallback
            if self._rewrite_is_grounded(rewritten, source):
                return rewritten
        except HfHubHTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in {401, 403}:
                self.generator = None
        except Exception:
            pass
        return fallback

    def _conversation_answer(self, question: str, history: list[dict[str, str]]) -> str | None:
        lowered = " ".join(question.lower().split())
        previous_question = next(
            (item["content"] for item in reversed(history) if item.get("role") == "user"),
            "",
        )
        previous_answer = next(
            (item["content"] for item in reversed(history) if item.get("role") == "assistant"),
            "",
        )
        topic_question = self._previous_topic_question(history)
        topic_answer = self._previous_topic_answer(history) or previous_answer

        conversation_intent = self._classify_conversation_intent(question, history)
        if conversation_intent:
            response_language = preferred_response_language(question)
            if conversation_intent == "memory":
                recent_questions = [
                    item["content"]
                    for item in history
                    if item.get("role") == "user"
                ]
                if re.search(r"\b(last|previous|question|asked|topic|discuss)\b", lowered):
                    if topic_question:
                        return f'That was about your question: "{clean_generated_answer(topic_question)}"'
                    return "There is no earlier SIMAD question in this chat yet."
                if recent_questions:
                    displayed = recent_questions[-10:]
                    return (
                        f"Yes. I remember {len(recent_questions)} questions in this chat. "
                        "Your most recent questions were:\n- " + "\n- ".join(displayed)
                    )
                return "There is no earlier conversation in this chat yet."

            if conversation_intent == "user_identity":
                name_match = re.search(
                    r"\b(?:my name is|call me|i am)\s+([A-Za-z][A-Za-z .'-]{0,40})",
                    question,
                    re.I,
                )
                if name_match:
                    return f"Nice to meet you, {name_match.group(1).strip()}. How can I help?"
                for item in reversed(history):
                    if item.get("role") != "user":
                        continue
                    name_match = re.search(
                        r"\b(?:my name is|call me|i am)\s+([A-Za-z][A-Za-z .'-]{0,40})",
                        item.get("content", ""),
                        re.I,
                    )
                    if name_match:
                        return f"You told me your name is {name_match.group(1).strip()}."
                return (
                    "I do not know your identity unless you tell me, but I can remember "
                    "what you share during this chat."
                )

            return self._conversation_reply(conversation_intent, response_language)

        follow_up_action = self._follow_up_action(question, has_previous=bool(topic_answer))
        if (
            self._is_factual_continuation(question)
            and self._semantic_follow_up(question, history)
        ):
            follow_up_action = None

        if follow_up_action:
            if not topic_answer:
                return "There is no previous answer in this chat yet."
            if follow_up_action == "topic":
                if topic_question:
                    return f'That answer was for your question: "{topic_question}"'
                return "There is no earlier question in this chat yet."

            if follow_up_action == "select_one":
                items = []

                # Strategy 1: bullet / numbered list lines  (- item, * item, 1. item)
                for line in topic_answer.splitlines():
                    line = line.strip()
                    bullet_match = re.match(r"^(?:-\s*|\*\s*|\d+[.)]\s*)(.+)$", line)
                    if bullet_match:
                        item_text = bullet_match.group(1).strip()
                        skip_prefixes = (
                            "undergraduate program", "verified", "simad records",
                            "the following", "these are",
                        )
                        if item_text and not item_text.lower().startswith(skip_prefixes):
                            items.append(item_text)

                # Strategy 2: sentence-embedded comma list
                # e.g. "The former rectors are Hassan, Farah and Ali."
                if not items:
                    intro_re = re.compile(
                        r"(?:the\s+\w+(?:\s+\w+){0,4}\s+(?:are|were|include|is|was)\s*:?\s*"
                        r"|(?:they|these)\s+(?:are|include|were)\s*:?\s*"
                        r"|(?:include|includes|including)\s*:?\s*"
                        r"|:\s*)",
                        re.I,
                    )
                    for line in topic_answer.splitlines():
                        line = line.strip()
                        if "," not in line:
                            continue
                        intro_match = intro_re.search(line)
                        if intro_match:
                            line = line[intro_match.end():]
                        line = re.sub(r"[.!?]+$", "", line).strip()
                        if not line:
                            continue
                        for part in re.split(r",\s*", line):
                            part = re.sub(r"^(?:and|or|also)\s+", "", part.strip(), flags=re.I).strip()
                            if len(part) >= 2 and not re.search(
                                r"\b(?:the|this|that|they|these|those|it|is|are|was|were|have|has)\b",
                                part, re.I
                            ):
                                items.append(part)

                # Strategy 3: whole-line fallback
                if not items:
                    for line in topic_answer.splitlines():
                        line = line.strip()
                        if line and len(line) >= 3:
                            items.append(line)
                            break

                if items:
                    first_item = items[0]
                    if "rector" in lowered:
                        return f"Based on SIMAD's records, one former rector is: {first_item}"
                    elif "faculty" in lowered or "faculties" in lowered:
                        return f"One of the faculties is: {first_item}"
                    elif "program" in lowered:
                        return f"One of the programs is: {first_item}"
                    elif "founder" in lowered or "father" in lowered:
                        return f"One of the founding fathers is: {first_item}"
                    else:
                        return f"One option is: {first_item}"
                return "I couldn't extract individual options from the previous list, but you can choose any item mentioned above."

            if follow_up_action == "confirm_only":
                if "faculty" in topic_question.lower() or "faculties" in topic_question.lower():
                    return "Yes, those are the only verified faculties at SIMAD University."
                if "program" in topic_question.lower():
                    return "Yes, those are the only verified undergraduate programs offered in that area at SIMAD University."
                if "rector" in topic_question.lower():
                    return "Yes, those are the only verified former rectors mentioned in SIMAD's history records."
                return "Yes, those are the only options listed in SIMAD's verified records."

            line_match = re.search(r"\bmake (?:this|it|that) (\d+) lines?\b", lowered)
            if line_match:
                line_count = max(1, min(int(line_match.group(1)), 10))
                return self._rewrite_previous_answer(
                    topic_answer, "summarize", question, line_count
                )

            if follow_up_action == "summarize":
                return self._rewrite_previous_answer(topic_answer, "summarize", question)

            if follow_up_action == "repeat":
                return clean_previous_answer(topic_answer)

            return self._rewrite_previous_answer(topic_answer, follow_up_action, question)

        if re.fullmatch(r"(why|why is that|how so)[?!. ]*", lowered):
            if previous_question:
                previous_intent = SimadChatbot._classify_intent(previous_question, [])
                if previous_intent == "student_guidance":
                    return (
                        "Because the best choice depends on how SIMAD's available programs "
                        "match your interests, strengths, and career goals. Tell me the field "
                        "you are interested in, and I can help you compare suitable options."
                    )
                if previous_intent == "admission_registration":
                    return (
                        "Because SIMAD's admission requirements determine what an applicant "
                        "needs to prepare before applying. I can explain each requirement or "
                        "help you make a preparation checklist."
                    )
                return (
                    f'Your previous question was: "{clean_generated_answer(previous_question)}" '
                    "Which part would you like me to explain further?"
                )
            return "What would you like me to explain?"

        return None

    def _previous_topic_question(self, history: list[dict[str, str]]) -> str:
        marked = next(
            (
                item.get("content", "")
                for item in reversed(history)
                if item.get("role") == "user" and item.get("kind") == "factual_question"
            ),
            "",
        )
        if marked:
            return marked
        for item in reversed(history):
            if item.get("role") != "user":
                continue
            if item.get("kind") in CONVERSATION_MESSAGE_INTENTS | {"follow_up_answer", "follow_up_question", "out_of_scope"}:
                continue
            candidate = " ".join(item.get("content", "").split())
            lowered = candidate.lower()
            if (
                candidate
                and not self._follow_up_action(candidate, has_previous=True)
                and not self._conversation_answer_without_follow_up(lowered)
                and not re.fullmatch(
                    r"(?:sure|ok(?:ay)?|understood|got it|sounds good|"
                    r"no problem|this is messy|that is messy)[?!. ]*",
                    lowered,
                )
            ):
                return candidate
        return ""

    def _previous_topic_answer(self, history: list[dict[str, str]]) -> str:
        marked = next(
            (
                item.get("content", "")
                for item in reversed(history)
                if item.get("role") == "assistant" and item.get("kind") == "factual_answer"
            ),
            "",
        )
        if marked:
            return clean_previous_answer(marked)
        for index in range(len(history) - 1, -1, -1):
            item = history[index]
            if item.get("role") != "assistant" or not item.get("content"):
                continue
            if item.get("kind") in CONVERSATION_MESSAGE_INTENTS | {"follow_up_answer", "follow_up_question", "out_of_scope"}:
                continue
            preceding_question = next(
                (
                    earlier.get("content", "")
                    for earlier in reversed(history[:index])
                    if earlier.get("role") == "user"
                ),
                "",
            )
            compact_question = " ".join(preceding_question.split())
            if (
                preceding_question
                and not self._follow_up_action(compact_question, has_previous=True)
                and not self._conversation_answer_without_follow_up(compact_question.lower())
                and not re.fullmatch(
                    r"(?:sure|ok(?:ay)?|understood|got it|sounds good|"
                    r"no problem|this is messy|that is messy)[?!. ]*",
                    compact_question.lower(),
                )
            ):
                return clean_previous_answer(item["content"])
        return ""

    def _semantic_prototype_scores(
        self, text: str, prototypes: dict[str, tuple[str, ...]]
    ) -> dict[str, float]:
        if not hasattr(self, "model"):
            return {}
        cache = getattr(self, "_intent_prototype_embeddings", {})
        cache_key = tuple((label, values) for label, values in prototypes.items())
        if cache_key not in cache:
            cache[cache_key] = {
                label: self.model.encode(
                    list(values), normalize_embeddings=True, show_progress_bar=False
                )
                for label, values in prototypes.items()
            }
            self._intent_prototype_embeddings = cache
        query_embedding = self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return {
            label: float((embeddings @ query_embedding).max())
            for label, embeddings in cache[cache_key].items()
        }

    def _semantic_topic(self, question: str) -> str:
        """Infer the user's SIMAD topic from meaning rather than exact wording."""
        normalized = canonical_question(normalize_academic_query(question))
        lowered = normalized.lower()
        # History guards MUST come before the leadership keyword check because
        # "rector" appears in both patterns — "who was the rector before X" must
        # route to history, not leadership.
        if re.search(r"\b(?:former|previous)\s+rectors?\b|\brectors?\b.*\b(?:former|previous)\b", lowered):
            return "history"
        if re.search(
            r"\brector\b.*\bbefore\b|\bbefore\b.*\brector\b|\bprevious\s+rector\b|\brector\s+before\b",
            lowered,
        ):
            return "history"
        # "who was before X" — rector predecessor question without explicitly saying 'rector'
        if re.search(r"\bwho\s+(?:was|were|is)\s+(?:(?:the|a)\s+)?(?:rector\s+)?before\b", lowered):
            return "history"
        # "who led SIMAD before the current rector" — distinct from founders
        if re.search(r"\b(?:who\s+led|who\s+ran|who\s+headed)\s+simad\b.*\b(?:before|prior|previous)\b", lowered):
            return "history"
        if re.search(r"\b(who led|who started|who created|who founded simad)\b", lowered):
            return "history"
        if re.search(r"\b(founders?|founding fathers?|who find|who found|who founded)\b", lowered):
            return "history"
        if re.search(r"\b(?:sponsor|sponsored|sponsors|who sponsored)\b", lowered):
            return "history"
        if re.search(
            r"\bwhen\b.*\b(?:become|became|upgrade|upgraded)\b.*\buniversity\b"
            r"|\bwhen\b.*\buniversity\b.*\b(?:become|became|upgrade|upgraded)\b",
            lowered,
        ):
            return "history"
        if re.search(r"\bestablished\b", lowered) and not re.search(
            r"\b(?:well.?established|established\s+(?:faculty|program|school|department|university))",
            lowered,
        ):
            return "history"
        # Leadership check runs after history guards.
        if re.search(
            r"\b(who runs|people in charge|top officials|senior team|"
            r"university officials|who manages|board|senate|rector)\b",
            lowered,
        ):
            return "leadership"
        # Route extracurricular / student-life questions before the semantic fallback.
        if re.search(
            r"\b(extra.?curricular|student.?activit|student\s+club|after.?school\s+activit|sport\s+(?:team|club))\b",
            lowered,
        ):
            return "student_life"
        # Route faculty-count and program-duration questions before the semantic fallback.
        if re.search(
            r"\bhow many facult"
            r"|\ball facult"
            r"|\blist.*facult|\bfacult.*list"
            r"|\bfacult.*(?:available|offered|provide|have|does simad)"
            r"|\bwhat facult"
            r"|\bfacult.*of simad\b",
            lowered,
        ):
            return "academics"
        if re.search(
            r"\bhow many years?\b|\byears? to (?:graduate|complete|finish)\b"
            r"|\bprogram duration\b|\blength of (?:degree|program)\b"
            r"|\bhow long\b.*\b(?:take|last|study|complete|finish|graduate)\b"
            r"|\bhow many semesters?\b",
            lowered,
        ):
            return "tuition"
        scores = self._semantic_prototype_scores(normalized, SEMANTIC_TOPIC_PROTOTYPES)
        if not scores:
            return "unknown"
        topic, best_score = max(scores.items(), key=lambda item: item[1])
        second_score = max(
            (score for label, score in scores.items() if label != topic),
            default=0.0,
        )
        if best_score >= 0.52 or (
            best_score >= 0.38 and best_score - second_score >= 0.025
        ):
            return topic
        return "unknown"

    def _topic_sources(
        self, question: str, semantic_topic: str | None = None
    ) -> tuple[str, ...]:
        """Choose authoritative documents using exact and semantic routing."""
        exact_sources = allowed_sources(question)
        if exact_sources:
            return exact_sources
        faculty_result = find_programs(question)
        if faculty_result:
            faculty, _programs = faculty_result
            sources = ACADEMIC_FACULTY_SOURCES.get(faculty)
            if sources:
                return sources
        parent = find_program_parent(question)
        if parent:
            faculty, _program = parent
            sources = ACADEMIC_FACULTY_SOURCES.get(faculty)
            if sources:
                return sources
        program_label = course_program_name(question)
        if program_label:
            parent = find_program_parent(program_label)
            if parent:
                faculty, _program = parent
                sources = ACADEMIC_FACULTY_SOURCES.get(faculty)
                if sources:
                    return sources
        topic = semantic_topic or self._semantic_topic(question)
        return SEMANTIC_TOPIC_SOURCES.get(topic, ())

    @staticmethod
    def _semantic_retrieval_question(question: str, topic: str) -> str:
        if (
            COURSE_QUERY_PATTERN.search(question)
            or course_program_name(question)
            or find_programs(question)
            or find_program_parent(question)
        ):
            return question
        expansions = {
            "leadership": "SIMAD University leadership rector senate administration officials",
            "admissions": "SIMAD University admission application registration requirements",
            "academics": "SIMAD University faculties degree programs courses curriculum",
            "tuition": "SIMAD University tuition fees semester cost",
            "scholarships": "SIMAD University scholarships financial aid eligibility",
            "campus_services": "SIMAD University campus student services library facilities",
            "exchange": "SIMAD University international student exchange program",
            "research": "SIMAD University research consultancy conferences innovation",
            "student_life": "SIMAD University student clubs activities cultural extracurricular",
            "history": "SIMAD University history founders timeline former rectors",
            "governance": "SIMAD University senate governance policies",
            "vision": "SIMAD University vision mission values",
            "accreditation": "SIMAD University accreditation ranking memberships",
            "grading": "SIMAD University grading GPA grade points",
            "disability_support": "SIMAD University disability accessibility student support",
            "student_conduct": "SIMAD University student conduct discipline rules",
            "overview": "general information about SIMAD University",
        }
        expansion = expansions.get(topic)
        return f"{question}\nSemantic topic: {expansion}" if expansion else question

    def _semantic_follow_up(
        self, question: str, history: list[dict[str, str]] | None
    ) -> bool:
        if not history or not self._previous_topic_question(history):
            return False
        scores = self._semantic_prototype_scores(
            question, FOLLOW_UP_QUESTION_PROTOTYPES
        )
        follow_score = scores.get("follow_up", 0.0)
        new_score = scores.get("new_question", 0.0)
        return follow_score >= 0.40 and follow_score - new_score >= 0.04

    @staticmethod
    def _is_factual_continuation(question: str) -> bool:
        lowered = " ".join(question.lower().split())
        return bool(
            re.fullmatch(
                r"(?:all of them|who are they|list them all|give me all of them|"
                r"tell me more|more details?|continue|go on|what about it|"
                r"give me the rest|show me the rest|list|list them)[?!. ]*",
                lowered,
            )
        )

    def _semantic_scope(self, question: str) -> str:
        """Return simad_related, out_of_scope, or unknown."""
        normalized = canonical_question(normalize_academic_query(question))
        lowered = normalized.lower()
        if OUT_OF_SCOPE_PATTERN.search(lowered) and "simad" not in lowered:
            return "out_of_scope"
        if (
            "simad" in lowered
            or allowed_sources(normalized)
            or find_programs(normalized)
            or find_program_parent(normalized)
            or course_program_name(normalized)
            or self._semantic_topic(normalized) != "unknown"
        ):
            return "simad_related"
        scores = self._semantic_prototype_scores(normalized, SEMANTIC_SCOPE_PROTOTYPES)
        if not scores:
            return "unknown"
        related = scores.get("simad_related", 0.0)
        unrelated = scores.get("out_of_scope", 0.0)
        if unrelated >= 0.42 and unrelated - related >= 0.05:
            return "out_of_scope"
        if related >= 0.40 and related - unrelated >= 0.04:
            return "simad_related"
        return "unknown"

    @staticmethod
    def _prototype_lexical_scores(
        text: str, prototypes: dict[str, tuple[str, ...]]
    ) -> dict[str, float]:
        normalized = " ".join(text.lower().split())
        text_terms = terms(normalized)
        scores: dict[str, float] = {}
        for label, values in prototypes.items():
            best = 0.0
            for prototype in values:
                prototype_normalized = " ".join(prototype.lower().split())
                prototype_terms = terms(prototype_normalized)
                if text_terms and prototype_terms:
                    overlap = len(text_terms & prototype_terms) / max(
                        1, min(len(text_terms), len(prototype_terms))
                    )
                else:
                    overlap = 0.0
                ratio = SequenceMatcher(None, normalized, prototype_normalized).ratio()
                best = max(best, overlap, ratio * 0.72)
            scores[label] = best
        return scores

    def _combined_prototype_scores(
        self, text: str, prototypes: dict[str, tuple[str, ...]]
    ) -> dict[str, float]:
        semantic_scores = self._semantic_prototype_scores(text, prototypes)
        lexical_scores = self._prototype_lexical_scores(text, prototypes)
        labels = set(semantic_scores) | set(lexical_scores)
        return {
            label: max(semantic_scores.get(label, 0.0), lexical_scores.get(label, 0.0))
            for label in labels
        }

    @staticmethod
    def _conversation_reply(intent: str, language: str = "English") -> str:
        if language == "Somali":
            return SOMALI_SMALL_TALK_MESSAGE
        replies = {
            "greeting": "Hello. How can I help you with SIMAD University?",
            "assistant_mood": "I'm doing well and ready to help with SIMAD University.",
            "assistant_identity": (
                "I am the SIMAD University assistant. I answer using verified SIMAD information."
            ),
            "chatbot_capability": (
                "I can help with SIMAD admissions, programs, courses, tuition, scholarships, "
                "services, leadership, and other verified university information."
            ),
            "thanks": "You're welcome. I'm glad I could help.",
            "goodbye": "Goodbye. You can come back whenever you need SIMAD information.",
            "conversation_control": (
                "No problem. What else would you like to know about SIMAD University?"
            ),
            "frustration": (
                "I understand. I apologize for the mistake. Please let me know what was "
                "incorrect, or rephrase your question, and I will try to find the correct "
                "information from SIMAD records."
            ),
        }
        return replies.get(intent, localized_small_talk(language))

    @staticmethod
    def _has_factual_history(history: list[dict[str, str]] | None) -> bool:
        if not history:
            return False
        return any(
            item.get("role") == "assistant"
            and item.get("content")
            and item.get("kind") in {"factual_answer", None, ""}
            for item in history
        )

    @staticmethod
    def _is_assistant_directed(text: str) -> bool:
        return bool(
            re.search(
                r"\b(you|your|yourself|assistant|bot|chatbot|help me|can you|could you)\b",
                text,
                re.I,
            )
        )

    @staticmethod
    def _conversation_signal_intent(lowered: str) -> str | None:
        if re.search(
            r"\b(fuck|shut up|stupid|idiot|hate you|wrong\s+answer|incorrect|that's\s+wrong|that\s+is\s+wrong|you\s+are\s+wrong|not\s+correct)\b",
            lowered,
        ):
            return "frustration"
        if re.search(r"\b(goodbye|bye|see you|farewell)\b", lowered):
            return "goodbye"
        if re.search(
            r"\b(skip|never mind|nevermind|forget it|leave it|okay|ok|sure|"
            r"understood|got it|sounds good|correct|another thing|another topic|"
            r"change topic)\b",
            lowered,
        ):
            return "conversation_control"
        if re.search(
            r"\b(remember|recall)\b.*\b(conversation|chat|question|asked|topic|name)\b",
            lowered,
        ):
            return "memory"
        if re.search(r"\b(who am i|my name|call me|remember my name)\b", lowered):
            return "user_identity"
        if (
            re.search(r"\b(help|capabilit|can do|able to do)\b", lowered)
            and SimadChatbot._is_assistant_directed(lowered)
        ):
            return "chatbot_capability"
        if re.search(r"\b(who|what)\b.*\b(you|your|assistant|bot|chatbot)\b", lowered):
            return "assistant_identity"
        # Guard: only treat as mood if NOT a factual "how many/much/do you know" question
        if re.search(r"\bhow\b.*\b(you|yourself|assistant|bot|chatbot)\b", lowered):
            _factual_how = re.search(
                r"\b(many|much|long|often|old|far|know about|know of|you know|do you know"
                r"|can you tell|can you list|tell me about)\b",
                lowered,
            )
            if not _factual_how:
                return "assistant_mood"
        return None

    def _classify_conversation_intent(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> str | None:
        normalized = " ".join(question.split())
        if not normalized:
            return None

        lowered = normalized.lower()
        has_factual_history = self._has_factual_history(history)
        signal_intent = self._conversation_signal_intent(lowered)
        if (
            has_factual_history
            and signal_intent not in {
                "greeting",
                "assistant_mood",
                "assistant_identity",
                "chatbot_capability",
                "thanks",
                "goodbye",
                "conversation_control",
                "frustration",
                "memory",
                "user_identity",
            }
            and ANSWER_TRANSFORM_FALLBACK_PATTERN.fullmatch(lowered)
        ):
            return None
        if (
            has_factual_history
            and signal_intent is None
            and self._follow_up_action(normalized, has_previous=True)
        ):
            return None
        if has_factual_history and self._is_factual_continuation(normalized):
            return None

        if signal_intent:
            intent = signal_intent
            if intent in {
                "greeting",
                "thanks",
                "goodbye",
                "conversation_control",
                "frustration",
            }:
                return intent
        else:
            scores = self._combined_prototype_scores(normalized, CONVERSATION_INTENT_PROTOTYPES)
            if not scores:
                return None
            intent, best_score = max(scores.items(), key=lambda item: item[1])
            second_score = max(
                (score for label, score in scores.items() if label != intent),
                default=0.0,
            )
            if best_score < 0.36 or (best_score < 0.52 and best_score - second_score < 0.025):
                return None

        assistant_directed = self._is_assistant_directed(lowered)
        if (
            re.search(r"\bdo you know\b", lowered)
            and not SIMAD_SCOPE_PATTERN.search(normalized)
        ):
            return None
        if intent == "user_identity" and not re.search(
            r"\b(who am i|my name|call me|remember my name|i am)\b",
            lowered,
        ):
            intent = "assistant_identity" if assistant_directed else ""
            if not intent:
                return None
        if intent in {
            "assistant_mood",
            "assistant_identity",
            "chatbot_capability",
            "memory",
        } and not assistant_directed:
            return None

        simad_factual_signal = bool(
            allowed_sources(normalized)
            or course_program_name(normalized)
            or find_programs(normalized)
            or find_program_parent(normalized)
            or (
                SIMAD_SCOPE_PATTERN.search(normalized)
                and not re.search(r"\b(you|your|assistant|bot|chatbot)\b", lowered)
            )
        )
        if simad_factual_signal and intent not in {"thanks", "frustration"}:
            return None

        return intent

    def _follow_up_action(self, question: str, has_previous: bool) -> str | None:
        if not has_previous:
            return None
        lowered = " ".join(question.lower().split())
        if re.search(
            r"\b(?:mention|give|tell|pick|select|show)\s+(?:me\s+)?(?:one|only one|just one)\b"
            r"|\b(?:one|only one|just one)\b.*\b(?:only|rector|program|faculty)\b"
            r"|\b(?:mention one|give me one|only one|just one)\b",
            lowered,
        ):
            return "select_one"
        if re.fullmatch(r"(?:only|only these|is that all|are these all|just these)[?!. ]*", lowered):
            return "confirm_only"
        word_count = len(re.findall(r"[a-z0-9]+", lowered))
        if re.search(
            r"\b(condense|essential|key points?|too long|brief|concise|shorten|short version|shorter)\b",
            lowered,
        ):
            return "summarize"
        if re.search(r"\b(too short|expand|more detail|more explanation)\b", lowered):
            return "expand"
        if re.search(r"\b(messy|clean|cleaner|organi[sz]e)\b", lowered):
            return "clean"
        if re.search(r"\b(simple|simply|plain|easy|easier|restate)\b", lowered):
            return "simplify"
        if re.search(r"\b(i (?:do not|don't) understand|confus(?:ed|ing)|clarify)\b", lowered):
            return "clarify"
        if re.search(r"\b(repeat|once more|show me (?:your )?response)\b", lowered):
            return "repeat"
        conversation_scores = self._combined_prototype_scores(
            lowered, CONVERSATION_INTENT_PROTOTYPES
        )
        if conversation_scores and not ANSWER_TRANSFORM_FALLBACK_PATTERN.fullmatch(lowered):
            conversation_intent, conversation_score = max(
                conversation_scores.items(), key=lambda item: item[1]
            )
            if (
                conversation_intent
                in {
                    "greeting",
                    "assistant_mood",
                    "assistant_identity",
                    "chatbot_capability",
                    "thanks",
                    "goodbye",
                    "conversation_control",
                    "frustration",
                    "user_identity",
                }
                and conversation_score >= 0.42
            ):
                return None
        explicit_reference = bool(
            re.search(
                r"\b(previous|last|answer|response|it|this|that|again|"
                r"shorter|simply|clearer|clean|cleaner|messy|confused|"
                r"repeat|summary|summarize|expand|detail|clarify|too short)\b",
                lowered,
            )
        )
        named_simad_topic = bool(
            SIMAD_SCOPE_PATTERN.search(lowered)
            or allowed_sources(lowered)
            or course_program_name(lowered)
            or find_programs(lowered)
        )
        if named_simad_topic and not explicit_reference:
            return None

        scores = self._semantic_prototype_scores(lowered, FOLLOW_UP_ACTION_PROTOTYPES)
        if scores:
            action, best_score = max(scores.items(), key=lambda item: item[1])
            second_score = max(
                (score for label, score in scores.items() if label != action),
                default=0.0,
            )
            if (
                word_count <= 16
                and best_score >= 0.32
                and (best_score - second_score >= 0.015 or explicit_reference)
            ):
                if re.search(r"\b(short|shorter|brief|concise|summary|summarize|important|long)\b", lowered):
                    return "summarize"
                if re.search(r"\b(question|topic|discussing|about)\b", lowered):
                    return "topic"
                if re.search(r"\b(repeat|again|once more)\b", lowered):
                    return "repeat"
                return action

        if ANSWER_TRANSFORM_FALLBACK_PATTERN.fullmatch(lowered):
            if re.search(r"\b(?:what|which).*(?:question|about)\b", lowered):
                return "topic"
            if re.search(r"\b(?:repeat|again)\b", lowered):
                return "repeat" if re.search(r"\brepeat\b", lowered) else "clarify"
            if re.search(
                r"\b(?:simple|simply|plain|easy|easier)\b",
                lowered,
            ):
                return "simplify"
            if re.search(r"\b(?:understand|mean|explain|clarify|confus(?:ed|ing))\b", lowered):
                return "clarify"
            if re.search(r"\b(?:clean|messy|organi[sz]e)\b", lowered):
                return "clean"
            if re.search(r"\b(?:too short|expand|detail)\b", lowered):
                return "expand"
            return "summarize"
        return None

    def _classify_message_intent(
        self, question: str, history: list[dict[str, str]]
    ) -> str:
        lowered = " ".join(question.lower().split())
        if OUT_OF_SCOPE_PATTERN.search(lowered) and "simad" not in lowered:
            return "out_of_scope"
        if re.search(
            r"\b(skip|never mind|nevermind|forget it|leave it|okay|ok|sure|"
            r"understood|got it|sounds good|another thing|another topic|change topic)\b",
            lowered,
        ):
            return "conversation_control"

        # Check fast regex-based signals FIRST so clear greetings/farewells/frustration
        # are never mis-routed as factual follow-ups when turns alternate.
        conv_signal = self._conversation_signal_intent(lowered)
        if conv_signal:
            return conv_signal

        has_previous = bool(self._previous_topic_answer(history)) if history else False
        follow_up_action = self._follow_up_action(question, has_previous)
        semantic_follow_up = self._semantic_follow_up(question, history)
        if semantic_follow_up and self._is_factual_continuation(question):
            return "follow_up_question"
        if has_previous and self._is_factual_continuation(question):
            return "follow_up_question"

        if follow_up_action:
            return "follow_up_answer"

        semantic_topic = self._semantic_topic(question)
        if semantic_topic != "unknown" and not self._is_assistant_directed(lowered):
            return "new_simad_question"

        # Full semantic conversation classifier runs AFTER topic check so that
        # factual questions like "who runs the institution" are not accidentally
        # matched as thanks/assistant_identity by the semantic model.
        conversation_intent = self._classify_conversation_intent(question, history)
        if conversation_intent:
            return conversation_intent

        if re.search(r"\bdo you know\b", lowered) and not self._is_likely_simad_scope(question):
            return "out_of_scope"

        if self._is_follow_up(question, history) or semantic_follow_up:
            return "follow_up_question"
        semantic_scope = self._semantic_scope(question)
        if semantic_scope == "simad_related" or self._is_likely_simad_scope(question):
            return "new_simad_question"
        if semantic_scope == "out_of_scope":
            return "out_of_scope"

        scores = self._semantic_prototype_scores(lowered, MESSAGE_INTENT_PROTOTYPES)
        if scores:
            label, best_score = max(scores.items(), key=lambda item: item[1])
            if label == "small_talk" and best_score >= 0.38:
                return "conversation_control"
            if label == "out_of_scope" and best_score >= 0.34:
                return label
        if OUT_OF_SCOPE_PATTERN.search(lowered):
            return "out_of_scope"
        return "out_of_scope"

    def _conversation_answer_without_follow_up(self, lowered: str) -> bool:
        return self._classify_conversation_intent(lowered, []) is not None

    @staticmethod
    def _classify_intent(question: str, history: list[dict[str, str]]) -> str:
        lowered = " ".join(question.lower().split())
        if SimadChatbot._is_follow_up(question, history):
            return "follow_up"
        if COURSE_QUERY_PATTERN.search(question):
            return "course_curriculum"
        if PROGRAM_QUERY_PATTERN.search(question):
            return "academic_programs"
        if GUIDANCE_PATTERN.search(lowered):
            return "student_guidance"
        if ADMISSION_PATTERN.search(lowered):
            return "admission_registration"
        if OUT_OF_SCOPE_PATTERN.search(lowered) and "simad" not in lowered:
            return "out_of_scope"
        return "simad_factual"

    @staticmethod
    def _is_follow_up(
        question: str, history: list[dict[str, str]] | None = None
    ) -> bool:
        lowered = " ".join(question.lower().split())
        if history and SimadChatbot._is_factual_continuation(question):
            return True
        if re.fullmatch(r"(why|why is that|how so)[?!. ]*", lowered):
            return True
        if re.match(r"^(how about|what about|and what about|i mean|also)\b", lowered):
            return True
        if re.search(
            r"\b(?:tell me more|explain (?:it|that|this)|what about (?:it|that|them)|"
            r"what do you mean|what you mean)\b",
            lowered,
        ):
            return True

        if history:
            previous_answer = next(
                (
                    item["content"]
                    for item in reversed(history)
                    if item.get("role") == "assistant"
                ),
                "",
            )
            if "please specify" in previous_answer.lower() and len(terms(question)) <= 5:
                return True

        # A named SIMAD topic makes the question self-contained even if it uses a pronoun,
        # for example: "Does SIMAD offer scholarships, and who qualifies for them?"
        if allowed_sources(question) or "simad" in lowered:
            return False
        return bool(FOLLOW_UP_PATTERN.search(question))

    def _contextual_question(self, question: str, history: list[dict[str, str]]) -> str:
        if not (
            self._is_follow_up(question, history)
            or self._semantic_follow_up(question, history)
        ):
            return question
        previous = self._previous_topic_question(history)
        previous_answer = next(
            (item["content"] for item in reversed(history) if item.get("role") == "assistant"),
            "",
        )
        lowered = " ".join(question.lower().split())
        previous_topic = (
            self._semantic_topic(
                f"{previous}\n{self._previous_topic_answer(history)}"
            )
            if previous
            else "unknown"
        )
        if previous and re.fullmatch(
            r"(?:all of them|who are they|list them all|give me all of them|"
            r"give me the rest|show me the rest|list|list them)[?!. ]*",
            lowered,
        ):
            if previous_topic in {"leadership", "governance"}:
                return "List all verified SIMAD University Senate leaders and officials."
            return f"{previous}\nList all verified items related to this SIMAD question."
        if previous and re.fullmatch(
            r"(?:tell me more|more details?|continue|go on|what about it)[?!. ]*",
            lowered,
        ):
            if previous_topic in {"leadership", "governance"}:
                return "List all verified SIMAD University Senate leaders and officials."
            return f"{previous}\nProvide more verified information about this SIMAD question."
        if previous and re.match(r"^(how about|what about|and what about)\b", lowered):
            previous_intent = SimadChatbot._classify_intent(previous, [])
            subject = re.sub(
                r"^(?:how about|what about|and what about)\s+", "", question, flags=re.I
            )
            if previous_intent == "academic_programs":
                return f"What undergraduate programs are in {subject}?"
            if previous_intent == "course_curriculum":
                # Allow the follow-up to change the semester, the program, or both.
                sem = semester_number(question)
                new_program = course_program_name(subject)
                prev_program = course_program_name(previous)
                new_q = previous
                # Replace program name first (so the semester sub works on the updated string)
                if (
                    new_program
                    and prev_program
                    and new_program.lower() != prev_program.lower()
                ):
                    new_q = re.sub(
                        re.escape(prev_program), new_program, new_q, flags=re.I, count=1
                    )
                # Replace semester number if the follow-up specifies one
                if sem:
                    new_q = re.sub(
                        r"\bsemester\s+\w+\b",
                        f"semester {sem}",
                        new_q,
                        flags=re.I,
                        count=1,
                    )
                if new_q != previous:
                    return new_q
            if COURSE_QUERY_PATTERN.search(question) and course_program_name(question):
                return question
        if re.match(r"^i mean\b", lowered):
            return question
        if previous and "please specify" in previous_answer.lower():
            return f"{previous}\nUser clarification: {question}"
        return f"{previous}\nFollow-up question: {question}" if previous else question

    @staticmethod
    def _program_answer(question: str) -> str:
        """Legacy compatibility shim; factual answers now go through retrieval."""
        return ""

    @staticmethod
    def _pretty_academic_label(value: str) -> str:
        return " ".join(word.capitalize() for word in normalize_academic_query(value).split())

    @staticmethod
    def _student_guidance_context(question: str) -> str:
        normalized = normalize_academic_query(question)
        guidance_words = {
            "advice",
            "advise",
            "career",
            "choose",
            "choosing",
            "field",
            "help",
            "interest",
            "interested",
            "program",
            "programs",
            "recommend",
            "study",
            "want",
        }
        wanted = terms(normalized) - guidance_words
        scored = []
        for faculty, programs in faculty_programs().items():
            searchable = terms(faculty)
            for program in programs:
                searchable |= terms(program)
            score = len(wanted & searchable)
            if score:
                scored.append((score, faculty, programs))

        selected = []
        if scored:
            best = max(score for score, _, _ in scored)
            selected = [
                (faculty, programs)
                for score, faculty, programs in scored
                if score == best
            ]
            heading = "Verified undergraduate programs matching the student's interest:"
        else:
            selected = list(faculty_programs().items())
            heading = "Verified undergraduate faculties and programs:"

        lines = [heading]
        for faculty, programs in selected:
            lines.append(f"- Faculty of {faculty}: {', '.join(programs)}")
        return "\n".join(lines)

    @staticmethod
    def _comparison_context(question: str) -> str:
        if not ACADEMIC_COMPARISON_PATTERN.search(question):
            return ""
        normalized = normalize_academic_query(question)
        lowered = normalized.lower()
        query_terms = terms(normalized)
        generic_faculty_terms = {
            "faculty",
            "faculties",
            "school",
            "schools",
            "science",
            "sciences",
        }
        matched = []
        for faculty, programs in faculty_programs().items():
            faculty_terms = terms(faculty) - generic_faculty_terms
            faculty_label = faculty.lower()
            compact_faculty_label = re.sub(r"\s*&\s*", " and ", faculty_label)
            distinctive_overlap = faculty_terms & query_terms
            if faculty_label in lowered or compact_faculty_label in lowered or distinctive_overlap:
                matched.append((faculty, programs))

        unique = []
        seen = set()
        for faculty, programs in matched:
            if faculty.lower() in seen:
                continue
            seen.add(faculty.lower())
            unique.append((faculty, programs))

        if len(unique) < 2:
            return ""
        lines = ["Verified academic comparison from local SIMAD program data:"]
        for faculty, programs in unique[:4]:
            lines.append(f"- Faculty of {faculty}: {', '.join(programs)}")
        lines.append(
            "The available data supports comparing the faculties by their verified "
            "program lists only; it does not provide a detailed curriculum comparison."
        )
        return "\n".join(lines)

    @staticmethod
    def _course_program_availability_context(question: str) -> str:
        program_label = course_program_name(question)
        if not program_label:
            return ""

        courses = [
            course
            for course in all_courses()
            if display_program_name(course.program).lower() == program_label.lower()
        ]
        faculties = []
        for course in courses:
            faculty = re.sub(r"^Faculty of\s+", "", course.faculty, flags=re.I).strip()
            if faculty and faculty not in faculties:
                faculties.append(faculty)
        if not faculties:
            return ""

        lines = ["Verified academic availability from local SIMAD course data:"]
        lines.append(
            f"Requested academic label: {SimadChatbot._pretty_academic_label(program_label)}"
        )
        for faculty in faculties:
            lines.append(f"Verified faculty: Faculty of {faculty}")
            programs = faculty_programs().get(faculty)
            if programs:
                lines.append(f"Verified undergraduate programs in that faculty: {', '.join(programs)}")
        return "\n".join(lines)

    @staticmethod
    def _database_context(question: str, intent: str) -> str:
        if intent == "student_guidance":
            return SimadChatbot._student_guidance_context(question)

        admin_context = administration_context(question)
        if admin_context:
            return admin_context

        comparison_context = SimadChatbot._comparison_context(question)
        if comparison_context:
            return comparison_context

        lowered = question.lower()
        # Program duration: "how many years to graduate / complete / finish" or "how long does X take"
        if re.search(
            r"\bhow many years?\b|\byears? to (?:graduate|complete|finish)\b"
            r"|\bprogram duration\b|\blength of (?:degree|program)\b"
            r"|\bhow long\b.*\b(?:take|last|study|complete|finish|graduate)\b"
            r"|\b(?:take|last|study|complete|finish)\b.*\bhow long\b",
            lowered,
        ):
            records = tuition_records()
            if records:
                target_program = course_program_name(question)
                if target_program:
                    matching = [
                        r for r in records
                        if terms(target_program) <= terms(normalize_academic_query(r.program))
                    ]
                else:
                    matching = list(records)
                if matching:
                    lines = ["Verified program durations from SIMAD data:"]
                    for record in matching[:20]:
                        lines.append(
                            f"- {normalize_academic_query(record.program)}: {record.years} years"
                        )
                    return "\n".join(lines)

        # Semester count: "how many semesters does X take / have"
        if re.search(
            r"\bhow many semesters?\b",
            lowered,
        ):
            records = tuition_records()
            if records:
                target_program = course_program_name(question)
                if target_program:
                    matching = [
                        r for r in records
                        if terms(target_program) <= terms(normalize_academic_query(r.program))
                    ]
                else:
                    matching = list(records)
                if matching:
                    lines = ["Verified program durations from SIMAD data:"]
                    for record in matching[:10]:
                        semesters = record.years * 2
                        lines.append(
                            f"- {normalize_academic_query(record.program)}: "
                            f"{record.years} years = {semesters} semesters"
                        )
                    return "\n".join(lines)

        # Faculty list: "how many faculties / list all faculties / faculties available"
        # Also catches: 'list the faculties', 'list faculties of SIMAD', 'what faculties'
        if re.search(
            r"\bhow many facult"
            r"|\ball facult"
            r"|\blist.*facult|\bfacult.*list"
            r"|\bfacult.*(?:available|offered|provide|have|does simad)"
            r"|\bwhat facult"
            r"|\bfacult.*of simad\b",
            lowered,
        ):
            all_faculties = faculty_programs()
            if all_faculties:
                lines = [
                    "Verified faculties list:",
                    f"SIMAD University has {len(all_faculties)} verified faculties:"
                ]
                for faculty, programs in all_faculties.items():
                    lines.append(
                        f"- Faculty of {faculty}: undergraduate programs: {', '.join(programs)}"
                    )
                return "\n".join(lines)

        # Faculty programs: "what programs are under Faculty of X" — use find_programs for full list
        if re.search(r"\bprograms?\b.*\bfacult|\bfacult\b.*\bprograms?", lowered):
            fp_result = find_programs(question)
            if fp_result:
                fac, progs = fp_result
                lines = [
                    f"Verified faculty: Faculty of {fac}",
                    "Verified undergraduate programs: " + ", ".join(progs),
                ]
                return "\n".join(lines)

        if "tuition" in lowered or "fee" in lowered or "cost" in lowered or "price" in lowered:
            records = tuition_records()
            if records:
                target_program = course_program_name(question)
                faculty_match = find_programs(question)
                wanted = terms(question) - {
                    "tuition",
                    "fee",
                    "fees",
                    "semester",
                    "cost",
                    "price",
                    "highest",
                    "most",
                    "maximum",
                    "max",
                    "expensive",
                }
                selected_records = []
                if re.search(r"\b(most|highest|max|maximum|expensive)\b", lowered):
                    selected_records = [max(records, key=lambda item: item.total)]
                elif target_program:
                    target_terms = terms(target_program)
                    selected_records = [
                        record
                        for record in records
                        if target_terms <= terms(normalize_academic_query(record.program))
                    ]
                elif faculty_match:
                    _, programs = faculty_match
                    selected_records = [
                        record
                        for record in records
                        if any(program.lower() in record.program.lower() for program in programs)
                    ]
                else:
                    scored_records = []
                    for record in records:
                        score = len(wanted & terms(normalize_academic_query(record.program)))
                        if score:
                            scored_records.append((score, record))
                    if scored_records:
                        best_score = max(score for score, _ in scored_records)
                        selected_records = [
                            record for score, record in scored_records if score == best_score
                        ]
                if not selected_records:
                    if target_program or faculty_match:
                        # A specific program was requested but is not offered at SIMAD.
                        # Return empty so the retrieval layer can produce a not-found answer
                        # rather than dumping all programs as if they match.
                        return ""
                    # General tuition query with no specific target — show all records.
                    selected_records = records
                lines = ["Verified tuition records from local SIMAD data:"]
                for record in selected_records[:20]:
                    lines.append(
                        f"- {normalize_academic_query(record.program)}; years: "
                        f"{record.years}; tuition: ${record.fee:,}; charges: "
                        f"${record.charges:,}; total semester fee: ${record.total:,}"
                    )
                return "\n".join(lines)

        if COURSE_QUERY_PATTERN.search(question):
            courses = find_courses(question)
            if courses:
                program = course_program_name(question) or courses[0].program
                semester = semester_number(question)
                full_details = bool(
                    re.search(r"\b(full curriculum table|full details|all details)\b", lowered)
                )
                include_code = full_details or bool(
                    re.search(r"\b(with|include|show)\s+(?:course\s+)?codes?\b", lowered)
                    or re.search(r"\bcourse codes?\b", lowered)
                )
                include_credits = full_details or bool(
                    re.search(r"\b(with|include|show)\s+credit hours?\b", lowered)
                    or re.search(r"\bcredits?\b", lowered)
                )
                heading = f"Verified course records for {program}"
                if semester is not None:
                    heading += f" semester {semester}"
                lines = [heading + ":"]
                for course in courses:
                    pieces = [course.title]
                    if include_code:
                        pieces.append(f"code: {course.code}")
                    if include_credits:
                        pieces.append(f"credit hours: {course.credits}")
                    if full_details:
                        pieces.extend(
                            [
                                f"semester: {course.semester}",
                                f"theory contact hours: {course.theory or '0'}",
                                f"practical contact hours: {course.practice or '0'}",
                            ]
                        )
                    lines.append("- " + "; ".join(pieces))
                return "\n".join(lines)
            named_courses = find_named_courses(question)
            if named_courses:
                lines = ["Verified matching course records:"]
                for course in named_courses:
                    lines.append(
                        f"- {course.title}; faculty: {course.faculty}; program label: "
                        f"{course.program}; semester: {course.semester}; code: {course.code}; "
                        f"credit hours: {course.credits}"
                    )
                return "\n".join(lines)
            return ""

        parent = find_program_parent(question)
        if parent and (
            PROGRAM_AVAILABILITY_PATTERN.search(question)
            or (
                PROGRAM_QUERY_PATTERN.search(question)
                # Do NOT let this branch fire when the user is asking for the full program list of
                # a faculty (e.g. "What programs are under Faculty of Economics?") — those should
                # go through find_programs above to return all programs, not just one match.
                and not re.search(r"\bfacult(?:y|ies)\b|\bschool\b", question, re.I)
                and not re.search(r"\bprograms?\b.*\bfacult|\bfacult\b.*\bprograms?", question, re.I)
            )
        ):
            faculty, program = parent
            return (
                "Verified academic availability from local SIMAD program data:\n"
                f"Requested question: {question}\n"
                f"Matched verified undergraduate program: {program}\n"
                f"Verified faculty: Faculty of {faculty}"
            )

        course_program_context = SimadChatbot._course_program_availability_context(question)
        if course_program_context and PROGRAM_AVAILABILITY_PATTERN.search(question):
            return course_program_context

        result = find_programs(question)
        if result:
            faculty, programs = result
            return (
                "Use only the exact facts below; do not infer a description of the faculty.\n"
                f"Verified faculty: Faculty of {faculty}\n"
                f"Verified undergraduate programs: {', '.join(programs)}"
            )

        named_courses = find_named_courses(question)
        if named_courses:
            lines = ["Verified matching course records:"]
            for course in named_courses:
                lines.append(
                    f"- {course.title}; faculty: {course.faculty}; program label: "
                    f"{course.program}; semester: {course.semester}; code: {course.code}; "
                    f"credit hours: {course.credits}"
                )
            return "\n".join(lines)
        return ""

    @staticmethod
    @lru_cache(maxsize=1)
    def _verified_senate_members() -> tuple[str, ...]:
        """Legacy compatibility shim; factual answers now go through retrieval."""
        return tuple()

    @staticmethod
    def _faculty_overview_answer(question: str) -> str | None:
        """Legacy compatibility shim; factual answers now go through retrieval."""
        return None

    @staticmethod
    def _structured_answer(question: str) -> str | None:
        """Legacy compatibility shim; factual answers now go through retrieval."""
        return None

    @staticmethod
    def _overview_answer_from_matches(matches: list[SearchResult]) -> str:
        qa_pairs: list[tuple[str, str]] = []
        for match in matches:
            if not Path(match.source).name.endswith("SIMAD UNIVERSITY GENERAL INFORMATION.pdf"):
                continue
            for qa_question, qa_answer in re.findall(
                r"\bQ:\s*(.+?)\s+A:\s*(.*?)(?=\s+Q:|\Z)",
                clean_evidence_text(match.text),
                flags=re.I | re.S,
            ):
                if is_project_noise(f"{qa_question} {qa_answer}"):
                    continue
                qa_pairs.append(
                    (
                        clean_generated_answer(qa_question),
                        clean_generated_answer(qa_answer),
                    )
                )

        overview = ""
        vision_or_mission = ""
        for qa_question, qa_answer in qa_pairs:
            lowered = qa_question.lower()
            if "what is simad university" in lowered and not overview:
                overview = qa_answer
            elif (
                ("vision" in lowered or "mission" in lowered)
                and not vision_or_mission
                and re.search(r"\bquality (?:higher )?education\b", qa_answer, re.I)
            ):
                vision_or_mission = qa_answer

        if not overview:
            return ""

        overview_sentences = split_sentences(overview)
        first_sentence = next(
            (
                sentence
                for sentence in overview_sentences
                if re.search(r"\bprivate\b.*\bhigher education\b", sentence, re.I)
            ),
            overview_sentences[0] if overview_sentences else overview,
        )
        year_match = re.search(r"\bestablished in\s+(\d{4})\b", overview, re.I)
        second_sentence = ""
        if vision_or_mission:
            focus = re.sub(
                r"^SIMAD University (?:aims to|mission is to)\s+",
                "",
                split_sentences(vision_or_mission)[0],
                flags=re.I,
            )
            focus = re.sub(r"^provide\s+", "", focus, flags=re.I)
            focus = re.sub(r"\bquality higher education\b", "quality education", focus, flags=re.I)
            focus = re.sub(r"\s+that contribute.*$", "", focus, flags=re.I).strip(" .")
            if year_match:
                second_sentence = f"It was established in {year_match.group(1)} and focuses on {focus}."
            else:
                second_sentence = f"It focuses on {focus}."
        elif year_match:
            second_sentence = f"It was established in {year_match.group(1)}."

        return "\n".join(
            sentence for sentence in [first_sentence, second_sentence] if sentence
        )

    @staticmethod
    def _admission_answer_from_matches(question: str, matches: list[SearchResult]) -> str:
        passages = []
        for match in matches:
            if not Path(match.source).name.endswith("ADMISSION BROCHURE.pdf"):
                continue
            passages.extend(evidence_passages(match.text))
        requirement_sections = [
            passage
            for passage in passages
            if re.search(r"\bADMISSION REQUIREMENTS\b", passage, re.I)
        ]
        if not requirement_sections:
            return ""

        def _section_score(section: str) -> tuple[int, int, int]:
            # Primary: prefer sections that open directly with ADMISSION REQUIREMENTS
            # (no bank account or other preamble before the heading).
            starts_clean = 1 if re.match(r"^\s*ADMISSION REQUIREMENTS\b", section, re.I) else 0
            # Secondary: prefer sections with more requirement action verbs.
            req_verbs = len(re.findall(r"\b(?:Should|Bring|Pay|Submit|Provide)\b", section, re.I))
            # Tertiary: among ties, prefer shorter (less noise).
            return (starts_clean, req_verbs, -len(section))

        section = max(requirement_sections, key=_section_score)
        body = re.sub(r"^\s*ADMISSION REQUIREMENTS\s*", "", section, flags=re.I).strip()
        body = re.sub(r"\s+\d{2}\s+(?=[A-Z])", "\n", body)
        body = re.sub(r"\s+\d{2}\s*$", "", body)
        body = re.sub(r"\s+%", "%", body)
        parts = [part.strip(" .") for part in body.splitlines() if part.strip(" .")]
        if len(parts) <= 1:
            parts = [
                part.strip(" .")
                for part in re.split(
                    r"(?=\b(?:Should|Pay|Bring|Pass|Submit|Provide)\b)",
                    body,
                    flags=re.I,
                )
                if part.strip(" .")
            ]
        cleaned_items = []
        seen = set()
        for part in parts:
            item = _simplify_answer_item(part)
            item = re.sub(r"\s+%", "%", item).strip(" .")
            if not item:
                continue
            key = re.sub(r"\W+", " ", item.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            cleaned_items.append(item)

        if not cleaned_items:
            return ""

        lowered = question.lower()
        if re.search(r"\b(register|registration|apply|application|join|enroll|enrol)\b", lowered):
            heading = "To apply or register at SIMAD, the available admission requirements are:"
        else:
            heading = "SIMAD admission requirements:"
        return "\n".join(
            [heading]
            + [
                f"{index}. {item}."
                for index, item in enumerate(cleaned_items[:8], start=1)
            ]
        )

    @staticmethod
    def _scholarship_answer_from_matches(
        question: str, matches: list[SearchResult]
    ) -> str:
        scholarship_texts = [
            clean_evidence_text(match.text)
            for match in matches
            if Path(match.source).name.endswith("Scholarships.pdf")
        ]
        if not scholarship_texts:
            return ""
        text = clean_generated_answer(" ".join(scholarship_texts))
        options: list[str] = []
        including_match = re.search(
            r"We offer various forms of financial support,\s+including\s+(.+?Low-Income Scholarships)",
            text,
            flags=re.I | re.S,
        ) or re.search(
            r"including\s+(.+?)\.(?:\s|$)",
            text,
            flags=re.I | re.S,
        )
        if including_match:
            raw_options = including_match.group(1)
            if len(raw_options) <= 350 and "Quran competition" not in raw_options:
                raw_options = re.sub(r"\s+,", ",", raw_options)
                raw_options = raw_options.replace(", and ", ", ")
                raw_options = re.sub(r"\s+and\s+(?=[A-Z][A-Za-z -]+Scholarship)", ", ", raw_options)
                for item in raw_options.split(","):
                    cleaned = re.sub(r"^(?:and|the)\s+", "", item.strip(), flags=re.I)
                    if cleaned:
                        cleaned = cleaned[0].upper() + cleaned[1:]
                    if cleaned and cleaned.lower() not in {option.lower() for option in options}:
                        options.append(cleaned)

        qualification_lines = []
        if re.search(r"academic merit scholarships", text, re.I):
            qualification_lines.append(
                "Academic merit scholarships are linked to academic excellence."
            )
        if re.search(r"Dr\.\s*Sumait Hospital Scholarship\s*\(especially for women\)", text, re.I):
            qualification_lines.append(
                "The Dr. Sumait Hospital Scholarship is especially for women."
            )
        low_income_match = re.search(
            r"scholarships for low-income students are intended for\s+(.+?)(?:\.| If you|$)",
            text,
            flags=re.I,
        )
        if low_income_match:
            qualification_lines.append(
                "Low-income scholarships are intended for "
                + low_income_match.group(1).strip()
                + "."
            )
        quran_match = re.search(
            r"prospective students must register at the university and participate in the cultural week[^.]*",
            text,
            flags=re.I,
        )
        if quran_match:
            quran_line = clean_generated_answer(quran_match.group(0)).rstrip(".")
            if quran_line:
                quran_line = quran_line[0].upper() + quran_line[1:]
            qualification_lines.append(
                quran_line + "."
            )

        output = []
        if options:
            output.append("SIMAD records mention these scholarship and support options:")
            output.extend(f"- {option}" for option in options[:8])
        if qualification_lines:
            if output:
                output.append("Qualification details found:")
            else:
                output.append("SIMAD scholarship qualification details found:")
            output.extend(f"- {line}" for line in qualification_lines[:6])
        if not output:
            return ""
        if re.search(r"\bqualif|eligible|who\b", question, re.I) and not qualification_lines:
            output.append(
                "The exact eligibility details were not found in the available SIMAD data."
            )
        return "\n".join(output)

    @staticmethod
    def _grading_answer_from_matches(matches: list[SearchResult]) -> str:
        text = " ".join(
            clean_evidence_text(match.text)
            for match in matches
            if Path(match.source).name.endswith("Grading System and GPA.pdf")
        )
        if not text:
            return ""
        rows = re.findall(
            r"\b\d+\s+(\d{1,3}-\d{1,3})\s+(\d\.\d{2})\s+([A-F][+-]?)\s+"
            r"([A-Za-z ]+?)(?=\s+\d+\s+\d{1,3}-|\s*$)",
            text,
        )
        if not rows:
            return clean_generated_answer(text)
        output = ["SIMAD grading system:"]
        for marks, points, grade, description in rows[:10]:
            output.append(f"- {marks} {points} {grade} ({description.strip()}).")
        return "\n".join(output)

    @staticmethod
    def _senate_answer_from_matches(matches: list[SearchResult]) -> str:
        senate_matches = [
            match
            for match in matches
            if Path(match.source).name.endswith("THE SENATE.pdf")
        ]
        if not senate_matches:
            return ""

        def page_number(match: SearchResult) -> int:
            page_match = re.search(r"\bpage\s+(\d+)\b", match.location, re.I)
            return int(page_match.group(1)) if page_match else 999

        passages = [clean_evidence_text(match.text) for match in sorted(senate_matches, key=page_number)]
        return clean_generated_answer(" ".join(passage for passage in passages if passage))

    def _history_people_answer_from_matches(
        self, question: str, matches: list[SearchResult]
    ) -> str:
        """Extract founder/rector names from retrieved SIMAD HISTORY chunks.

        Combines chunks already in ``matches`` with a targeted fetch of ONLY
        the SIMAD HISTORY.pdf documents from ChromaDB, so we never miss a
        founder name when the semantic search happens not to rank every chunk
        in the top-N results.
        """
        history_parts = [
            clean_evidence_text(match.text)
            for match in matches
            if Path(match.source).name.endswith("SIMAD HISTORY.pdf")
        ]
        # Targeted fetch: only SIMAD HISTORY.pdf chunks, not the whole collection.
        try:
            records = self.collection.get(include=["documents", "metadatas"])
            history_parts.extend(
                clean_evidence_text(text)
                for text, metadata in zip(records["documents"], records["metadatas"])
                if Path(str(metadata.get("source", ""))).name == "SIMAD HISTORY.pdf"
            )
        except Exception:
            pass
        history_text = " ".join(history_parts)
        if not history_text:
            return ""

        lowered = question.lower()
        if re.search(
            r"\b(?:sponsor|sponsored|sponsoring|sponsors|who sponsored)\b",
            lowered,
        ):
            return "SIMAD University was founded and sponsored by Direct Aid-Kuwait."

        if re.search(
            r"\bwhen\b.*\b(?:become|became|upgrade|upgraded)\b.*\buniversity\b"
            r"|\bwhen\b.*\buniversity\b.*\b(?:become|became|upgrade|upgraded)\b",
            lowered,
        ):
            return "SIMAD University was established as a higher education institute in 1999 and became a full-fledged university in 2011 (academic year 2010-2011)."

        # Previous rector patterns — must be checked BEFORE the founders/"who led" branch
        # so that "who led SIMAD before the current rector" returns previous rectors, not founders.
        if re.search(
            r"\b(?:former|previous|before|preceding)\s+rectors?\b"
            r"|\brectors?\b.*\b(?:former|previous|before)\b"
            r"|\bwho\s+(?:was|were)\s+(?:the\s+)?rector\s+before\b"
            r"|\bwho\s+(?:was|were|is)\s+(?:(?:the|a)\s+)?(?:rector\s+)?before\b"
            r"|\bwho\s+led\b.*\b(?:before|prior|previous)\b"
            r"|\b(?:before|prior to)\s+(?:the\s+)?current\s+rector\b",
            lowered,
        ):
            names = []
            if re.search(r"\bAbdirahman Mohamed Hussein Odowa\b", history_text, re.I):
                names.append("Abdirahman Mohamed Hussein Odowa")
            elif re.search(r"\bAbdirahman Mohamed Husien Odawa\b", history_text, re.I):
                names.append("Abdirahman Mohamed Hussein Odowa")
            if re.search(r"\bDahir Hassan Arab\b", history_text, re.I):
                names.append("Dahir Hassan Arab")
            elif re.search(r"\bDahir Hassan Abdi\b", history_text, re.I):
                names.append("Dahir Hassan Abdi")
            if names:
                unique_names = list(dict.fromkeys(names))
                return "\n".join(
                    ["SIMAD history records mention these previous rectors:"]
                    + [f"- {name}" for name in unique_names]
                )

        if re.search(
            r"\b(founders?|founding fathers?|who find|who found|who founded|who led|who started|who created)\b",
            lowered,
        ):
            founders = []
            if re.search(r"\bHassan Sheikh Mohammou?d\b", history_text, re.I):
                founders.append("Hassan Sheikh Mohammoud")
            if re.search(r"\bFarah Sheikh Abdikadir\b", history_text, re.I):
                founders.append("Farah Sheikh Abdikadir")
            if re.search(r"\bMohamed Hussein Dhobale\b", history_text, re.I):
                founders.append("Mohamed Hussein Dhobale")
            if founders:
                return "\n".join(
                    ["SIMAD history records identify these founding fathers:"]
                    + [f"- {name}" for name in founders]
                )
        return ""

    def _local_grounded_answer(
        self,
        question: str,
        matches: list[SearchResult],
        database_context: str = "",
        intent: str = "simad_factual",
    ) -> str:
        if database_context and (
            intent == "student_guidance"
            or self._is_academic_database_context(database_context)
            or PROGRAM_QUERY_PATTERN.search(question)
            or COURSE_QUERY_PATTERN.search(question)
            or re.search(r"\b(?:tuition|fees?|cost|price)\b", question, re.I)
        ):
            return self._answer_from_database_context(question, database_context)

        topic = self._semantic_topic(question)
        if is_general_overview_question(question):
            overview_answer = self._overview_answer_from_matches(matches)
            if overview_answer:
                return overview_answer

        if topic == "admissions" or intent == "admission_registration":
            admission_answer = self._admission_answer_from_matches(question, matches)
            if admission_answer:
                return admission_answer

        if topic == "scholarships":
            scholarship_answer = self._scholarship_answer_from_matches(question, matches)
            if scholarship_answer:
                return scholarship_answer

        if topic == "grading":
            grading_answer = self._grading_answer_from_matches(matches)
            if grading_answer:
                if re.search(r"\b(my|mine|i|me)\b.*\bgpa\b|\bgpa\b.*\b(my|mine|i|me)\b", question.lower()):
                    return (
                        "I do not have access to your personal student records or grades to know your GPA. "
                        "However, here is the official SIMAD University grading scale to help you check your grades:\n"
                        f"{grading_answer}"
                    )
                return grading_answer

        if topic == "history":
            history_people_answer = self._history_people_answer_from_matches(question, matches)
            if history_people_answer:
                return history_people_answer

        if topic in {"leadership", "governance"} and matches:
            if re.search(r"\bcurrent rector\b|\bwho is (?:the )?rector\b", question, re.I):
                for match in matches:
                    for passage in evidence_passages(match.text):
                        for sentence in split_sentences(passage):
                            if re.search(r"\bcurrent Rector\b", sentence):
                                return clean_generated_answer(sentence)
            senate_answer = self._senate_answer_from_matches(matches)
            if senate_answer:
                return senate_answer
            return self._retrieval_only_answer(question, matches)

        if not hasattr(self, "model"):
            return self._retrieval_only_answer(
                question, matches, database_context=database_context
            )

        qa_candidates = []
        for match_index, match in enumerate(matches):
            for qa_question, qa_answer in re.findall(
                r"\bQ:\s*(.+?)\s+A:\s*(.*?)(?=\s+Q:|\Z)",
                clean_evidence_text(match.text),
                flags=re.I | re.S,
            ):
                qa_score = coverage(terms(question), qa_question)
                if qa_score >= 0.50:
                    qa_candidates.append((qa_score, -match_index, clean_generated_answer(qa_answer)))
        if qa_candidates:
            return max(qa_candidates, key=lambda item: (item[0], item[1]))[2]

        candidates = []
        if database_context:
            structured_answer = self._answer_from_database_context(question, database_context)
            if structured_answer:
                for line in structured_answer.splitlines():
                    if line.strip():
                        candidates.append((-1, line.strip()))
        for match_index, match in enumerate(matches):
            for passage in evidence_passages(match.text):
                candidates.append((match_index, passage))
        if not candidates:
            return NOT_FOUND_MESSAGE

        query_embedding = self.model.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )[0]
        passage_embeddings = self.model.encode(
            [passage for _, passage in candidates],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_terms = terms(question)
        ranked = []
        for (match_index, passage), passage_embedding in zip(candidates, passage_embeddings):
            semantic_score = float(passage_embedding @ query_embedding)
            lexical_score = coverage(query_terms, passage)
            source_bonus = max(0.0, 0.10 - (0.02 * match_index))
            length_penalty = max(0.0, (len(passage) - 420) / 3000)
            short_fragment_penalty = 0.30 if len(passage) < 80 else 0.0
            noise_penalty = 0.0
            if not re.search(r"\b(contact|phone|email|bank|account)\b", question, re.I):
                noise_penalty += 0.12 * bool(
                    re.search(r"\b(?:bank accounts?|email|web:|\+\d{3}|contact us)\b", passage, re.I)
                )
            score = (
                semantic_score
                + (0.30 * lexical_score)
                + source_bonus
                - length_penalty
                - short_fragment_penalty
                - noise_penalty
            )
            ranked.append((score, semantic_score, passage))
        ranked.sort(reverse=True, key=lambda item: item[0])

        selected = []
        selected_terms = []
        broad_request = bool(
            re.search(
                r"\b(tell me about|information|describe|what are|list|services|"
                r"requirements|scholarships|programs|activities|founders?|founded)\b",
                question,
                re.I,
            )
        )
        maximum = 3 if broad_request else 2
        for score, semantic_score, passage in ranked:
            if semantic_score < 0.30 and score < 0.45:
                continue
            passage_terms = terms(passage)
            if any(
                passage.lower() in existing_text.lower()
                or existing_text.lower() in passage.lower()
                or len(passage_terms & existing_terms)
                / max(1, len(passage_terms | existing_terms))
                > 0.55
                for existing_text, existing_terms in zip(selected, selected_terms)
            ):
                continue
            selected.append(passage)
            selected_terms.append(passage_terms)
            if len(selected) >= maximum:
                break
        if not selected:
            return NOT_FOUND_MESSAGE

        if re.search(r"\brequirements?\b", question, re.I):
            section = next(
                (
                    passage
                    for passage in selected
                    if re.search(r"\bADMISSION REQUIREMENTS\b", passage, re.I)
                ),
                "",
            )
            if section:
                return clean_generated_answer(section)

        answer = clean_generated_answer(" ".join(selected))
        return answer[:1200].rsplit(" ", 1)[0] if len(answer) > 1200 else answer

    def _local_document_answer(
        self, question: str, matches: list[SearchResult]
    ) -> str | None:
        """Legacy compatibility shim; factual answers now go through retrieval."""
        return None

    @staticmethod
    def _is_academic_database_context(database_context: str) -> bool:
        return database_context.startswith(
            (
                "Verified academic availability",
                "Verified academic comparison",
                "Verified faculty:",
                "Verified faculties list:",
                "Verified undergraduate program:",
                "Verified course records",
                "Verified matching course records",
                "Verified program durations",
                "Verified tuition records",
                "Verified SIMAD administration records",
                "Use only the exact facts below",
            )
        )

    @staticmethod
    def _database_context_is_self_sufficient(
        question: str, intent: str, database_context: str
    ) -> bool:
        if not database_context:
            return False
        if intent == "student_guidance":
            return True
        if SimadChatbot._is_academic_database_context(database_context):
            return True
        return database_context.startswith("Verified tuition records")

    @staticmethod
    def _program_list(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    @staticmethod
    def _format_program_bullets(programs: str) -> list[str]:
        return [f"- {program}" for program in SimadChatbot._program_list(programs)]

    @staticmethod
    def _parse_faculty_program_lines(lines: list[str]) -> list[tuple[str, str]]:
        parsed = []
        for line in lines:
            match = re.match(r"^-\s+Faculty of\s+(.+?):\s*(.+)$", line, re.I)
            if match:
                parsed.append((match.group(1).strip(), match.group(2).strip()))
        return parsed

    @staticmethod
    def _format_comparison_context(lines: list[str]) -> str:
        faculties = SimadChatbot._parse_faculty_program_lines(lines)
        if not faculties:
            return ""
        output = [
            "SIMAD records compare them by verified program lists:"
        ]
        for index, (faculty, programs) in enumerate(faculties, start=1):
            output.append(f"{index}. Faculty of {faculty}: {programs}")
        output.append(
            "No additional comparison details were found in the SIMAD documents."
        )
        return "\n".join(output)

    @staticmethod
    def _format_guidance_context(question: str, lines: list[str]) -> str:
        faculties = SimadChatbot._parse_faculty_program_lines(lines)
        if not faculties:
            return ""

        focused_interest = len(faculties) == 1
        if focused_interest:
            faculty, programs = faculties[0]
            output = [
                f"Based on SIMAD's verified program records, the closest match is Faculty of {faculty}:"
            ]
            output.extend(SimadChatbot._format_program_bullets(programs))
            output.append(
                "Choose the option that best matches your interests, strengths, and career goal."
            )
            output.append(
                "Tell me which area interests you most, and I can help you compare the matching SIMAD programs."
            )
            return "\n".join(output)

        output = [
            "Based on SIMAD's verified program records, you can start by choosing the faculty that matches your interests:"
        ]
        for faculty, programs in faculties:
            output.append(f"- Faculty of {faculty}: {programs}")
        output.append(
            "Tell me the field you prefer, such as computing, engineering, business, medicine, law, or social sciences."
        )
        return "\n".join(output)

    @staticmethod
    def _format_course_context(lines: list[str]) -> str:
        if not lines:
            return ""
        heading = lines[0]
        match = re.match(
            r"^Verified course records for\s+(.+?)(?:\s+semester\s+(\d+))?:$",
            heading,
            re.I,
        )
        if match:
            program = match.group(1).strip()
            semester = match.group(2)
            if semester:
                output = [f"The courses in {program} Semester {semester} are:"]
            else:
                output = [f"The courses in {program} are:"]
        else:
            output = ["The verified courses are:"]
        output.extend(line for line in lines[1:] if line.startswith("- "))
        return "\n".join(output)

    @staticmethod
    def _format_named_course_context(lines: list[str]) -> str:
        output = ["Matching course record:"]
        for line in lines[1:]:
            if not line.startswith("- "):
                continue
            content = line[2:].strip()
            fields = [part.strip() for part in content.split(";") if part.strip()]
            if not fields:
                continue
            title = fields[0]
            details = ", ".join(fields[1:])
            output.append(f"- {title}: {details}" if details else f"- {title}")
        return "\n".join(output) if len(output) > 1 else ""

    @staticmethod
    def _format_tuition_context(question: str, lines: list[str]) -> str:
        records = []
        for line in lines[1:]:
            match = re.match(
                r"^-\s+(.+?);\s+years:\s+(\d+);\s+tuition:\s+(\$[\d,]+);\s+"
                r"charges:\s+(\$[\d,]+);\s+total semester fee:\s+(\$[\d,]+)$",
                line,
                re.I,
            )
            if match:
                records.append(match.groups())
        if not records:
            return ""

        lowered = question.lower()
        highest_request = bool(
            re.search(r"\b(most|highest|max|maximum|expensive)\b", lowered)
        )
        if highest_request and len(records) == 1:
            program, _years, tuition, charges, total = records[0]
            return (
                "The highest total semester fee in SIMAD tuition records is:\n"
                f"1. {program}: {total} per semester "
                f"({tuition} tuition + {charges} charges)."
            )

        output = ["SIMAD tuition records:"]
        for program, _years, tuition, charges, total in records[:10]:
            output.append(
                f"- {program}: {total} per semester ({tuition} tuition + {charges} charges)."
            )
        if len(records) > 10:
            output.append("Ask for a specific faculty or program to narrow the list.")
        return "\n".join(output)

    @staticmethod
    def _format_availability_context(lines: list[str]) -> str:
        requested = ""
        matched_program = ""
        faculty = ""
        faculty_programs_line = ""
        for line in lines:
            requested_match = re.match(r"^Requested academic label:\s*(.+)$", line, re.I)
            matched_match = re.match(
                r"^Matched verified undergraduate program:\s*(.+)$", line, re.I
            )
            faculty_match = re.match(r"^Verified faculty:\s*(Faculty of .+)$", line, re.I)
            programs_match = re.match(
                r"^Verified undergraduate programs in that faculty:\s*(.+)$",
                line,
                re.I,
            )
            if requested_match:
                requested = requested_match.group(1).strip()
            elif matched_match:
                matched_program = matched_match.group(1).strip()
            elif faculty_match:
                faculty = faculty_match.group(1).strip()
            elif programs_match:
                faculty_programs_line = programs_match.group(1).strip()

        if matched_program and faculty:
            return f"The verified SIMAD data lists {matched_program} under {faculty}."

        if requested and faculty:
            output = [
                f"SIMAD academic records connect {requested} with {faculty}."
            ]
            if faculty_programs_line:
                output.append(f"The verified undergraduate programs listed under {faculty} are:")
                output.extend(SimadChatbot._format_program_bullets(faculty_programs_line))
            return "\n".join(output)
        return ""

    @staticmethod
    def _format_faculty_context(lines: list[str]) -> str:
        faculty = ""
        programs = ""
        for line in lines:
            faculty_match = re.match(r"^Verified faculty:\s*(Faculty of .+)$", line, re.I)
            programs_match = re.match(r"^Verified undergraduate programs:\s*(.+)$", line, re.I)
            if faculty_match:
                faculty = faculty_match.group(1).strip()
            elif programs_match:
                programs = programs_match.group(1).strip()
        if not faculty or not programs:
            return ""
        program_items = SimadChatbot._program_list(programs)
        label = "Undergraduate program:" if len(program_items) == 1 else "Undergraduate programs:"
        return "\n".join([faculty, label, *[f"- {program}" for program in program_items]])

    @staticmethod
    def _answer_from_database_context(question: str, database_context: str) -> str:
        cleaned_context = re.sub(
            r"^Use only the exact facts below; do not infer a description of the faculty\.\s*",
            "",
            database_context,
            flags=re.I,
        )
        cleaned_context = clean_generated_answer(cleaned_context)
        lines = [line.strip() for line in cleaned_context.splitlines() if line.strip()]
        if not lines:
            return ""

        first_line = lines[0].lower()
        if first_line.startswith("verified academic comparison"):
            return SimadChatbot._format_comparison_context(lines)
        if first_line.startswith("verified undergraduate programs matching") or first_line.startswith(
            "verified undergraduate faculties"
        ):
            return SimadChatbot._format_guidance_context(question, lines)
        if first_line.startswith("verified course records"):
            return SimadChatbot._format_course_context(lines)
        if first_line.startswith("verified matching course records"):
            return SimadChatbot._format_named_course_context(lines)
        if first_line.startswith("verified tuition records"):
            return SimadChatbot._format_tuition_context(question, lines)
        if first_line.startswith("verified simad administration records"):
            return "\n".join(lines[1:])
        if first_line.startswith("verified academic availability"):
            return SimadChatbot._format_availability_context(lines)
        if first_line.startswith("verified faculty:"):
            return SimadChatbot._format_faculty_context(lines)
        if first_line.startswith("verified faculties list:"):
            return "\n".join(lines[1:])
        return cleaned_context

    @staticmethod
    def _retrieval_only_answer(
        question: str, matches: list[SearchResult], database_context: str = ""
    ) -> str:
        """Compatibility helper used by tests and simple non-model callers."""
        if database_context:
            return SimadChatbot._answer_from_database_context(question, database_context)
        query_terms = terms(question)
        qa_candidates = []
        for match_index, match in enumerate(matches):
            pairs = re.findall(
                r"\bQ:\s*(.+?)\s+A:\s*(.*?)(?=\s+Q:|\Z)",
                match.text,
                flags=re.I | re.S,
            )
            for qa_question, qa_answer in pairs:
                answer = clean_generated_answer(qa_answer)
                if not answer:
                    continue
                score = coverage(query_terms, qa_question) if query_terms else 0.0
                if re.search(r"\bwhat is SIMAD University\b", qa_question, re.I):
                    score += 2.0
                qa_candidates.append((score, -match_index, answer))

        if qa_candidates:
            _, _, answer = max(qa_candidates, key=lambda item: (item[0], item[1]))
            return answer

        if matches:
            clean_excerpt = clean_generated_answer(clean_evidence_text(matches[0].text))
            looks_like_clean_prose = (
                len(clean_excerpt) <= 2500
                and "|" not in clean_excerpt
                and "Q:" not in clean_excerpt
                and clean_excerpt.count("/") <= 2
            )
            if looks_like_clean_prose:
                return (
                    clean_excerpt[:1200].rsplit(" ", 1)[0]
                    if len(clean_excerpt) > 1200
                    else clean_excerpt
                )

        return GENERATION_UNAVAILABLE_MESSAGE


def main() -> None:
    try:
        bot = SimadChatbot()
    except Exception as exc:
        raise SystemExit(
            f"Could not open the knowledge base: {exc}\nRun `python train_data.py` first."
        ) from exc

    history: list[dict[str, str]] = []
    print("SIMAD University Assistant")
    print("Ask a question, or type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        try:
            answer = bot.answer(question, history)
        except Exception as exc:
            answer = f"I could not generate an answer because of an API error: {exc}"

        print(f"\nAssistant: {answer}\n")
        history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])


if __name__ == "__main__":
    main()
