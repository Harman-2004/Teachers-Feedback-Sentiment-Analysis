"""
generate_dataset.py
-------------------
Generates feedback_dataset.csv with 500 unique, realistic teacher
feedback entries across 20+ teachers, multiple subjects, and
all sentiment categories.

Run:
    python data/generate_dataset.py
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(2024)

# ── Teachers & Subjects ──────────────────────────────────────────────────────
TEACHERS = [
    ("Dr. Aisha Rahman",       "Mathematics"),
    ("Prof. Ravi Shankar",     "Physics"),
    ("Ms. Priya Nair",         "Computer Science"),
    ("Mr. James Okafor",       "English Literature"),
    ("Dr. Emily Chen",         "Chemistry"),
    ("Prof. Samuel Adeyemi",   "Biology"),
    ("Ms. Lakshmi Iyer",       "History"),
    ("Mr. Carlos Mendez",      "Economics"),
    ("Dr. Fatima Al-Zahra",    "Data Science"),
    ("Prof. Wei Zhang",        "Statistics"),
    ("Ms. Anjali Desai",       "Psychology"),
    ("Mr. David Kimani",       "Geography"),
    ("Dr. Sophia Patel",       "Artificial Intelligence"),
    ("Prof. Omar Hassan",      "Philosophy"),
    ("Ms. Neha Gupta",         "Environmental Science"),
    ("Mr. Ethan Brooks",       "Business Studies"),
    ("Dr. Maria Fernandez",    "Sociology"),
    ("Prof. Arjun Mehta",      "Electrical Engineering"),
    ("Ms. Chloe Dubois",       "French Language"),
    ("Mr. Rohan Verma",        "Physical Education"),
    ("Dr. Yuki Tanaka",        "Robotics"),
    ("Prof. Linda Osei",       "Nursing & Healthcare"),
]

SEMESTERS = ["Spring 2023", "Fall 2023", "Spring 2024", "Fall 2024"]

# ── Date range ───────────────────────────────────────────────────────────────
START_DATE = date(2023, 1, 10)
END_DATE   = date(2024, 12, 20)
DATE_RANGE = (END_DATE - START_DATE).days

def random_date():
    return (START_DATE + timedelta(days=random.randint(0, DATE_RANGE))).strftime("%Y-%m-%d")

# ── Feedback pools per dimension ─────────────────────────────────────────────

# COMMUNICATION ──────────────────────────────────────────────────────────────
COMM_POSITIVE = [
    "Explains every topic with exceptional clarity, making even the most complex ideas accessible.",
    "Her communication style is outstanding — she breaks down difficult theories into simple, digestible parts.",
    "One of the most articulate teachers I have ever encountered; never leaves a concept half-explained.",
    "Speaks at a perfect pace — clear, confident, and never rushes through important material.",
    "Always uses well-structured language that makes the subject feel intuitive rather than intimidating.",
    "His verbal explanations are accompanied by excellent analogies that stick in your memory.",
    "The way she narrates case studies makes abstract concepts come alive in the classroom.",
    "Never uses jargon without explaining it first — a rare and appreciated quality in an educator.",
    "Even during difficult Q&A sessions, maintains composure and communicates answers clearly.",
    "Written feedback on essays is detailed and constructive — clear articulation extends beyond lectures.",
    "Uses a blend of visual aids and verbal narration that caters to all kinds of learners.",
    "Her tone is always encouraging, making it easy to follow along even when the content is tough.",
    "Every slide and every word is intentionally placed — communication is precise and efficient.",
    "Gives clear instructions before each assignment, eliminating confusion from the start.",
    "Masterfully simplifies dense academic papers into understandable summaries during class.",
]

COMM_NEGATIVE = [
    "Speaks far too fast and frequently skips steps mid-explanation, leaving students confused.",
    "The accent is at times hard to understand and no effort is made to slow down or repeat.",
    "Instructions for assignments are vague and lead to widespread misunderstandings every semester.",
    "Tends to mumble during lectures, making it nearly impossible to take accurate notes.",
    "Jumps between topics without transitions — lectures feel disjointed and hard to follow.",
    "Gives feedback on submissions in one-word comments that offer no direction for improvement.",
    "Rarely repeats key points, so if you miss something, you have no way to catch up.",
    "Email replies are written in a confusing manner; often raises more questions than it answers.",
    "The board work is illegible — notes are cramped and the handwriting is difficult to read.",
    "Lacks clarity when explaining grading criteria, making it hard to know what is expected.",
    "Explanations are overly technical without building up from foundational concepts first.",
    "Often trails off mid-sentence and changes direction, which disrupts the flow of learning.",
    "Uses acronyms and shorthand without ever defining them — very frustrating for new students.",
    "Does not check for understanding — just moves on regardless of whether students are following.",
]

COMM_NEUTRAL = [
    "Communication is generally fine, though sometimes explanations could be more structured.",
    "Adequate communicator — covers the material but could make lectures more engaging with better delivery.",
    "The lectures are informative but can feel a bit dry due to a monotonous speaking style.",
    "Explains concepts at a satisfactory level, though more real-world examples would help.",
    "Delivers content accurately but without much flair — the pace is steady if unexciting.",
    "Reasonable clarity in lectures; some ambiguity in written feedback but nothing insurmountable.",
]

# ENGAGEMENT ─────────────────────────────────────────────────────────────────
ENG_POSITIVE = [
    "Creates an incredibly interactive learning environment where every student feels heard.",
    "Uses gamified quizzes and group challenges that make learning genuinely fun and competitive.",
    "Poses thought-provoking questions mid-lecture that ignite genuine curiosity among students.",
    "Student participation is actively encouraged and rewarded — the class feels like a community.",
    "Brings in guest speakers and industry professionals to keep the content real and relevant.",
    "The energy in the classroom is infectious — you cannot help but get involved.",
    "Designs collaborative projects that build both subject mastery and interpersonal skills.",
    "Always relates lessons to current events, which makes abstract theories feel immediately relevant.",
    "Makes students feel like co-creators of knowledge rather than passive receivers of information.",
    "The use of polls, debates, and role-plays keeps every lecture dynamic and participatory.",
    "Remembers individual student names and learning styles — a personal touch that boosts engagement.",
    "Encourages critical thinking by consistently challenging students to question assumptions.",
    "Live coding demonstrations paired with immediate student practice sessions are highly effective.",
    "Classroom discussions feel genuinely democratic — every opinion is valued and explored.",
    "The enthusiasm for the subject is contagious; I started loving a topic I once dreaded.",
]

ENG_NEGATIVE = [
    "Classes are purely lecture-based with zero opportunities for student interaction or questions.",
    "Rarely acknowledges student questions and often dismisses them without adequate explanation.",
    "The teaching style feels like reading directly from the textbook — completely disengaging.",
    "No group activities, no discussions, no real connection with students — deeply monotonous.",
    "Students who try to contribute are often interrupted or hurried along, discouraging participation.",
    "Relies entirely on PowerPoint slides with no supplementary engagement — very passive experience.",
    "Does not seem interested in teaching — the lack of enthusiasm is demoralizing for the class.",
    "Never poses questions to students, making it easy to zone out for the entire lecture.",
    "Interaction is limited to 'any questions?' at the very end when there is no time left.",
    "The classroom environment feels tense and unwelcoming — students are afraid to speak up.",
    "Group projects are assigned but never meaningfully supervised or supported.",
    "The same presentation format, week after week, with no variation — painfully dull.",
]

ENG_NEUTRAL = [
    "The engagement level is average — some days are interactive, but most follow a standard lecture format.",
    "Occasionally uses case studies but could incorporate more active learning strategies consistently.",
    "Student interaction happens but feels forced rather than organic within the lesson structure.",
    "The class is neither overly boring nor particularly exciting — solidly mediocre engagement.",
    "Uses slides well but misses opportunities to spark discussion around the presented material.",
    "Engagement improves in smaller tutorial sessions but is lacking in the main lectures.",
]

# RESPONSIVENESS ─────────────────────────────────────────────────────────────
RESP_POSITIVE = [
    "Responds to emails within hours — remarkable availability for such a senior faculty member.",
    "Holds extra office hours before every major exam, showing genuine care for student success.",
    "Always follows up on unanswered questions from previous lectures at the start of the next class.",
    "Created a class WhatsApp group to answer real-time doubts — incredibly supportive initiative.",
    "When a student struggles, proactively reaches out rather than waiting for the student to ask.",
    "Never dismisses a query — even seemingly simple questions receive a thorough, patient response.",
    "Sets up one-on-one review sessions for students who fall behind, at no additional cost.",
    "Feedback on draft submissions is returned within 48 hours with detailed, actionable comments.",
    "Always available after class and never seems in a rush to leave — truly student-centered.",
    "Maintains an open-door policy that students actually use — approachability is a real strength.",
    "Quickly acknowledged a mistake in the lecture notes and corrected them with a detailed email.",
    "Responds thoughtfully to even the most off-topic questions, redirecting with patience.",
    "Has never missed a scheduled office hour — reliability and responsiveness go hand in hand.",
    "When asked a complex question in class, promises a full answer by email and always delivers.",
]

RESP_NEGATIVE = [
    "Takes over two weeks to reply to emails, and the responses are often incomplete.",
    "Office hours are scheduled but rarely actually held — very disappointing for students who need help.",
    "Dismisses students who ask for extra support, citing time constraints that seem exaggerated.",
    "Graded assignments are returned weeks after the deadline with no explanation for the delay.",
    "Never addresses doubt-clearing requests — students are left to figure out everything on their own.",
    "Unavailable outside of class time and discourages emails unless they are 'truly urgent'.",
    "Feedback on assignments is generic, copy-pasted, and returns no useful guidance.",
    "Students who visit during office hours are frequently told to 'check the textbook' without help.",
    "Ignores follow-up questions posted on the course forum — makes students feel unsupported.",
    "Announced a change in exam format via a brief note buried in slides — zero proactive communication.",
]

RESP_NEUTRAL = [
    "Response time is acceptable for emails but could be faster given the volume of student queries.",
    "Office hours are maintained consistently, though the depth of support provided could improve.",
    "Generally available but sometimes the guidance offered is too brief to be truly helpful.",
    "Responds to most messages but occasionally misses queries sent close to deadlines.",
    "Satisfactory responsiveness overall — not exceptional but not problematic either.",
]

# ASSIGNMENT QUALITY ─────────────────────────────────────────────────────────
ASSIGN_POSITIVE = [
    "Assignments are thoughtfully designed to bridge theory and real-world application brilliantly.",
    "Each task builds on the last in a perfectly scaffolded progression toward mastery.",
    "The case study assignments are genuinely challenging and deeply relevant to the industry.",
    "Problem sets are challenging yet fair — they stretch thinking without being unreasonable.",
    "Projects require independent research that significantly deepens understanding of the subject.",
    "Assignment rubrics are crystal clear, so students know exactly what is expected of them.",
    "The creative freedom given in project briefs encourages innovative thinking and ownership.",
    "Lab assignments perfectly complement theory lectures — a seamlessly integrated curriculum.",
    "Grading is consistent and transparent — students never feel that marks are arbitrary.",
    "The assignments push beyond memorization into genuine critical analysis and synthesis.",
    "Provides model answers after submission that serve as excellent study resources.",
    "Group assignments are structured to ensure fair contribution and accountability from all members.",
    "The practical coding exercises assigned weekly are the best preparation for real-world work.",
    "Makes the purpose of every assignment explicit — students understand why each task matters.",
]

ASSIGN_NEGATIVE = [
    "Assignments are poorly worded and often require clarification that is slow to come.",
    "The workload is excessive relative to the credit hours allocated to this course.",
    "Grading is inconsistent — the same quality of work receives vastly different marks week to week.",
    "Tasks feel disconnected from what is taught in lectures, making them very frustrating to complete.",
    "Deadlines are unrealistically tight and show no consideration for students' other commitments.",
    "No model answers or solutions are provided after submission, leaving students to guess.",
    "Feedback on assignments is superficial — just a score with a single line of generic comment.",
    "Many assignments are reused year after year without update — students access past solutions easily.",
    "The marking scheme is never shared in advance, creating unnecessary anxiety among students.",
    "Group projects lack structure, leading to unequal distribution of work and unfair grading.",
    "Written assignments are assigned but the grading focuses almost entirely on formatting, not content.",
]

ASSIGN_NEUTRAL = [
    "Assignment quality is acceptable — they test course material adequately if not innovatively.",
    "Workload is manageable and coverage is reasonable, though depth could be improved.",
    "Submissions are marked fairly but feedback is minimal — you know your score but not much else.",
    "The tasks are standard for the subject and neither particularly inspiring nor discouraging.",
    "Grading turnaround is reasonable; rubrics exist but could be more detailed.",
]

# SUBJECT KNOWLEDGE ──────────────────────────────────────────────────────────
KNOW_POSITIVE = [
    "An absolute authority on the subject — references cutting-edge research with ease in every lecture.",
    "Possesses encyclopaedic knowledge and can link any student question back to broader theoretical frameworks.",
    "Demonstrates mastery by going well beyond the syllabus to give broader intellectual context.",
    "Cites primary research papers during lectures, showing genuine depth of scholarship.",
    "Can explain the same concept ten different ways until every student in the room understands.",
    "Keeps the course content updated with the latest industry developments — always current.",
    "Brings personal research expertise into the classroom, making lectures uniquely insightful.",
    "Handles even the most unexpected or advanced questions with confidence and precision.",
    "The breadth and depth of knowledge is remarkable — genuinely one of the most learned educators I have met.",
    "Connects the current topic to related fields seamlessly, enriching the overall learning experience.",
    "Explains historical context of theories, giving students a richer understanding of the subject.",
    "Every claim in lectures is backed by evidence and peer-reviewed sources — intellectually rigorous.",
    "Can discuss nuanced debates within the field, exposing students to the complexity of real scholarship.",
    "The passion for the subject translates directly into deeply informed, inspiring lectures.",
]

KNOW_NEGATIVE = [
    "Relies too heavily on notes and struggles to answer questions that deviate from the slide deck.",
    "Core concepts are sometimes explained incorrectly, raising serious concerns about subject mastery.",
    "The course content feels dated — no references to modern developments or current research.",
    "Cannot confidently field student questions that go slightly beyond the textbook material.",
    "Made multiple factual errors in a single lecture with no acknowledgment or correction.",
    "The practical examples used are poorly chosen and sometimes technically inaccurate.",
    "Appears to read slides for the first time in class — little evidence of deep preparation.",
    "Unable to explain foundational theories beyond the surface level when probed by students.",
    "The course seems to be delivered from memory of a single outdated textbook from years ago.",
]

KNOW_NEUTRAL = [
    "Solid knowledge of core syllabus topics but seems less confident with advanced or edge-case material.",
    "Subject expertise is adequate for the level of the course though not particularly inspiring.",
    "Good foundational knowledge; the integration of recent research could be improved.",
    "Knows the topic well enough to teach it effectively but rarely goes beyond prescribed material.",
    "Reliable knowledge base — delivers accurate content consistently, if without great depth.",
]

# ── Build template system ────────────────────────────────────────────────────
# Each entry is (pool, sentiment_weight) — weight guides how many we pull
DIMENSION_POOLS = [
    (COMM_POSITIVE,   "Positive"),
    (COMM_NEGATIVE,   "Negative"),
    (COMM_NEUTRAL,    "Neutral"),
    (ENG_POSITIVE,    "Positive"),
    (ENG_NEGATIVE,    "Negative"),
    (ENG_NEUTRAL,     "Neutral"),
    (RESP_POSITIVE,   "Positive"),
    (RESP_NEGATIVE,   "Negative"),
    (RESP_NEUTRAL,    "Neutral"),
    (ASSIGN_POSITIVE, "Positive"),
    (ASSIGN_NEGATIVE, "Negative"),
    (ASSIGN_NEUTRAL,  "Neutral"),
    (KNOW_POSITIVE,   "Positive"),
    (KNOW_NEGATIVE,   "Negative"),
    (KNOW_NEUTRAL,    "Neutral"),
]

# ── Sentence combiners for mixed feedback ────────────────────────────────────
CONJUNCTIONS_MIXED = [
    "{pos} However, {neg}",
    "{pos} On the downside, {neg}",
    "{neg} That said, {pos}",
    "{neg} Despite this, {pos}",
    "{pos} Nevertheless, {neg}",
    "{neg} To give credit where it is due, {pos}",
    "{pos} One area needing improvement: {neg}",
    "{pos} With that in mind, {neg}",
]

def build_mixed(pos_pool, neg_pool):
    pattern = random.choice(CONJUNCTIONS_MIXED)
    pos = random.choice(pos_pool).rstrip(".")
    neg = random.choice(neg_pool).rstrip(".")
    return pattern.format(pos=pos, neg=neg.lower()) + "."


def semester_for_date(d: str) -> str:
    dt = date.fromisoformat(d)
    if dt.month <= 6:
        return f"Spring {dt.year}"
    return f"Fall {dt.year}"


# ── Generation ───────────────────────────────────────────────────────────────
ALL_POS = COMM_POSITIVE + ENG_POSITIVE + RESP_POSITIVE + ASSIGN_POSITIVE + KNOW_POSITIVE
ALL_NEG = COMM_NEGATIVE + ENG_NEGATIVE + RESP_NEGATIVE + ASSIGN_NEGATIVE + KNOW_NEGATIVE
ALL_NEU = COMM_NEUTRAL  + ENG_NEUTRAL  + RESP_NEUTRAL  + ASSIGN_NEUTRAL  + KNOW_NEUTRAL

TARGET = 500
rows = []
used_texts = set()

def add_row(teacher, subject, text, d):
    if text in used_texts:
        return False
    used_texts.add(text)
    sem = semester_for_date(d)
    rows.append({
        "teacher_name": teacher,
        "subject":      subject,
        "feedback_text": text,
        "semester":     sem,
        "date":         d,
    })
    return True

# Distribute across teachers evenly (~22-23 per teacher)
teacher_list = TEACHERS * 25        # plenty of repeats to draw from
random.shuffle(teacher_list)

# Sentiment distribution: 45% pos, 30% neg, 15% neutral, 10% mixed
sentiment_weights = (
    ["pos"] * 45 + ["neg"] * 30 + ["neu"] * 15 + ["mix"] * 10
)

attempts = 0
while len(rows) < TARGET and attempts < TARGET * 10:
    attempts += 1
    teacher, subject = random.choice(TEACHERS)
    d = random_date()
    sentiment = random.choice(sentiment_weights)

    if sentiment == "pos":
        pool = ALL_POS
        text = random.choice(pool)
    elif sentiment == "neg":
        pool = ALL_NEG
        text = random.choice(pool)
    elif sentiment == "neu":
        pool = ALL_NEU
        text = random.choice(pool)
    else:  # mixed
        text = build_mixed(ALL_POS, ALL_NEG)

    # Slight teacher-name personalisation
    text = text.replace("She ", f"{'She' if 'Ms.' in teacher or 'Dr.' in teacher and random.random() > 0.5 else 'He'} ")
    text = text.replace("His ", f"{'Her' if 'Ms.' in teacher else 'His'} ")
    text = text.replace("Her ", f"{'Her' if 'Ms.' in teacher else 'His'} ")

    add_row(teacher, subject, text, d)

# If we still need more, generate compound sentences
while len(rows) < TARGET:
    teacher, subject = random.choice(TEACHERS)
    d = random_date()
    p1 = random.choice(ALL_POS).rstrip(".")
    p2 = random.choice(ALL_POS).rstrip(".")
    if p1 != p2:
        text = f"{p1}. Additionally, {p2.lower()}."
        add_row(teacher, subject, text, d)

# Sort by date
rows.sort(key=lambda r: r["date"])

# ── Write CSV ────────────────────────────────────────────────────────────────
OUT = Path(__file__).parent / "feedback_dataset.csv"
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["teacher_name","subject","feedback_text","semester","date"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅  Written {len(rows)} rows → {OUT}")
