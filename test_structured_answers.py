from unittest import TestCase
from unittest.mock import Mock

from academic_data import (
    course_program_name,
    find_courses,
    find_program_parent,
    find_programs,
    normalize_academic_query,
    semester_number,
    tuition_records,
)
from chatbot_clean import (
    NOT_FOUND_MESSAGE,
    SimadChatbot,
    allowed_sources,
    canonical_question,
    clean_generated_answer,
    clean_previous_answer,
    focused_answer,
)


class StructuredAnswerTests(TestCase):
    def setUp(self):
        self.bot = SimadChatbot.__new__(SimadChatbot)
        self.bot.generator = None
        self.bot.hf_model = "test-model"
        self.bot.last_answer_mode = "ready"
        self.bot.search = Mock(return_value=[])

    def test_program_list_uses_retrieval_context_values(self):
        answer = self.bot.answer(
            "What programs are inside the Faculty of Computing?"
            , []
        )
        self.bot.search.assert_not_called()
        self.assertIn("Computer Science", answer)
        self.assertIn("Information Technology", answer)
        self.assertIn("Graphics and Multimedia", answer)
        self.assertNotIn("Python Programming", answer)
        self.assertNotIn("Software Development", answer)
        self.assertNotEqual(self.bot.last_answer_mode, "structured_database")

    def test_program_availability_uses_academic_data_not_semantic_topic_noise(self):
        for question, expected in {
            "Does SIMAD offer International Relations?": "Political Science & International Relations",
            "Does SIMAD offer Civil Engineering?": "Civil Engineering",
            "Does SIMAD offer Electrical Engineering?": "Electrical Engineering",
        }.items():
            answer = self.bot.answer(question, [])
            self.assertIn(expected, answer)
            self.assertNotIn("Senate Secretary", answer)
            self.assertEqual(self.bot.last_semantic_topic, "academics")

    def test_faculty_comparison_uses_verified_program_lists(self):
        answer = self.bot.answer(
            "What is the difference between Engineering and Computing?", []
        )
        self.assertIn("Faculty of Computing: Computer Science", answer)
        self.assertIn("Faculty of Engineering: Telecommunication", answer)
        self.assertIn("verified program lists", answer)
        self.assertNotIn("Senate", answer)
        self.assertNotIn("Verified academic comparison", answer)

    def test_faculty_comparison_filters_to_requested_faculties(self):
        answer = self.bot.answer(
            "what is difference between faculty of computing and Management Sciences",
            [],
        )
        self.assertIn("Faculty of Computing: Computer Science", answer)
        self.assertIn("Faculty of Management Sciences: Business Administration", answer)
        self.assertNotIn("Faculty of Medicine", answer)
        self.assertNotIn("Faculty of Social Sciences", answer)

    def test_comparison_followup_keeps_previous_two_faculties(self):
        history = [
            {
                "role": "user",
                "content": "what is difference between faculty of computing and Management Sciences",
                "kind": "factual_question",
            },
            {
                "role": "assistant",
                "content": (
                    "Faculty of Computing and Faculty of Management Sciences were compared."
                ),
                "kind": "factual_answer",
            },
        ]

        answer = self.bot.answer("what about computing compare both of them", history)

        self.assertIn("Faculty of Computing: Computer Science", answer)
        self.assertIn("Faculty of Management Sciences: Business Administration", answer)
        self.assertNotIn("Faculty of Medicine", answer)

    def test_economics_typo_still_finds_verified_programs(self):
        result = find_programs("programs inside faculty of econimics")
        self.assertEqual(result[0], "Economics")
        self.assertIn("Biostatistics", result[1])

    def test_generalized_simad_spelling_correction(self):
        self.assertEqual(
            normalize_academic_query("faclty of econmics progrms"),
            "faculty of economics programs",
        )
        self.assertEqual(
            normalize_academic_query("computr scince semster thre courses"),
            "computer science semester three courses",
        )
        self.assertEqual(
            allowed_sources("admisson requirments"),
            ("ADMISSION BROCHURE.pdf",),
        )
        self.assertEqual(allowed_sources("scholrships"), ("Scholarships.pdf",))
        self.assertEqual(
            allowed_sources("admissom requirments"),
            ("ADMISSION BROCHURE.pdf",),
        )
        self.assertEqual(allowed_sources("tell me about the libary"), ("CAMPUS SERVICES.pdf",))
        self.assertEqual(
            allowed_sources("disabilty support"),
            ("Disability Support Services (DSS).pdf",),
        )
        self.assertEqual(
            allowed_sources("SIMAD acredition"),
            ("Accreditation, Ranking, & Memberships.pdf",),
        )

    def test_broad_simad_information_question_is_canonicalized(self):
        self.assertEqual(
            canonical_question("give me a information about simad"),
            "What is SIMAD University?",
        )
        self.assertEqual(
            allowed_sources("give me a information about simad"),
            ("SIMAD UNIVERSITY GENERAL INFORMATION.pdf",),
        )

    def test_misspelled_course_query_uses_verified_curriculum(self):
        answer = self.bot.answer("computr scince semster thre courses", [])
        self.assertIn("Computer Science Semester 3", answer)
        self.assertIn("- Physics", answer)
        self.assertNotEqual(self.bot.last_answer_mode, "structured_database")

    def test_program_is_not_treated_as_faculty(self):
        self.assertEqual(
            find_program_parent("programs inside CS"),
            ("Computing", "Computer Science"),
        )
        context = self.bot._database_context("programs inside CS", "academic_programs")
        self.assertIn("Matched verified undergraduate program: Computer Science", context)
        self.assertIn("Verified faculty: Faculty of Computing", context)

    def test_course_program_aliases_are_understood(self):
        self.assertEqual(course_program_name("semester one courses for IT"), "Information Technology")
        self.assertEqual(course_program_name("semester one courses for CS"), "Computer Science")
        self.assertEqual(course_program_name("semester one courses for GM"), "Graphics and Multimedia")

    def test_course_names_only_and_program_semester_isolation(self):
        answer = self.bot.answer("courses in Computer Science semester 3", [])
        self.assertIn("The courses in Computer Science Semester 3 are:", answer)
        self.assertIn("- Physics", answer)
        self.assertNotIn("PHY1101", answer)
        self.assertNotIn("Credit hours:", answer)
        self.assertNotIn("Digital Illustration", answer)
        self.assertNotIn("Physiology", answer)

    def test_course_optional_fields(self):
        with_code = self.bot.answer("courses in Computer Science semester 3 with code", [])
        with_credits = self.bot.answer(
            "courses in Computer Science semester 3 with credit hours", []
        )
        full = self.bot.answer("full curriculum table for Computer Science semester 3", [])
        self.assertIn("Physics; code: PHY1101", with_code)
        self.assertNotIn("credit hours:", with_code.lower())
        self.assertIn("Physics; credit hours: 4", with_credits)
        self.assertNotIn("PHY1101", with_credits)
        self.assertIn("theory contact hours:", full.lower())
        self.assertIn("practical contact hours:", full.lower())

    def test_course_query_asks_for_missing_target(self):
        answer = self.bot.answer("courses in Faculty of Computing semester 1", [])
        self.assertEqual(answer, NOT_FOUND_MESSAGE)
        answer = self.bot.answer("Computer Science courses", [])
        self.assertIn("The courses in Computer Science are:", answer)
        self.assertIn("- Fundamentals of Computing", answer)

    def test_course_clarification_uses_previous_question(self):
        history = [
            {"role": "user", "content": "courses in Faculty of Computing semester one"},
            {
                "role": "assistant",
                "content": "Please specify the academic program, such as Computer Science.",
            },
        ]
        contextual = self.bot._contextual_question("computer science", history)
        answer = self.bot.answer(contextual, [])
        self.assertIn("Computer Science Semester 1", answer)
        self.assertIn("Fundamentals of Computing", answer)

    def test_how_about_faculty_followup_keeps_program_intent(self):
        history = [
            {"role": "user", "content": "programs inside faculty of economics"},
            {"role": "assistant", "content": "Economics programs"},
        ]
        contextual = self.bot._contextual_question("how about faculty of law", history)
        self.assertEqual(contextual, "What undergraduate programs are in faculty of law?")
        self.assertIn("Law", self.bot.answer(contextual, []))

    def test_i_mean_uses_corrected_current_topic(self):
        history = [
            {"role": "user", "content": "semester one courses for IT"},
            {"role": "assistant", "content": "IT courses"},
        ]
        self.assertEqual(
            self.bot._contextual_question("i mean Artificial Intelligence", history),
            "i mean Artificial Intelligence",
        )
        context = self.bot._database_context("i mean Artificial Intelligence", "follow_up")
        self.assertIn("Artificial Intelligence", context)
        self.assertIn("semester: 8", context)

    def test_how_about_it_course_request_uses_information_technology(self):
        history = [
            {"role": "user", "content": "Computer Science semester one courses"},
            {"role": "assistant", "content": "Computer Science courses"},
        ]
        contextual = self.bot._contextual_question(
            "how about semester one courses for IT", history
        )
        answer = self.bot.answer(contextual, [])
        self.assertIn("Information Technology Semester 1", answer)

    def test_only_tabular_answers_are_hard_structured(self):
        factual_questions = [
            "Who founded SIMAD?",
            "Who is the current Rector?",
            "What scholarships are available?",
            "Tell me about the library",
            "What are admission requirements?",
        ]
        for question in factual_questions:
            self.assertIsNone(self.bot._structured_answer(question))

    def test_database_context_supplies_verified_faculty_facts(self):
        context = self.bot._database_context("give me information about faculty of law", "simad_factual")
        self.assertIn("Faculty of Law", context)
        self.assertIn("undergraduate programs: Law", context)
        guidance = self.bot._database_context("help me choose a program", "student_guidance")
        self.assertIn("Faculty of Computing", guidance)
        self.assertIn("Faculty of Medicine & Health Sciences", guidance)

    def test_administration_context_answers_dean_questions(self):
        answer = self.bot.answer("Who is the dean of Faculty of Computing?", [])
        self.assertIn("Dr. Mohamed Hassan Ahmed", answer)
        self.assertIn("Dean, Faculty of Computing", answer)
        self.assertNotIn("School of Engineering", answer)

    def test_administration_context_filters_department_heads(self):
        answer = self.bot.answer("Who heads Computer Science department?", [])
        self.assertIn("Mr. Ubaid Mohamed Dahir", answer)
        self.assertIn("Head, Department of Computer Science", answer)
        self.assertNotIn("Graphics & Multimedia", answer)

    def test_technology_guidance_filters_to_computing_programs(self):
        guidance = self.bot._database_context(
            "can you give me an advice if i want to choose my career technology",
            "student_guidance",
        )
        self.assertIn("Faculty of Computing", guidance)
        self.assertIn("Computer Science", guidance)
        self.assertIn("Information Technology", guidance)
        self.assertNotIn("Faculty of Medicine", guidance)
        self.assertNotIn("Faculty of Law", guidance)

    def test_broad_faculty_overview_uses_only_database_programs(self):
        answer = self.bot.answer("give me information about faculty of law", [])
        self.assertIn("Faculty of Law", answer)
        self.assertIn("Undergraduate program:", answer)
        self.assertIn("- Law", answer)
        self.assertNotIn("justice", answer.lower())
        self.assertNotIn("training", answer.lower())

    def test_highest_tuition_is_database_derived(self):
        highest = max(tuition_records(), key=lambda item: item.total)
        self.assertEqual(highest.program, "Bachelor of Medicine and Surgery (MBBS)")
        answer = self.bot.answer("Give me the highest tuition fee", [])
        self.assertIn("$1,350", answer)
        self.assertNotEqual(self.bot.last_answer_mode, "structured_database")

    def test_named_program_tuition_is_database_derived(self):
        answer = self.bot.answer("What is the tuition fee for Computer Science?", [])
        self.assertIn("Bachelor of Computer Science", answer)
        self.assertIn("$401", answer)

    def test_non_faculty_program_questions_are_not_forced_into_undergraduate_handler(self):
        self.assertIsNone(self.bot._structured_answer("Tell me about exchange programs"))
        self.assertIsNone(self.bot._structured_answer("What postgraduate programs are available?"))

    def test_conversation_and_reactions_are_natural(self):
        cases = {
            "hello": "Hello",
            "can you help me": "I can help",
            "who are you": "SIMAD University assistant",
            "fuck off": "I understand",
            "skip it": "No problem",
        }
        for question, expected in cases.items():
            self.assertIn(expected, self.bot._conversation_answer(question, []))
        self.assertEqual(
            self.bot.answer("do you know messi", []),
            "I can only answer questions related to SIMAD University.",
        )

    def test_name_is_remembered_inside_chat(self):
        history = [
            {"role": "user", "content": "My name is Ahmed"},
            {"role": "assistant", "content": "Nice to meet you, Ahmed."},
        ]
        self.assertIn("Ahmed", self.bot._conversation_answer("who am I?", history))

    def test_whole_chat_memory_counts_all_questions(self):
        history = []
        for number in range(12):
            history.extend(
                [
                    {"role": "user", "content": f"Question {number}"},
                    {"role": "assistant", "content": f"Answer {number}"},
                ]
            )
        answer = self.bot._conversation_answer("Do you remember our conversation?", history)
        self.assertIn("12 questions", answer)
        self.assertIn("Question 11", answer)

    def test_previous_answer_transformations_use_chat_memory(self):
        history = [
            {"role": "user", "content": "What are the admission requirements?"},
            {
                "role": "assistant",
                "content": (
                    "The admission requirements are:\n"
                    "- Pass the National High School Exam.\n"
                    "- Bring the secondary school certificate.\n"
                    "- Bring four passport photos.\n"
                    "- Pass the admission interview.\n"
                    "- Pay the processing fee."
                ),
            },
        ]
        summary = self.bot._conversation_answer("summarize this", history)
        five_lines = self.bot._conversation_answer("make it 5 lines", history)
        explanation = self.bot._conversation_answer("explain again", history)
        question = self.bot._conversation_answer("for what question?", history)

        self.assertLessEqual(len(summary.splitlines()), 3)
        self.assertEqual(len(five_lines.splitlines()), 5)
        self.assertNotIn("Your question was:", explanation)
        self.assertNotIn("In simple terms:", explanation)
        self.assertNotEqual(explanation, history[-1]["content"])
        self.assertNotIn("\n- ", explanation)
        self.assertNotEqual(explanation, history[-1]["content"])
        self.assertIn("What are the admission requirements?", question)

    def test_topic_memory_skips_previous_transform_followups(self):
        history = [
            {"role": "user", "content": "Who founded SIMAD?"},
            {"role": "assistant", "content": "SIMAD has three founding fathers."},
            {"role": "user", "content": "summarize this"},
            {"role": "assistant", "content": "SIMAD has three founding fathers."},
        ]
        answer = self.bot._conversation_answer("for what question?", history)
        self.assertIn("Who founded SIMAD?", answer)
        five_lines = self.bot._conversation_answer("make it 5 lines", history)
        self.assertIn("SIMAD has three founding fathers.", five_lines)

    def test_chained_transform_uses_last_substantive_answer(self):
        history = [
            {"role": "user", "content": "What are the requirements?"},
            {
                "role": "assistant",
                "content": "Requirements:\n- One\n- Two\n- Three\n- Four\n- Five",
            },
            {"role": "user", "content": "summarize this"},
            {"role": "assistant", "content": "Requirements:\n- One\n- Two"},
        ]
        answer = self.bot._conversation_answer("make it 5 lines", history)
        self.assertEqual(len(answer.splitlines()), 5)
        self.assertIn("- Four", answer)

    def test_marked_factual_memory_ignores_small_talk_and_cleanup_commands(self):
        history = [
            {
                "role": "user",
                "content": "What are the admission requirements?",
                "kind": "factual_question",
            },
            {
                "role": "assistant",
                "content": "Requirements:\n1.\nPass exam\n2.\nBring certificate",
                "kind": "factual_answer",
            },
            {"role": "user", "content": "sure", "kind": "small_talk"},
            {"role": "assistant", "content": "Okay.", "kind": "small_talk"},
            {"role": "user", "content": "this is messy", "kind": "follow_up_answer"},
            {"role": "assistant", "content": "A shorter answer.", "kind": "follow_up_answer"},
        ]
        self.assertEqual(
            self.bot._previous_topic_question(history),
            "What are the admission requirements?",
        )
        self.assertIn("1. Pass exam", self.bot._previous_topic_answer(history))
        self.assertIn(
            "What are the admission requirements?",
            self.bot._conversation_answer("what was that about?", history),
        )

    def test_previous_answer_cleanup_removes_website_noise_and_fixes_lists(self):
        answer = clean_previous_answer(
            "Requirements:\n1.\nPass exam\n2.\nBring certificate\n"
            "Founded and Sponsored by Direct Aid-Kuwait HOME ABOUT US"
        )
        self.assertEqual(answer, "Requirements:\n1. Pass exam\n2. Bring certificate")

    def test_cleanup_followups_rewrite_previous_answer_without_retrieval(self):
        factual_answer = (
            "The undergraduate admission requirements are:\n"
            "- Should successfully pass the National High School Exam with a minimum "
            "overall average of 50 %\n"
            "- Should bring the original and a copy of secondary school certificate\n"
            "- Should successfully pass an admission interview and/or test"
        )
        history = [
            {
                "role": "user",
                "content": "What are the admission requirements?",
                "kind": "factual_question",
            },
            {
                "role": "assistant",
                "content": factual_answer,
                "kind": "factual_answer",
            },
        ]
        self.bot.search = Mock(side_effect=AssertionError("retrieval should not run"))

        for command in (
            "this is messy",
            "explain simply",
            "make it clean",
            "I don't understand",
        ):
            answer = self.bot.answer(command, history)
            self.assertNotEqual(answer, factual_answer)
            self.assertNotIn("In simple terms:", answer)
            self.assertNotIn("Your question was:", answer)
            self.assertIn("National High School Exam with at least 50%", answer)
            self.assertEqual(len(answer.splitlines()), len(set(answer.splitlines())))

    def test_followup_transformations_have_intention_specific_output(self):
        factual_answer = (
            "Student exchange details:\n"
            "- SIMAD University offers student exchange programs through Erasmus+\n"
            "- Students can study at partner universities for one or two semesters\n"
            "- The program supports international academic learning and cultural experience\n"
            "- Students remain part of SIMAD University during the exchange\n"
            "- The exchange uses partner universities"
        )
        history = [
            {
                "role": "user",
                "content": "Tell me about student exchange.",
                "kind": "factual_question",
            },
            {
                "role": "assistant",
                "content": factual_answer,
                "kind": "factual_answer",
            },
        ]
        self.bot.generator = None
        self.bot.search = Mock(side_effect=AssertionError("retrieval should not run"))

        clean = self.bot.answer("this is messy", history)
        simple = self.bot.answer("explain simply", history)
        short = self.bot.answer("make it short", history)
        expanded = self.bot.answer("too short", history)
        clarified = self.bot.answer("I don't understand", history)

        self.assertIn("\n- ", clean)
        self.assertNotIn("\n- ", simple)
        self.assertLessEqual(len(short.splitlines()), 3)
        self.assertGreater(len(expanded.splitlines()), len(short.splitlines()))
        self.assertNotIn("\n- ", clarified)
        self.assertNotEqual(simple, factual_answer)

    def test_focused_answer_removes_duplicate_and_unrelated_overview(self):
        answer = focused_answer(
            "Who is the current rector?",
            (
                "Dr. Example is the current Rector. "
                "SIMAD University is a private higher education institution. "
                "Dr. Example is the current Rector."
            ),
        )
        self.assertEqual(answer, "Dr. Example is the current Rector.")

    def test_general_overview_stays_short_and_removes_extra_sections(self):
        answer = focused_answer(
            "Tell me about SIMAD University",
            (
                "SIMAD University is a private higher education institution located in Mogadishu, Somalia. "
                "It was established in 1999 and provides undergraduate and postgraduate programs. "
                "The mission of SIMAD University is to develop competent professionals through quality education, research, and community service.\n"
                "- Faculty of Computing\n"
                "- Faculty of Engineering\n"
                "The AI chatbot can provide information about university programs."
            ),
        )
        self.assertLessEqual(len(answer.splitlines()), 2)
        self.assertIn("private higher education institution", answer)
        self.assertIn("quality education", answer)
        self.assertNotIn("Faculty of Computing", answer)
        self.assertNotIn("programs", answer)
        self.assertNotIn("chatbot", answer.lower())

    def test_intent_classification(self):
        self.assertEqual(self.bot._classify_message_intent("hello", []), "greeting")
        cases = {
            "Should I choose SIMAD for my career?": "student_guidance",
            "How can I register at SIMAD?": "admission_registration",
            "Who is the current rector?": "simad_factual",
            "why?": "follow_up",
            "What programs are in Faculty of Law?": "academic_programs",
            "Computer Science semester 3 courses": "course_curriculum",
        }
        for question, expected in cases.items():
            self.assertEqual(self.bot._classify_intent(question, []), expected)

    def test_generated_answer_cleanup(self):
        answer = clean_generated_answer(
            "## Answer\n**SIMAD** information. [Source 1: file.pdf]\nSource: file.pdf"
        )
        self.assertEqual(answer, "Answer\nSIMAD information.")
        self.assertEqual(
            clean_generated_answer("I don't have enough verified SIMAD information."),
            NOT_FOUND_MESSAGE,
        )
        self.assertEqual(
            clean_generated_answer("Details are not provided in the reference text."),
            "Details are not provided in SIMAD records.",
        )

    def test_topic_source_restrictions(self):
        self.assertEqual(allowed_sources("What scholarships are available?"), ("Scholarships.pdf",))
        self.assertEqual(allowed_sources("Tell me about the library"), ("CAMPUS SERVICES.pdf",))
        self.assertEqual(
            allowed_sources("How can I register at SIMAD?"),
            ("ADMISSION BROCHURE.pdf",),
        )
        self.assertEqual(allowed_sources("Tell me about SIMAD clubs"), ("CLUBS.pdf",))
        self.assertEqual(
            allowed_sources("support for disabled students"),
            ("Disability Support Services (DSS).pdf",),
        )

    def test_semester_parser(self):
        self.assertEqual(semester_number("first semester civil engineering"), 1)
        self.assertEqual(semester_number("Computer Science semester 3"), 3)
        self.assertTrue(find_courses("Computer Science semester 3 courses"))
