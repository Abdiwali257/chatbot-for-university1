from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from chatbot_clean import (
    GENERATION_UNAVAILABLE_MESSAGE,
    NOT_FOUND_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    SearchResult,
    SimadChatbot,
)
from huggingface_hub.errors import HfHubHTTPError


class FakeGenerator:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class HuggingFaceProviderTests(TestCase):
    def make_bot(self, generator, matches=None):
        bot = SimadChatbot.__new__(SimadChatbot)
        bot.generator = generator
        bot.hf_model = "test-model"
        bot.last_answer_mode = "ready"
        bot.search = lambda question, limit=8: matches if matches is not None else [
            SearchResult("Campus services include a library.", "source.pdf", "page 1", 0.1)
        ]
        return bot

    def test_factual_answer_uses_hugging_face(self):
        generator = FakeGenerator("SIMAD provides campus services.")
        bot = self.make_bot(generator)

        answer = bot.answer("Tell me about campus services", [])

        self.assertIn("campus services", answer)
        self.assertEqual(bot.last_answer_mode, "huggingface")
        self.assertEqual(len(generator.calls), 1)

    def test_new_factual_question_includes_bounded_memory_and_keeps_grounding_rule(self):
        generator = FakeGenerator("Campus services include a library.")
        bot = self.make_bot(generator)
        history = [
            {"role": "user", "content": "Old question"},
            {"role": "assistant", "content": "Old answer"},
            {"role": "user", "content": "Recent question"},
            {"role": "assistant", "content": "Recent answer"},
        ]

        bot.answer("Tell me about campus services", history)

        messages = generator.calls[0]["messages"]
        self.assertEqual(len(messages), 3)
        self.assertIn("Conversation memory", messages[1]["content"])
        self.assertIn("Old question", str(messages))
        self.assertIn("Recent answer", str(messages))
        self.assertIn("Never treat an earlier assistant answer as factual evidence", str(messages))
        self.assertIn("SIMAD UNIVERSITY REFERENCE MATERIAL", str(messages))

    def test_somali_question_requests_somali_answer(self):
        generator = FakeGenerator("SIMAD waa jaamacad bixisa adeegyo arday.")
        bot = self.make_bot(
            generator,
            matches=[
                SearchResult(
                    "SIMAD University provides student services.",
                    "source.pdf",
                    "page 1",
                    0.1,
                )
            ],
        )

        answer = bot.answer("SIMAD maxay tahay ii sheeg", [])

        self.assertIn("SIMAD waa", answer)
        self.assertEqual(bot.last_answer_mode, "huggingface")
        self.assertIn("Answer language: Somali", str(generator.calls[0]["messages"]))

    def test_somali_grounding_still_rejects_unsupported_numbers(self):
        generator = FakeGenerator("SIMAD waxaa la aasaasay 2005.")
        bot = self.make_bot(
            generator,
            matches=[
                SearchResult(
                    "SIMAD University was established in 1999.",
                    "source.pdf",
                    "page 1",
                    0.1,
                )
            ],
        )

        answer = bot.answer("SIMAD goorma ayaa la aasaasay?", [])

        self.assertNotIn("2005", answer)
        self.assertEqual(bot.last_answer_mode, "local_grounded")

    def test_guidance_uses_database_context_and_hugging_face(self):
        generator = FakeGenerator(
            "SIMAD undergraduate faculties and programs can guide student study choices."
        )
        bot = self.make_bot(generator, matches=[])

        answer = bot.answer("Should I choose SIMAD for my career?", [])

        self.assertIn("undergraduate faculties and programs", answer)
        self.assertEqual(bot.last_answer_mode, "huggingface")
        prompt = str(generator.calls[0]["messages"])
        self.assertIn("Faculty of Computing", prompt)
        self.assertIn("Faculty of Law", prompt)

    def test_technology_guidance_prompt_is_filtered(self):
        generator = FakeGenerator(
            "For a technology career, SIMAD Computing programs include Computer Science, "
            "Information Technology, and Graphics and Multimedia."
        )
        bot = self.make_bot(generator, matches=[])

        answer = bot.answer(
            "can you give me an advice if i want to choose my career technology", []
        )

        self.assertIn("technology career", answer)
        prompt = str(generator.calls[0]["messages"])
        self.assertIn("Faculty of Computing", prompt)
        self.assertIn("Information Technology", prompt)
        self.assertNotIn("Faculty of Medicine", prompt)
        self.assertNotIn("Faculty of Law", prompt)

    def test_comparison_prompt_contains_only_requested_faculties(self):
        generator = FakeGenerator(
            "Computing lists Computer Science, Information Technology, and Graphics and "
            "Multimedia. Management Sciences lists Business Administration, Banking & "
            "Finance, Procurement and Logistics, Digital Marketing, and Entrepreneurship "
            "and Innovation."
        )
        bot = self.make_bot(generator, matches=[])

        answer = bot.answer(
            "what is difference between faculty of computing and Management Sciences", []
        )

        self.assertIn("Computing lists Computer Science", answer)
        prompt = str(generator.calls[0]["messages"])
        self.assertIn("Faculty of Computing", prompt)
        self.assertIn("Faculty of Management Sciences", prompt)
        self.assertNotIn("Faculty of Medicine", prompt)
        self.assertNotIn("Faculty of Social Sciences", prompt)

    def test_faculty_overview_uses_retrieved_database_context_without_curriculum_dump(self):
        generator = FakeGenerator("The Faculty of Law offers the Law program.")
        bot = self.make_bot(
            generator,
            matches=[
                SearchResult(
                    "LAW4228 Administrative Law course row",
                    "Faculty of Law.xlsx",
                    "sheet Law",
                    0.1,
                )
            ],
        )

        answer = bot.answer("give me information about faculty of law", [])

        self.assertIn("Faculty of Law", answer)
        self.assertIn("Law program", answer)
        self.assertNotIn("LAW4228", answer)
        self.assertEqual(len(generator.calls), 1)
        self.assertIn("Verified undergraduate programs: Law", str(generator.calls[0]))

    def test_casual_messages_bypass_retrieval_and_generation(self):
        generator = FakeGenerator("This should not be used.")
        bot = self.make_bot(generator)
        bot.search = Mock(side_effect=AssertionError("retrieval should not run"))
        for question in ["hello", "can you help me", "skip it", "do you know messi", "fuck off"]:
            self.assertTrue(bot.answer(question, []))
        self.assertEqual(generator.calls, [])

    def test_previous_answer_transformations_bypass_retrieval(self):
        generator = FakeGenerator("This should not be used.")
        bot = self.make_bot(generator)
        bot.search = Mock(side_effect=AssertionError("retrieval should not run"))
        history = [
            {"role": "user", "content": "Who founded SIMAD?"},
            {
                "role": "assistant",
                "content": (
                    "SIMAD was established in 1999.\n"
                    "- Hassan Sheikh Mohammoud\n"
                    "- Farah Sheikh Abdikadir\n"
                    "- Mohamed Hussein Dhobale"
                ),
            },
        ]

        for follow_up in [
            "summarize this",
            "make it short",
            "make it 5 lines",
            "explain again",
            "what do you mean?",
            "for what question?",
        ]:
            self.assertTrue(bot.answer(follow_up, history))
        self.assertGreater(len(generator.calls), 0)

    def test_followup_rewrite_uses_only_previous_answer_and_changes_wording(self):
        generator = FakeGenerator(
            "SIMAD exchange opportunities let students study at partner universities "
            "for one or two semesters through Erasmus+."
        )
        bot = self.make_bot(generator)
        bot.search = Mock(side_effect=AssertionError("retrieval should not run"))
        history = [
            {
                "role": "user",
                "content": "Tell me about SIMAD student exchange.",
                "kind": "factual_question",
            },
            {
                "role": "assistant",
                "content": (
                    "SIMAD University offers student exchange programs through the Erasmus+ "
                    "program. Students can study at partner universities for one or two semesters."
                ),
                "kind": "factual_answer",
            },
        ]

        answer = bot.answer("explain simply", history)

        self.assertIn("exchange opportunities", answer)
        self.assertNotEqual(answer, history[-1]["content"])
        prompt = str(generator.calls[0]["messages"])
        self.assertIn("Verified source answer", prompt)
        self.assertNotIn("New verified SIMAD reference", prompt)

    def test_followup_rewrite_rejects_unchanged_or_invented_output(self):
        source = "SIMAD University was established in 1999."
        self.assertFalse(SimadChatbot._rewrite_is_grounded(source, source))
        self.assertFalse(
            SimadChatbot._rewrite_is_grounded(
                "Students can study abroad.\nThey can learn internationally.",
                "Students can study abroad. They can learn internationally. "
                "The exchange lasts one semester.",
            )
        )
        self.assertFalse(
            SimadChatbot._rewrite_is_grounded(
                "SIMAD University was established in 2005.", source
            )
        )
        self.assertFalse(
            SimadChatbot._rewrite_is_grounded(
                "Bring your high school diploma to the college.",
                "Bring your secondary school certificate to SIMAD University.",
            )
        )

    def test_single_long_model_summary_uses_short_line_fallback(self):
        generator = FakeGenerator(
            "Applicants must pass the exam, bring a certificate, provide photos, "
            "complete an interview, and pay the fee."
        )
        bot = self.make_bot(generator)
        bot.search = Mock(side_effect=AssertionError("retrieval should not run"))
        history = [
            {"role": "user", "content": "What are the requirements?", "kind": "factual_question"},
            {
                "role": "assistant",
                "content": (
                    "Requirements:\n- Pass the exam\n- Bring a certificate\n"
                    "- Provide photos\n- Complete an interview\n- Pay the fee"
                ),
                "kind": "factual_answer",
            },
        ]

        answer = bot.answer("make it short", history)

        self.assertGreaterEqual(len(answer.splitlines()), 2)
        self.assertLessEqual(len(answer.splitlines()), 3)

    def test_program_and_course_lists_use_retrieval_before_database_fallback(self):
        generator = FakeGenerator("This should not be used.")
        bot = self.make_bot(generator)
        bot.search = Mock(return_value=[])

        programs = bot.answer("What programs are inside Faculty of Computing?", [])
        courses = bot.answer("Computer Science semester 3 courses", [])

        self.assertIn("Computer Science", programs)
        self.assertIn("- Physics", courses)
        self.assertGreaterEqual(bot.search.call_count, 2)
        self.assertNotEqual(bot.last_answer_mode, "structured_database")

    def test_program_availability_blocks_wrong_leadership_context(self):
        generator = FakeGenerator(
            "Deputy Rector for Research & Consultancy - SIMAD University Senate Secretary"
        )
        bot = self.make_bot(
            generator,
            matches=[
                SearchResult(
                    "Deputy Rector for Research & Consultancy - SIMAD University Senate Secretary",
                    "data\\THE SENATE.pdf",
                    "page 2",
                    0.1,
                )
            ],
        )

        answer = bot.answer("Does SIMAD offer Electrical Engineering?", [])

        self.assertIn("Electrical Engineering", answer)
        self.assertIn("Faculty of Engineering", answer)
        self.assertNotIn("Senate Secretary", answer)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(bot.last_semantic_topic, "academics")

    def test_provider_failure_returns_verified_database_fallback(self):
        bot = self.make_bot(FakeGenerator(error=RuntimeError("service unavailable")))

        answer = bot.answer("Tell me about campus services", [])

        self.assertEqual(answer, "Campus services include a library.")
        self.assertEqual(bot.last_answer_mode, "local_grounded")

    def test_authentication_failure_disables_generator(self):
        error = HfHubHTTPError("401 Unauthorized", response=Mock(status_code=401))
        bot = self.make_bot(FakeGenerator(error=error))

        answer = bot.answer("Tell me about campus services", [])

        self.assertEqual(answer, "Campus services include a library.")
        self.assertIsNone(bot.generator)

    def test_fallback_extracts_clean_answer_instead_of_dumping_raw_chunk(self):
        raw_chunk = (
            "SIMAD UNIVERSITY GENERAL INFORMATION "
            "Q: What is SIMAD University? "
            "A: SIMAD University is a private higher education institution in Mogadishu. "
            "Q: What is the mission? A: A different answer. "
            "Facts & Figures | Annual Report | Meeting Minutes"
        )
        answer = SimadChatbot._retrieval_only_answer(
            "What is SIMAD University?",
            [SearchResult(raw_chunk, "general.pdf", "page 1", 0.1)],
        )
        self.assertEqual(
            answer,
            "SIMAD University is a private higher education institution in Mogadishu.",
        )
        self.assertNotIn("Facts & Figures", answer)
        self.assertNotIn("Q:", answer)

    def test_fallback_never_dumps_large_unstructured_chunk(self):
        answer = SimadChatbot._retrieval_only_answer(
            "Tell me about SIMAD",
            [SearchResult("Raw table | " * 200, "source.pdf", "page 1", 0.1)],
        )
        self.assertEqual(answer, GENERATION_UNAVAILABLE_MESSAGE)

    def test_no_matches_returns_exact_not_found(self):
        bot = self.make_bot(FakeGenerator("unused"), matches=[])
        answer = bot.answer("What is the helicopter parking policy?", [])
        self.assertEqual(answer, NOT_FOUND_MESSAGE)
        self.assertEqual(bot.last_answer_mode, "not_found")

    def test_generated_answer_with_unverified_number_is_rejected(self):
        self.assertFalse(
            SimadChatbot._generated_answer_is_grounded(
                "SIMAD was established in 2005.",
                "SIMAD was established in 1999.",
            )
        )


class RetrievalRelevanceTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = SimadChatbot()
        cls.bot.generator = None

    def test_scholarship_query_only_retrieves_scholarship_document(self):
        matches = self.bot.search("Does SIMAD offer scholarships, and who qualifies for them?")
        self.assertTrue(matches)
        self.assertTrue(all(match.source.endswith("Scholarships.pdf") for match in matches))

    def test_unrelated_query_returns_no_matches(self):
        self.assertEqual(self.bot.search("Explain quantum entanglement and black holes"), [])

    def test_faculty_law_has_database_context(self):
        context = self.bot._database_context("give information about faculty of law", "simad_factual")
        self.assertIn("Faculty of Law", context)

    def test_unknown_parking_policy_does_not_reuse_history(self):
        history = [
            {"role": "user", "content": "Tell me about international exchange"},
            {"role": "assistant", "content": "Previous international exchange answer."},
        ]
        answer = self.bot.answer("What is the helicopter parking policy?", history)
        self.assertEqual(answer, NOT_FOUND_MESSAGE)
        self.assertNotIn("exchange", answer.lower())

    def test_out_of_scope_question_is_rejected(self):
        answer = self.bot.answer("Who won the World Cup?", [])
        self.assertEqual(answer, OUT_OF_SCOPE_MESSAGE)

    def test_local_engine_answers_document_topics_without_hugging_face(self):
        cases = {
            "give me information about SIMAD": "private higher education institution",
            "What are the admission requirements?": "National High School Exam",
            "What scholarships are available?": "Academic merit scholarships",
            "What postgraduate programs are available?": "Master of Computer Science",
            "How is GPA calculated?": "95-100 4.00 A",
            "Who is the current rector?": "Abdikarim Mohaidin Ahmed",
        }
        for question, expected in cases.items():
            answer = self.bot.answer(question, [])
            self.assertIn(expected, answer)
            self.assertEqual(self.bot.last_answer_mode, "local_grounded")

    def test_local_general_overview_does_not_repeat_establishment_fact(self):
        answer = self.bot.answer("give me information about SIMAD", [])
        self.assertEqual(answer.lower().count("established in 1999"), 1)
        self.assertLessEqual(len(answer.splitlines()), 2)
        self.assertNotIn("facult", answer.lower())
        self.assertNotIn("programs", answer.lower())
        self.assertNotIn("chatbot", answer.lower())

    def test_general_overview_prompt_excludes_chatbot_project_text(self):
        generator = FakeGenerator(
            "SIMAD University is a private higher education institution located in Mogadishu, Somalia. "
            "It was established in 1999 and focuses on quality education, research, and community service."
        )
        raw_general_text = (
            "Q: What is SIMAD University? "
            "A: SIMAD University is a private higher education institution located in Mogadishu, Somalia. "
            "It was established in 1999 and provides undergraduate and postgraduate programs across multiple academic disciplines. "
            "Q: What is the vision of SIMAD University? "
            "A: SIMAD University aims to provide quality higher education, research, and community service that contribute to development. "
            "Q: Who can use the AI-based university chatbot? "
            "A: The chatbot can be used by students, prospective students, parents, visitors, and staff."
        )
        bot = SimadChatbot.__new__(SimadChatbot)
        bot.generator = generator
        bot.hf_model = "test-model"
        bot.last_answer_mode = "ready"
        bot.search = lambda question, limit=8: [
            SearchResult(
                raw_general_text,
                "data\\SIMAD UNIVERSITY GENERAL INFORMATION.pdf",
                "page 1",
                0.1,
            )
        ]

        answer = bot.answer("Tell me about SIMAD University", [])
        prompt = str(generator.calls[0]["messages"])

        self.assertNotIn("AI-based university chatbot", prompt)
        self.assertNotIn("chatbot can be used", prompt)
        self.assertNotIn("programs across multiple", prompt)
        self.assertNotIn("chatbot", answer.lower())
        self.assertLessEqual(len(answer.splitlines()), 2)

    def test_local_pdf_answers_are_clean_and_focused(self):
        admission = self.bot.answer("What are the admission requirements?", [])
        self.assertIn("SIMAD admission requirements:", admission)
        self.assertIn("1. Pass the National High School Exam", admission)
        self.assertNotIn("ADMISSION REQUIREMENTS Should", admission)
        self.assertNotIn(" 01 ", admission)

        scholarships = self.bot.answer(
            "Does SIMAD offer scholarships, and who qualifies for them?", []
        )
        self.assertIn("- Academic merit scholarships", scholarships)
        self.assertIn("Qualification details found:", scholarships)
        self.assertNotIn("take place. Join us", scholarships)
        self.assertNotIn("Q:", scholarships)

    def test_semantic_followup_classifier_understands_natural_variants(self):
        history = [
            {"role": "user", "content": "What are the admission requirements?"},
            {
                "role": "assistant",
                "content": (
                    "Requirements:\n- Pass the national exam\n- Bring certificate\n"
                    "- Bring photos\n- Pass interview\n- Pay processing fee"
                ),
            },
        ]
        variants = {
            "please condense your response to the essential details": "summarize",
            "that response is much too long": "summarize",
            "could you restate that in plain language": "simplify",
            "I am confused by your explanation": "clarify",
            "say it simply": "simplify",
            "this is messy": "clean",
            "too short": "expand",
            "show me your response once more": "repeat",
            "remind me what we were discussing": "topic",
        }
        for message, expected_action in variants.items():
            self.assertEqual(
                self.bot._follow_up_action(message, has_previous=True),
                expected_action,
                message,
            )
            self.assertEqual(
                self.bot._classify_message_intent(message, history),
                "follow_up_answer",
                message,
            )

    def test_semantic_classifier_distinguishes_message_categories(self):
        history = [
            {"role": "user", "content": "Who founded SIMAD?"},
            {"role": "assistant", "content": "SIMAD has three founding fathers."},
        ]
        self.assertEqual(
            self.bot._classify_message_intent("What is the tuition fee for CS?", history),
            "new_simad_question",
        )
        self.assertEqual(
            self.bot._classify_message_intent("What is the weather forecast?", history),
            "out_of_scope",
        )
        self.assertEqual(
            self.bot._classify_message_intent("much appreciated", history),
            "thanks",
        )
        self.assertEqual(
            self.bot._classify_message_intent("hello there", history),
            "greeting",
        )
        self.assertEqual(
            self.bot._classify_message_intent("what can you help me with", history),
            "chatbot_capability",
        )
        self.assertEqual(
            self.bot._classify_message_intent("ok skip it", history),
            "conversation_control",
        )

    def test_natural_semantic_followups_bypass_retrieval(self):
        history = [
            {"role": "user", "content": "What are the admission requirements?"},
            {
                "role": "assistant",
                "content": "Requirements:\n- Exam\n- Certificate\n- Photos\n- Interview\n- Fee",
            },
        ]
        original_search = self.bot.search
        self.bot.search = Mock(side_effect=AssertionError("retrieval should not run"))
        try:
            for message in [
                "please condense your response to the essential details",
                "could you restate that in plain language",
                "show me your response once more",
                "remind me what we were discussing",
            ]:
                self.assertTrue(self.bot.answer(message, history))
                self.assertEqual(self.bot.last_answer_mode, "conversation")
        finally:
            self.bot.search = original_search

    def test_semantic_topic_routing_understands_leadership_meanings_and_typos(self):
        questions = [
            "who runs the institution",
            "tell me about the people in charge",
            "who are the university officials",
            "who is the president",
            "who manages simaad unversity",
            "what does the board do",
            "could you describe the senior team",
        ]
        for question in questions:
            self.assertEqual(self.bot._semantic_topic(question), "leadership", question)
            self.assertIn("THE SENATE.pdf", self.bot._topic_sources(question), question)
            self.assertEqual(
                self.bot._classify_message_intent(question, []),
                "new_simad_question",
                question,
            )

    def test_semantic_topic_routing_understands_unseen_natural_phrasings(self):
        cases = {
            "what student funding can I get": "scholarships",
            "tell me about studying in another country": "exchange",
            "how are marks worked out": "grading",
            "who handles the place": "leadership",
        }
        for question, expected_topic in cases.items():
            self.assertEqual(self.bot._semantic_topic(question), expected_topic, question)
            self.assertEqual(self.bot._semantic_scope(question), "simad_related", question)

    def test_semantic_leadership_answer_uses_verified_senate_roles(self):
        answer = self.bot.answer("who runs the institution", [])
        self.assertIn("The Senate", answer)
        self.assertIn("Dr. Abdikarim Mohaidin Ahmed The Rector", answer)
        self.assertIn("Deputy Rector for Student Affairs", answer)
        self.assertNotIn("Founding fathers", answer)

    def test_semantic_leadership_answer_does_not_allow_generator_to_reassign_roles(self):
        generator = FakeGenerator(
            "The Rector is Mr. Yusuf Moallim Ahmed and the advisor is Dr. Abdikarim."
        )
        self.bot.generator = generator
        try:
            answer = self.bot.answer("could you describe the senior team", [])
        finally:
            self.bot.generator = None

        self.assertIn("Dr. Abdikarim Mohaidin Ahmed The Rector", answer)
        self.assertIn("Mr. Yusuf Moallim Ahmed Senior Advisor of the Rector", answer)
        self.assertEqual(generator.calls, [])

    def test_natural_factual_followups_resolve_last_simad_topic(self):
        history = [
            {
                "role": "user",
                "content": "Who are the people in charge of SIMAD?",
                "kind": "factual_question",
            },
            {
                "role": "assistant",
                "content": "SIMAD University leadership listed by the Senate includes officials.",
                "kind": "factual_answer",
            },
        ]
        for follow_up in [
            "all of them",
            "who are they",
            "tell me more",
            "what about it",
            "give me the rest",
        ]:
            self.assertEqual(
                self.bot._classify_message_intent(follow_up, history),
                "follow_up_question",
                follow_up,
            )
            contextual = self.bot._contextual_question(follow_up, history)
            self.assertIn("SIMAD University Senate leaders", contextual)
            answer = self.bot.answer(follow_up, history)
            self.assertIn("Senate", answer)
            self.assertIn("Senate Secretary", answer)

    def test_semantic_scope_rejects_unrelated_meanings(self):
        for question in [
            "who won the world cup",
            "how do I cook rice",
            "what is tomorrow's weather",
        ]:
            self.assertEqual(
                self.bot._classify_message_intent(question, []),
                "out_of_scope",
                question,
            )
            self.assertEqual(self.bot.answer(question, []), OUT_OF_SCOPE_MESSAGE)

    def test_regression_small_talk_bypasses_retrieval(self):
        original_search = self.bot.search
        self.bot.search = Mock(side_effect=AssertionError("retrieval should not run"))
        try:
            goodbye = self.bot.answer("bye", [])
            self.assertIn("Goodbye", goodbye)
            self.assertNotIn("SIMAD University is", goodbye)
            self.assertNotEqual(goodbye, NOT_FOUND_MESSAGE)

            control_history = [
                {
                    "role": "user",
                    "content": "What are the admission requirements?",
                    "kind": "factual_question",
                },
                {
                    "role": "assistant",
                    "content": "SIMAD admission requirements are listed.",
                    "kind": "factual_answer",
                },
            ]
            control = self.bot.answer("can we talk about another thing?", control_history)
            self.assertIn("What else would you like to know", control)
            self.assertNotIn("That answer was for your question", control)
            self.assertNotEqual(control, NOT_FOUND_MESSAGE)
        finally:
            self.bot.search = original_search

    def test_regression_full_simad_question_overrides_followup_memory(self):
        history = [
            {
                "role": "user",
                "content": "What are the admission requirements?",
                "kind": "factual_question",
            },
            {
                "role": "assistant",
                "content": "SIMAD admission requirements are listed.",
                "kind": "factual_answer",
            },
        ]

        answer = self.bot.answer("Who are the previous rectors of SIMAD?", history)

        self.assertNotIn("That answer was for your question", answer)
        self.assertNotEqual(answer, OUT_OF_SCOPE_MESSAGE)
        self.assertEqual(self.bot.last_message_intent, "new_simad_question")

    def test_regression_gpa_answer_is_clean(self):
        answer = self.bot.answer("Explain SIMAD GPA system", [])

        self.assertIn("95-100", answer)
        self.assertIn("4.00", answer)
        self.assertNotIn("Home", answer)
        self.assertNotIn("Admission", answer)
        self.assertNotIn("Marks Grade Points", answer)
        self.assertNotIn("Marks Grade Points Grade", answer)

    def test_regression_leadership_answer_is_clean(self):
        answer = self.bot.answer("Who are the top officials at SIMAD?", [])

        self.assertIn("Dr. Abdikarim Mohaidin Ahmed", answer)
        self.assertIn("Rector", answer)
        self.assertNotIn("The Senate The Senate List Profiles", answer)

    def test_regression_missed_exam_does_not_return_gpa(self):
        answer = self.bot.answer("What happens if a student misses an exam?", [])

        self.assertTrue(answer == NOT_FOUND_MESSAGE or "exam" in answer.lower())
        self.assertNotIn("95-100", answer)
        self.assertNotIn("Marks Grade Points", answer)
        self.assertNotIn("Grade Points Grade", answer)

    def test_regression_history_people_answers_are_extracted_cleanly(self):
        previous_rectors = self.bot.answer("who was the previous rectors", [])
        self.assertIn("Abdirahman Mohamed Hussein Odowa", previous_rectors)
        self.assertIn("Dahir Hassan Arab", previous_rectors)
        self.assertNotIn("History And Awards", previous_rectors)
        self.assertNotIn("SIMAD Timeline", previous_rectors)
        self.assertNotIn("Launched ICE Institute", previous_rectors)

        founders = self.bot.answer("who found SIMAD", [])
        self.assertIn("Farah Sheikh Abdikadir", founders)
        self.assertIn("Mohamed Hussein Dhobale", founders)
        self.assertIn("Hassan Sheikh", founders)
        self.assertNotIn("History And Awards", founders)
        self.assertNotIn("The Visionaries Behind SIMAD", founders)
