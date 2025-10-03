import unittest

from sglang.srt.entrypoints.harmony_utils import parse_response_input


class ResponsesSystemRoleTest(unittest.TestCase):
    def test_system_role_message_is_preserved(self) -> None:
        parsed = parse_response_input(
            {"role": "system", "content": "Stay in character."},
            prev_responses=[],
        )
        self.assertEqual(parsed.author.role, "system")
        self.assertEqual(parsed.content[0].text, "Stay in character.")

    def test_system_role_content_list_preserves_text(self) -> None:
        parsed = parse_response_input(
            {
                "role": "system",
                "content": [{"type": "text", "text": "Guidance"}],
            },
            prev_responses=[],
        )
        self.assertEqual(parsed.author.role, "system")
        self.assertEqual(parsed.content[0].text, "Guidance")


if __name__ == "__main__":
    unittest.main()
