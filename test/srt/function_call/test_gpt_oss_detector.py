from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.test_utils import CustomTestCase


class TestGptOssDetector(CustomTestCase):
    def test_deduplicates_and_limits_tool_calls(self):
        tools = [
            Tool(function=Function(name="get_current_weather", parameters={})),
            Tool(function=Function(name="get_current_stock_price", parameters={})),
        ]

        parser = FunctionCallParser(tools, "gpt-oss")

        text = (
            "<|channel|>analysis<|message|>Need to gather data<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.get_current_weather"
            "<|constrain|>json<|message|>{\"location\": \"San Francisco, CA\", \"unit\": \"fahrenheit\"}<|call|>"
            "<|channel|>commentary to=functions.get_current_weather<|constrain|>json"
            "<|message|>{\"location\": \"New York, NY\", \"unit\": \"fahrenheit\"}<|call|>"
            "<|channel|>commentary to=functions.get_current_stock_price<|constrain|>json"
            "<|message|>{\"symbol\": \"AAPL\", \"exchange\": \"NASDAQ\"}<|call|>"
            "<|channel|>commentary to=functions.get_current_stock_price<|constrain|>json"
            "<|message|>{\"symbol\": \"GOOGL\", \"exchange\": \"NASDAQ\"}<|call|>"
        )

        _, calls = parser.parse_non_stream(text)

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            {call.name for call in calls},
            {"get_current_weather", "get_current_stock_price"},
        )
        self.assertListEqual([call.tool_index for call in calls], [0, 1])
