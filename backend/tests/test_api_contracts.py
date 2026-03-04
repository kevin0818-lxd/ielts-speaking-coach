import ast
import os
import unittest


class ApiContractsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.main_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "main.py",
        )
        with open(self.main_path, "r", encoding="utf-8") as f:
            self.tree = ast.parse(f.read())

    def test_analyze_routes_exist(self):
        route_paths = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                for deco in node.decorator_list:
                    if isinstance(deco, ast.Call) and getattr(deco.func, "attr", "") in {"post", "get"}:
                        if deco.args and isinstance(deco.args[0], ast.Constant) and isinstance(deco.args[0].value, str):
                            route_paths.add(deco.args[0].value)
        self.assertIn("/analyze", route_paths)
        self.assertIn("/analyze_audio", route_paths)
        self.assertIn("/analyze_audio_sync", route_paths)
        self.assertIn("/task_status/{task_id}", route_paths)

    def test_text_input_has_user_id(self):
        class_nodes = [n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == "TextInput"]
        self.assertTrue(class_nodes)
        txt = class_nodes[0]
        attrs = [n.target.id for n in txt.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
        self.assertIn("user_id", attrs)


if __name__ == "__main__":
    unittest.main()
