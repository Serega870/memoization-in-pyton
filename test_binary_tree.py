"""Проверки структуры деревьев, мемоизации и независимости словарей.

Запуск: python -m unittest -v test_binary_tree.py
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from binary_tree import (
    Node,
    LeafRule,
    TreeDict,
    gen_bin_tree,
    gen_bin_tree_with_namedtuple,
    memoize,
)


def plain_tree(
    root: int = 1,
    height: int = 5,
    left_leaf: LeafRule | None = None,
    right_leaf: LeafRule | None = None,
) -> TreeDict:
    """Эталонная рекурсия без мемоизации, возвращающая словари."""
    if left_leaf is None:
        left_leaf = lambda x: x * 2
    if right_leaf is None:
        right_leaf = lambda x: x + 3
    if height <= 0:
        return {"value": root, "left": None, "right": None}
    return {
        "value": root,
        "left": plain_tree(left_leaf(root), height - 1, left_leaf, right_leaf),
        "right": plain_tree(right_leaf(root), height - 1, left_leaf, right_leaf),
    }


def as_dict(node: Node | None) -> TreeDict | None:
    """Приводит namedtuple к формату эталона для сравнения структуры."""
    if node is None:
        return None
    return {
        "value": node.value,
        "left": as_dict(node.left),
        "right": as_dict(node.right),
    }


class TestTree(unittest.TestCase):
    """Проверить структуру, границы высоты и разделение поддеревьев."""

    def assert_matches_plain(
        self,
        root: int = 1,
        height: int = 5,
        left_leaf: LeafRule | None = None,
        right_leaf: LeafRule | None = None,
    ) -> None:
        """Сравнить обе реализации с независимой рекурсией без кеша."""
        expected = plain_tree(root, height, left_leaf, right_leaf)
        self.assertEqual(gen_bin_tree(root, height, left_leaf, right_leaf), expected)
        self.assertEqual(as_dict(gen_bin_tree_with_namedtuple(root, height, left_leaf, right_leaf)), expected)

    def test_default_parameters(self) -> None:
        """Сравнить деревья с параметрами по умолчанию."""
        self.assert_matches_plain()

    def test_custom_parameters(self) -> None:
        """Сравнить деревья с пользовательскими правилами и параметрами."""
        self.assert_matches_plain(
            root=7, height=4,
            left_leaf=lambda x: x - 2,
            right_leaf=lambda x: x * 3,
        )

    def test_zero_and_negative_height(self) -> None:
        """Проверить один корень при неположительной высоте."""
        for height in (0, -1):
            with self.subTest(height=height):
                self.assert_matches_plain(root=9, height=height)
                self.assertEqual(
                    gen_bin_tree_with_namedtuple(root=9, height=height),
                    Node(9, None, None),
                )

    def test_leaf_functions_are_not_called_at_zero_height(self) -> None:
        """Проверить отсутствие вызовов правил для листа."""
        left, right = Mock(), Mock()
        gen_bin_tree_with_namedtuple(height=0, left_leaf=left, right_leaf=right)
        left.assert_not_called()
        right.assert_not_called()

    def test_identical_subtrees_share_identity(self) -> None:
        """Проверить общие объекты на нескольких уровнях дерева."""
        tree = gen_bin_tree_with_namedtuple(
            root=1, height=3,
            left_leaf=lambda x: x + 1,
            right_leaf=lambda x: x + 1,
        )
        assert tree.left is not None and tree.right is not None
        assert tree.left.left is not None and tree.right.right is not None
        self.assertIs(tree.left, tree.right)
        self.assertIs(tree.left.left, tree.left.right)
        self.assertIsNotNone(tree.left.left.left)
        self.assertIs(tree.left.left.left, tree.right.right.right)

    def test_shared_subtrees_with_default_rules(self) -> None:
        """Проверить совпадающие поддеревья для стандартных правил."""
        tree = gen_bin_tree_with_namedtuple(root=3, height=2)
        # Обе функции для корня 3 дают 6.
        assert tree.left is not None
        self.assertEqual(tree.left.value, 6)
        self.assertIs(tree.left, tree.right)

    def test_different_remaining_heights_are_not_merged(self) -> None:
        """Проверить учёт высоты в ключе кеша."""
        tree = gen_bin_tree_with_namedtuple(
            height=2, left_leaf=lambda x: x, right_leaf=lambda x: x,
        )
        assert tree.left is not None and tree.left.left is not None
        self.assertEqual(tree.value, tree.left.value)
        self.assertIsNot(tree, tree.left)
        self.assertIsNone(tree.left.left.left)

    def test_cached_subtree_is_built_once(self) -> None:
        """Проверить расчёт каждого уникального состояния один раз."""
        left = Mock(side_effect=lambda x: x + 1)
        right = Mock(side_effect=lambda x: x + 1)
        gen_bin_tree_with_namedtuple(height=3, left_leaf=left, right_leaf=right)
        # Только три уникальных нелистовых состояния вместо семи позиций.
        self.assertEqual(left.call_count, 3)
        self.assertEqual(right.call_count, 3)

    def test_cache_is_local_to_each_tree(self) -> None:
        """Проверить изоляцию кешей разных построений."""
        first = gen_bin_tree_with_namedtuple(height=1, left_leaf=lambda x: x + 10)
        second = gen_bin_tree_with_namedtuple(height=1, left_leaf=lambda x: x + 20)
        assert first.left is not None and second.left is not None
        self.assertEqual(first.left.value, 11)
        self.assertEqual(second.left.value, 21)
        self.assertIsNot(first, second)

    def test_dictionary_branches_are_independent(self) -> None:
        """Проверить независимость изменяемых словарей веток."""
        tree = gen_bin_tree(root=3, height=2)
        assert tree["left"] is not None and tree["right"] is not None
        assert tree["left"]["left"] is not None
        assert tree["right"]["left"] is not None
        self.assertEqual(tree["left"], tree["right"])
        self.assertIsNot(tree["left"], tree["right"])
        self.assertIsNot(tree["left"]["left"], tree["right"]["left"])
        tree["left"]["left"]["value"] = -999
        self.assertEqual(tree["right"]["left"]["value"], 12)


class TestMemoize(unittest.TestCase):
    """Проверить число вызовов декорированной функции и кеширование None."""

    def test_same_arguments_do_not_repeat_call(self) -> None:
        """Проверить сохранение результата и расчёт новых аргументов."""
        calls: list[tuple[int, int]] = []

        @memoize
        def calculate(a: int, b: int) -> dict[str, int]:
            """Сложить аргументы и записать факт вычисления."""
            calls.append((a, b))
            return {"sum": a + b}

        first = calculate(2, 3)
        self.assertIs(calculate(2, 3), first)
        self.assertEqual(first, {"sum": 5})
        self.assertEqual(calls, [(2, 3)])
        self.assertEqual(calculate(4, 3), {"sum": 7})
        self.assertEqual(calls, [(2, 3), (4, 3)])

    def test_keyword_arguments_are_cached(self) -> None:
        """Проверить кеширование повторных именованных аргументов."""
        function = Mock(side_effect=lambda a, b: a + b)
        cached = memoize(function)
        self.assertEqual(cached(a=2, b=5), 7)
        self.assertEqual(cached(a=2, b=5), 7)
        function.assert_called_once_with(a=2, b=5)

    def test_none_result_is_cached(self) -> None:
        """Проверить кеширование результата None."""
        function = Mock(return_value=None)
        cached = memoize(function)
        self.assertIsNone(cached(1))
        self.assertIsNone(cached(1))
        function.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
