"""Построение двоичных деревьев с локальной мемоизацией.

Высота задаёт число рёбер от корня до листа. При высоте не больше нуля
возвращается только корень. Значения узлов и результаты правил — целые числа.
Правила должны быть чистыми: результат зависит только от значения аргумента.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, NamedTuple, ParamSpec, TypeVar, TypedDict, cast

P = ParamSpec("P")
R = TypeVar("R")
LeafRule = Callable[[int], int]


class Node(NamedTuple):
    """Неизменяемый узел: значение и ссылки на левое и правое поддеревья.

    Отсутствующие потомки листа представлены значением None.
    """

    value: int
    left: Node | None
    right: Node | None


class TreeDict(TypedDict):
    """Изменяемый узел словаря с теми же полями, что и Node."""

    value: int
    left: TreeDict | None
    right: TreeDict | None


def memoize(func: Callable[P, R]) -> Callable[P, R]:
    """Вернуть функцию, кеширующую результаты без ограничения размера.

    Args:
        func: Функция с хешируемыми аргументами. Для одинаковых аргументов
            она должна возвращать одинаковый результат.

    Returns:
        Обёртка с исходной сигнатурой. Повторный вызов с тем же ключом кеша
        возвращает ранее сохранённый объект, в том числе None.

    Notes:
        Используется lru_cache: позиционная и именованная формы вызова
        могут создавать разные ключи. Кеш удерживает аргументы и результаты.
        При одновременных вызовах из разных потоков возможен повторный расчёт.

    Raises:
        TypeError: При вызове обёртки с нехешируемыми аргументами.
    """
    return cast(Callable[P, R], lru_cache(maxsize=None)(func))


def gen_bin_tree_with_namedtuple(
    root: int = 1,
    height: int = 5,
    left_leaf: LeafRule | None = None,
    right_leaf: LeafRule | None = None,
) -> Node:
    """Построить дерево из узлов NamedTuple с общими поддеревьями.

    Args:
        root: Значение корня.
        height: Число рёбер от корня до листа. При height <= 0 потомков нет.
        left_leaf: Правило для левого потомка; None означает x * 2.
        right_leaf: Правило для правого потомка; None означает x + 3.

    Returns:
        Корневой узел. Состояния с одинаковыми значением и оставшейся
        высотой представлены одним объектом в пределах одного построения.
        Каждый отдельный вызов создаёт собственный кеш.

    Raises:
        TypeError: Если правило возвращает нехешируемое значение.
        RecursionError: Если глубина превышает предел рекурсии Python.

    Examples:
        >>> tree = gen_bin_tree_with_namedtuple(root=3, height=1)
        >>> tree.left is tree.right
        True
    """
    if left_leaf is None:
        left_leaf = lambda x: x * 2
    if right_leaf is None:
        right_leaf = lambda x: x + 3

    @memoize
    def make_node(value: int, remaining_height: int) -> Node:
        """Вернуть узел для значения и оставшейся высоты, используя кеш."""
        if remaining_height <= 0:
            return Node(value, None, None)
        return Node(
            value,
            make_node(left_leaf(value), remaining_height - 1),
            make_node(right_leaf(value), remaining_height - 1),
        )

    return make_node(root, height)


def gen_bin_tree(
    root: int = 1,
    height: int = 5,
    left_leaf: LeafRule | None = None,
    right_leaf: LeafRule | None = None,
) -> TreeDict:
    """Построить дерево и вернуть независимые вложенные словари.

    Args:
        root: Значение корня.
        height: Число рёбер от корня до листа; height <= 0 даёт один узел.
        left_leaf: Правило для левого потомка; по умолчанию x * 2.
        right_leaf: Правило для правого потомка; по умолчанию x + 3.

    Returns:
        Словарь с ключами value, left и right. Отсутствующие потомки — None.
        Для каждой позиции создаётся отдельный словарь, даже если исходные
        узлы NamedTuple разделяют поддерево. Изменения веток независимы.

    Notes:
        Мемоизация применяется при построении NamedTuple. Преобразование
        разворачивает все позиции: при height >= 0 создаётся
        2 ** (height + 1) - 1 словарей.

    Raises:
        TypeError: Если правило возвращает нехешируемое значение.
        RecursionError: Если глубина превышает предел рекурсии Python.
    """
    cached_tree = gen_bin_tree_with_namedtuple(root, height, left_leaf, right_leaf)

    def to_dict(node: Node) -> TreeDict:
        """Преобразовать непустой узел в отдельный словарь с потомками."""
        return {
            "value": node.value,
            "left": to_dict(node.left) if node.left is not None else None,
            "right": to_dict(node.right) if node.right is not None else None,
        }

    return to_dict(cached_tree)
