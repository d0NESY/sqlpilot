"""从 Spider 官方结构化 Gold SQL 中提取训练期 Schema Link 标签。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .spider_schema import SpiderSchema


@dataclass(frozen=True)
class SchemaLink:
    """一条 SQL 实际使用的表和列。"""

    tables: tuple[str, ...]
    columns: tuple[str, ...]


class _SpiderAstVisitor:
    def __init__(self, schema: SpiderSchema) -> None:
        self.schema = schema
        self.table_ids: set[int] = set()
        self.column_ids: set[int] = set()

    def visit_column_unit(self, unit: Any) -> None:
        if (
            not isinstance(unit, list)
            or len(unit) < 3
            or not isinstance(unit[1], int)
        ):
            return
        column_id = unit[1]
        if self.schema.column_identifier(column_id) is not None:
            self.column_ids.add(column_id)

    def visit_value_unit(self, unit: Any) -> None:
        if not isinstance(unit, list) or len(unit) < 3:
            return
        self.visit_column_unit(unit[1])
        if unit[2] is not None:
            self.visit_column_unit(unit[2])

    def visit_condition_value(self, value: Any) -> None:
        if isinstance(value, dict):
            self.visit_sql(value)
            return
        if not isinstance(value, list):
            return
        if (
            len(value) >= 3
            and isinstance(value[0], int)
            and isinstance(value[1], int)
            and isinstance(value[2], bool)
        ):
            self.visit_column_unit(value)
            return
        for item in value:
            self.visit_condition_value(item)

    def visit_conditions(self, conditions: Any) -> None:
        if not isinstance(conditions, list):
            return
        for condition in conditions:
            if not isinstance(condition, list) or len(condition) < 5:
                continue
            self.visit_value_unit(condition[2])
            self.visit_condition_value(condition[3])
            self.visit_condition_value(condition[4])

    def visit_sql(self, sql: Any) -> None:
        if not isinstance(sql, dict):
            raise ValueError("Spider sql 字段必须是对象")

        select = sql.get("select")
        if isinstance(select, list) and len(select) >= 2:
            for item in select[1]:
                if isinstance(item, list) and len(item) >= 2:
                    self.visit_value_unit(item[1])

        from_clause = sql.get("from")
        if isinstance(from_clause, dict):
            for table_unit in from_clause.get("table_units", []):
                if not isinstance(table_unit, list) or len(table_unit) < 2:
                    continue
                if table_unit[0] == "table_unit":
                    table_id = table_unit[1]
                    self.schema.table_name(table_id)
                    self.table_ids.add(table_id)
                elif table_unit[0] == "sql":
                    self.visit_sql(table_unit[1])
            self.visit_conditions(from_clause.get("conds", []))

        self.visit_conditions(sql.get("where", []))
        self.visit_conditions(sql.get("having", []))

        for column_unit in sql.get("groupBy", []):
            self.visit_column_unit(column_unit)

        order_by = sql.get("orderBy")
        if isinstance(order_by, list) and len(order_by) >= 2:
            for value_unit in order_by[1]:
                self.visit_value_unit(value_unit)

        for operation in ("intersect", "union", "except"):
            nested = sql.get(operation)
            if nested is not None:
                self.visit_sql(nested)


def extract_schema_link(
    sql: dict[str, Any],
    schema: SpiderSchema,
) -> SchemaLink:
    """提取并按 tables.json 的原始编号稳定排序。"""

    visitor = _SpiderAstVisitor(schema)
    visitor.visit_sql(sql)

    for column_id in visitor.column_ids:
        table_id, _ = schema.column_names[column_id]
        if table_id >= 0:
            visitor.table_ids.add(table_id)

    tables = tuple(
        schema.table_name(table_id)
        for table_id in sorted(visitor.table_ids)
    )
    columns = tuple(
        ".".join(schema.column_identifier(column_id) or ())
        for column_id in sorted(visitor.column_ids)
    )
    return SchemaLink(tables=tables, columns=columns)


def serialize_schema_link(link: SchemaLink) -> str:
    """转换成 S4 completion 使用的固定文本格式。"""

    tables = ", ".join(link.tables) if link.tables else "none"
    columns = ", ".join(link.columns) if link.columns else "none"
    return (
        "<SCHEMA_LINK>\n"
        f"tables: {tables}\n"
        f"columns: {columns}\n"
        "</SCHEMA_LINK>"
    )
