import json

from automationbench.schema.quickbooks import QBPayment
from automationbench.schema.world import WorldState
from automationbench.tools.api.fetch import api_fetch


def test_payment_serialization_exposes_exact_payment_reference():
    payment = QBPayment(id="qp_001", payment_number=" PMT-2026-0401 ")

    assert payment.to_display_dict()["PaymentRefNum"] == " PMT-2026-0401 "


def test_payment_serialization_omits_missing_payment_reference():
    payment = QBPayment(id="qp_001")

    assert "PaymentRefNum" not in payment.to_display_dict()


def test_payment_query_and_get_responses_expose_payment_reference():
    world = WorldState(
        quickbooks={
            "payments": [
                {
                    "id": "qp_003",
                    "payment_number": "PMT-2026-0405",
                    "total_amt": "4100.00",
                }
            ]
        }
    )
    query_response = json.loads(
        api_fetch(
            world,
            "POST",
            "https://quickbooks.api.intuit.com/v3/company/12345/query",
            body=json.dumps({"query": "SELECT * FROM Payment"}),
        )
    )
    get_response = json.loads(
        api_fetch(
            world,
            "GET",
            "https://quickbooks.api.intuit.com/v3/company/12345/payment/qp_003",
        )
    )

    assert query_response["QueryResponse"]["PAYMENT"][0]["PaymentRefNum"] == "PMT-2026-0405"
    assert get_response["Payment"]["PaymentRefNum"] == "PMT-2026-0405"


def test_payment_create_response_exposes_payment_reference():
    world = WorldState()

    response = json.loads(
        api_fetch(
            world,
            "POST",
            "https://quickbooks.api.intuit.com/v3/company/12345/payment",
            body=json.dumps({"PaymentRefNum": "PMT-2026-0410", "TotalAmt": "25.00"}),
        )
    )

    assert response["Payment"]["PaymentRefNum"] == "PMT-2026-0410"
    assert world.quickbooks.payments[0].payment_number == "PMT-2026-0410"