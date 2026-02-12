import json

from bist_bot.execution.paper_broker import PaperBroker


def test_open_positions_tracks_avg_price_and_unrealized_pnl():
    broker = PaperBroker(starting_cash=1_000.0)
    broker.place_order("TEST.IS", "market", quantity=2.0, price=100.0)
    broker.place_order("TEST.IS", "market", quantity=1.0, price=200.0)

    open_positions = broker.get_open_positions(current_prices={"TEST.IS": 150.0})
    assert len(open_positions) == 1
    row = open_positions[0]
    assert row["symbol"] == "TEST.IS"
    assert round(float(row["quantity"]), 6) == 3.0
    assert round(float(row["avg_price"]), 6) == round(400.0 / 3.0, 6)
    assert round(float(row["unrealized_pnl"]), 6) == 50.0

    broker.place_order("TEST.IS", "market", quantity=-1.0, price=160.0)
    open_positions_after = broker.get_open_positions(current_prices={"TEST.IS": 160.0})
    assert len(open_positions_after) == 1
    row_after = open_positions_after[0]
    assert round(float(row_after["quantity"]), 6) == 2.0
    assert round(float(row_after["avg_price"]), 6) == round(400.0 / 3.0, 6)


def test_load_state_rebuilds_position_costs_when_missing(tmp_path):
    state_path = tmp_path / "paper_state.json"
    broker = PaperBroker(starting_cash=1_000.0, state_path=str(state_path))
    broker.place_order("TEST.IS", "market", quantity=2.0, price=100.0)
    broker.place_order("TEST.IS", "market", quantity=-1.0, price=120.0)
    broker.save_state()

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload.pop("position_costs", None)
    state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    restored = PaperBroker(state_path=str(state_path), auto_load=True)
    open_positions = restored.get_open_positions(current_prices={"TEST.IS": 120.0})
    assert len(open_positions) == 1
    row = open_positions[0]
    assert round(float(row["quantity"]), 6) == 1.0
    assert round(float(row["avg_price"]), 6) == 100.0
