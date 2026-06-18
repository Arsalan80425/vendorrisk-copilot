from pathlib import Path

import pandas as pd

from src.config import CONTRACTS_DIR, RAW_DATA_DIR
from src.data_generation.generate_synthetic_data import generate_synthetic_data


def test_generate_synthetic_data_counts_and_demo_vendors():
    counts = generate_synthetic_data()

    vendors = pd.read_csv(RAW_DATA_DIR / "vendor_master.csv")
    invoices = pd.read_csv(RAW_DATA_DIR / "invoices.csv")
    tickets = pd.read_csv(RAW_DATA_DIR / "support_tickets.csv")
    contracts = list(Path(CONTRACTS_DIR).glob("*.txt"))

    assert counts == {"vendors": 25, "invoices": 120, "support_tickets": 90, "contracts": 25}
    assert len(vendors) == 25
    assert len(invoices) == 120
    assert len(tickets) == 90
    assert len(contracts) == 25
    assert {"DataBridge Solutions", "CloudNova Systems", "Supportly India"}.issubset(
        set(vendors["vendor_name"])
    )
