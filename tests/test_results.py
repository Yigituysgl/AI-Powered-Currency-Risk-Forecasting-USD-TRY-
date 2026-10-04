import pandas as pd

from usdtry_risk.results import load_table, save_table


def test_save_and_load_round_trip(tmp_path):
    df = pd.DataFrame({"mae": [0.5, 0.6]}, index=pd.Index(["a", "b"], name="model"))
    df.attrs.update(n=2, start=pd.Timestamp("2020-01-02"))
    save_table(df, "scores", tmp_path)

    loaded = load_table("scores", tmp_path)
    pd.testing.assert_frame_equal(loaded, df)
    assert loaded.attrs["n"] == 2
    assert loaded.attrs["start"] == "2020-01-02"
    assert loaded.attrs["snapshot"].endswith(".csv")
