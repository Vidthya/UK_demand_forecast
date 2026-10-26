##All data here is small, made-up data, not NESO data
import numpy as np
import pandas as pd
import pytest

import demand_forecast as df_mod


def make_frame(start="2016-01-01", days=30, periods=48, seed=0):
    """Made-up half-hourly demand in the same layout load() produces."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in pd.date_range(start, periods=days):
        for p in range(1, periods + 1):
            rows.append((day, p, 25000 + 3000 * np.sin(p / 48 * 2 * np.pi) + rng.normal(0, 200)))
    return pd.DataFrame(rows, columns=["date", "period", "y"])


def write_csv(path, rows, columns=("SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND")):
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)


##Metric

def test_wape_perfect_forecast_is_zero():
    y = np.array([100.0, 200.0, 300.0])
    assert df_mod.wape(y, y) == 0


def test_wape_known_value():
    # errors 10 + 20 = 30, total demand 300 -> 10%
    y = np.array([100.0, 200.0])
    p = np.array([110.0, 180.0])
    assert df_mod.wape(y, p) == pytest.approx(10.0)


def test_wape_counts_over_and_under_prediction_equally():
    y = np.array([100.0, 100.0])
    assert df_mod.wape(y, y + 10) == pytest.approx(df_mod.wape(y, y - 10))


##Features

def test_lags_match_the_right_earlier_values():
    frame = make_frame(days=20)
    feats = df_mod.add_features(frame)
    lookup = frame.set_index(["date", "period"])["y"]
    for _, row in feats.sample(50, random_state=1).iterrows():
        for k, col in [(1, "lag1"), (2, "lag2"), (7, "lag7")]:
            earlier = (row["date"] - pd.Timedelta(days=k), row["period"])
            assert row[col] == pytest.approx(lookup[earlier])


def test_mean7_is_mean_of_previous_seven_days_same_period():
    frame = make_frame(days=20)
    feats = df_mod.add_features(frame)
    lookup = frame.set_index(["date", "period"])["y"]
    row = feats.iloc[100]
    expected = np.mean([lookup[(row["date"] - pd.Timedelta(days=k), row["period"])] for k in range(1, 8)])
    assert row["mean7"] == pytest.approx(expected)


def test_first_week_is_dropped_because_lags_are_missing():
    frame = make_frame(start="2016-01-01", days=20)
    feats = df_mod.add_features(frame)
    assert feats["date"].min() == pd.Timestamp("2016-01-08")


def test_features_never_contain_the_target_itself():
    assert "y" not in df_mod.features


def test_features_use_only_information_from_earlier_days():
    """Changing demand on a later day must not change the features of earlier rows."""
    frame = make_frame(days=30)
    base = df_mod.add_features(frame)

    changed = frame.copy()
    cutoff = pd.Timestamp("2016-01-20")
    changed.loc[changed["date"] >= cutoff, "y"] += 5000
    after = df_mod.add_features(changed)

    ##features for the first day of change use only days before it
    ##must be identical, rows on the cutoff day depend only on pre-cutoff days
    cols = df_mod.features
    b = base[base["date"] <= cutoff].reset_index(drop=True)
    a = after[after["date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(b[cols], a[cols])


def test_calendar_features_are_in_expected_ranges():
    feats = df_mod.add_features(make_frame(days=40))
    assert feats["dow"].between(0, 6).all()
    assert feats["doy_sin"].between(-1, 1).all()
    assert feats["doy_cos"].between(-1, 1).all()


##Loading

def test_load_parses_dates_and_renames_columns(tmp_path):
    write_csv(tmp_path / "a.csv", [("01-Jan-2016", 1, 20000), ("01-Jan-2016", 2, 21000)])
    out = df_mod.load(str(tmp_path))
    assert list(out.columns) == ["date", "period", "y"]
    assert out["date"].iloc[0] == pd.Timestamp("2016-01-01")


def test_load_removes_duplicates_and_non_positive_demand(tmp_path):
    write_csv(tmp_path / "a.csv", [
        ("01-Jan-2016", 1, 20000),
        ("01-Jan-2016", 1, 20000),   ##duplicate
        ("01-Jan-2016", 2, 0),       ##not a real value
        ("01-Jan-2016", 3, -5),      ##not a real value
    ])
    out = df_mod.load(str(tmp_path))
    assert len(out) == 1


def test_load_drops_years_before_first_year(tmp_path):
    write_csv(tmp_path / "a.csv", [("01-Jan-2005", 1, 20000), ("01-Jan-2016", 1, 20000)])
    out = df_mod.load(str(tmp_path))
    assert out["date"].dt.year.min() >= df_mod.first_year


def test_load_handles_lowercase_and_padded_column_names(tmp_path):
    write_csv(tmp_path / "a.csv", [("01-Jan-2016", 1, 20000)],
              columns=(" settlement_date", "Settlement_Period", "nd "))
    assert len(df_mod.load(str(tmp_path))) == 1


def test_load_skips_files_with_missing_columns(tmp_path, capsys):
    write_csv(tmp_path / "good.csv", [("01-Jan-2016", 1, 20000)])
    write_csv(tmp_path / "bad.csv", [("01-Jan-2016", 1, 20000)],
              columns=("SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "OTHER"))
    out = df_mod.load(str(tmp_path))
    assert len(out) == 1
    assert "Skipping bad.csv" in capsys.readouterr().out


def test_load_exits_with_clear_message_when_folder_has_no_csv(tmp_path):
    with pytest.raises(SystemExit):
        df_mod.load(str(tmp_path))


def test_load_sorts_by_date_and_period(tmp_path):
    write_csv(tmp_path / "a.csv", [("02-Jan-2016", 1, 1), ("01-Jan-2016", 2, 2), ("01-Jan-2016", 1, 3)])
    out = df_mod.load(str(tmp_path))
    assert out[["date", "period"]].apply(tuple, axis=1).tolist() == sorted(
        out[["date", "period"]].apply(tuple, axis=1).tolist())


##Model

def test_fit_is_repeatable_with_fixed_seed():
    feats = df_mod.add_features(make_frame(days=40))
    m1, m2 = df_mod.fit(feats), df_mod.fit(feats)
    np.testing.assert_allclose(m1.predict(feats[df_mod.features]), m2.predict(feats[df_mod.features]))


def test_model_is_not_wildly_worse_than_the_yesterday_baseline_on_learnable_data():
    feats = df_mod.add_features(make_frame(days=120, seed=3))
    train, test = feats.iloc[:-2000], feats.iloc[-2000:]
    model = df_mod.fit(train)
    model_err = df_mod.wape(test["y"], model.predict(test[df_mod.features]))
    baseline_err = df_mod.wape(test["y"], test["lag1"])
    assert model_err < baseline_err * 1.5   # sanity bound: model is not wildly worse


##End-to-end

def test_main_writes_results_on_small_made_up_data(tmp_path, monkeypatch):
    monkeypatch.setattr(df_mod, "first_year", 2016)
    monkeypatch.setattr(df_mod, "train_end", 2018)
    data, out = tmp_path / "data", tmp_path / "results"
    data.mkdir()
    for year in range(2016, 2021):
        frame = make_frame(start=f"{year}-01-01", days=365 if year % 4 else 366)
        rows = [(d.strftime("%d-%b-%Y"), p, y) for d, p, y in frame.itertuples(index=False)]
        write_csv(data / f"{year}.csv", rows)
    df_mod.main(str(data), str(out))
    res = pd.read_csv(out / "wape_by_year.csv")
    assert list(res["year"]) == [2019, 2020]
    assert {"gbm_frozen", "gbm_retrained_yearly", "same_time_yesterday"} <= set(res.columns)
    assert (out / "error_by_year.png").exists()
    assert (out / "mean_demand_by_year.png").exists()
