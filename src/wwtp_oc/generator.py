from __future__ import annotations

import math

import numpy as np
import pandas as pd


def generate_timeseries(
    duration_hours: int = 48,
    dt_minutes: int = 15,
    seed: int | None = 42,
) -> pd.DataFrame:
    if duration_hours <= 0:
        raise ValueError("duration_hours must be > 0")
    if dt_minutes <= 0:
        raise ValueError("dt_minutes must be > 0")

    rng = np.random.default_rng(seed)
    n_steps = (duration_hours * 60) // dt_minutes
    dt_hours = dt_minutes / 60.0

    time = pd.date_range(
        start="2025-01-01 00:00:00",
        periods=n_steps,
        freq=f"{dt_minutes}min",
    )

    hours = np.arange(n_steps) * dt_hours
    phase = 2 * math.pi * (hours % 24) / 24

    flow_rate = 100 + 10 * np.sin(phase) + rng.normal(0, 2, n_steps)
    flow_rate = np.clip(flow_rate, 1, None)

    aeration_rate = 55 + 5 * np.sin(phase - 0.5) + rng.normal(0, 1, n_steps)
    aeration_rate = np.clip(aeration_rate, 1, None)

    pump_activity = 0.4 * flow_rate
    mixer_activity = np.full(n_steps, 18.0)

    do = np.zeros(n_steps)
    nh4 = np.zeros(n_steps)
    tn = np.zeros(n_steps)

    do[0] = 2.0
    nh4[0] = 6.0
    tn[0] = 15.0

    for i in range(1, n_steps):
        do[i] = do[i - 1] + dt_hours * (0.08 * aeration_rate[i] - 0.03 * flow_rate[i] - 0.3 * do[i - 1])
        do[i] += rng.normal(0, 0.05)
        do[i] = np.clip(do[i], 0.1, 6.0)

        nh4[i] = nh4[i - 1] + dt_hours * (0.02 * flow_rate[i] - 0.6 * do[i] - 0.1 * nh4[i - 1])
        nh4[i] += rng.normal(0, 0.1)
        nh4[i] = np.clip(nh4[i], 0.1, 20.0)

        tn[i] = tn[i - 1] + dt_hours * (0.01 * flow_rate[i] - 0.08 * tn[i - 1])
        tn[i] += rng.normal(0, 0.1)
        tn[i] = np.clip(tn[i], 1.0, 30.0)

    return pd.DataFrame(
        {
            "time": time,
            "do": do,
            "nh4": nh4,
            "tn": tn,
            "flow_rate": flow_rate,
            "aeration_rate": aeration_rate,
            "pump_activity": pump_activity,
            "mixer_activity": mixer_activity,
        }
    )


