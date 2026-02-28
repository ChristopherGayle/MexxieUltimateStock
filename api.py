from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import asyncpg
import os
import re
import aiohttp
from datetime import datetime, timedelta, timezone
from bisect import bisect_left, bisect_right

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

QUESTDB_HOST = os.getenv("QUESTDB_HOST", "localhost")
QUESTDB_PORT = os.getenv("QUESTDB_PORT", 8812)
DB_POOL = None
MARKET_VAL_BASE = {
    "Worldwide": {"current": 27.7, "avg": 22.5, "high": 28.5, "low": 16.0},
    "US": {"current": 34.7, "avg": 24.5, "high": 34.7, "low": 18.0},
    "Europe": {"current": 22.0, "avg": 21.0, "high": 26.0, "low": 16.0},
    "Asia": {"current": 25.8, "avg": 22.0, "high": 29.4, "low": 18.0},
    "Africa": {"current": 16.5, "avg": 15.0, "high": 20.0, "low": 11.0},
}

def _parse_iso_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _to_naive_utc(dt)

def _to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)

def _latest_at_or_before(series, dt):
    if not series:
        return None
    times = [t for t, _ in series]
    idx = bisect_right(times, dt) - 1
    if idx < 0:
        return None
    return series[idx][1]

def _first_at_or_after(series, dt):
    if not series:
        return None
    times = [t for t, _ in series]
    idx = bisect_left(times, dt)
    if idx >= len(series):
        return None
    return series[idx][1]

def _split_into_quantiles(sorted_items, quantiles):
    n = len(sorted_items)
    base = n // quantiles
    rem = n % quantiles
    out = []
    start = 0
    for i in range(quantiles):
        size = base + (1 if i < rem else 0)
        out.append(sorted_items[start:start + size])
        start += size
    return out

def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default

async def _fetch_us_cape_from_multpl() -> float:
    url = "https://www.multpl.com/shiller-pe"
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"multpl_http_{resp.status}")
            html = await resp.text()
    m = re.search(r"Current Shiller PE Ratio:\s*([0-9]+(?:\.[0-9]+)?)", html, re.IGNORECASE)
    if not m:
        raise RuntimeError("multpl_parse_failed")
    return float(m.group(1))

@app.on_event("startup")
async def startup_db_pool():
    global DB_POOL
    DB_POOL = await asyncpg.create_pool(
        host=QUESTDB_HOST,
        port=QUESTDB_PORT,
        user="admin",
        password="quest",
        database="qdb",
        min_size=1,
        max_size=10
    )

@app.on_event("shutdown")
async def shutdown_db_pool():
    global DB_POOL
    if DB_POOL is not None:
        await DB_POOL.close()
        DB_POOL = None

@app.get("/api/stocks")
async def get_stocks(region: str = "Worldwide", sector: str = "All", limit: int = 20):
    if DB_POOL is None:
        await startup_db_pool()

    async with DB_POOL.acquire() as conn:
        # Build filters with parameters
        filters = []
        params = [limit]
        param_index = 2
        
        if region != "Worldwide":
            filters.append(f"s.region = ${param_index}")
            params.append(region)
            param_index += 1
        if sector != "All":
            filters.append(f"s.sector = ${param_index}")
            params.append(sector)
            param_index += 1
        
        where_clause = " AND ".join(filters) if filters else "TRUE"

        # Join latest row per symbol to avoid duplicates and stale rows.
        query = f"""
            WITH latest_stocks AS (
                SELECT symbol, name, sector, region
                FROM stocks
                LATEST ON update_time PARTITION BY symbol
            ),
            latest_factors AS (
                SELECT symbol, composite_score, wk52, vcp, greenblatt, tweedy, munger, simons,
                       piotroski, altman, rsi, ev_ebit, sh_yield, rev_growth, short_int,
                       sector_value, insider_conf, leverage, accruals, asset_growth
                FROM factor_scores
                LATEST ON timestamp PARTITION BY symbol
            ),
            latest_prices AS (
                SELECT symbol, close
                FROM daily_prices
                LATEST ON timestamp PARTITION BY symbol
            )
            SELECT
                s.symbol,
                s.name,
                s.sector,
                s.region,
                f.composite_score,
                f.wk52,
                f.vcp,
                f.greenblatt,
                f.tweedy,
                f.munger,
                f.simons,
                f.piotroski,
                f.altman,
                f.rsi,
                f.ev_ebit,
                f.sh_yield,
                f.rev_growth,
                f.short_int,
                f.sector_value,
                f.insider_conf,
                f.leverage,
                f.accruals,
                f.asset_growth,
                p.close as price
            FROM latest_stocks s
            JOIN latest_factors f ON s.symbol = f.symbol
            LEFT JOIN latest_prices p ON s.symbol = p.symbol
            WHERE {where_clause}
            ORDER BY f.composite_score DESC
            LIMIT $1
        """
        rows = await conn.fetch(query, *params)

    return [
        {
            "symbol": r["symbol"],
            "name": r["name"],
            "sector": r["sector"],
            "region": r["region"],
            "price": float(r["price"]) if r["price"] else 0.0,
            "change": 0.0,
            "score": float(r["composite_score"]) * 100,
            "wk52": float(r["wk52"]),
            "vcp": float(r["vcp"]) * 100,
            "greenblatt": float(r["greenblatt"]),
            "tweedy": float(r["tweedy"]),
            "munger": float(r["munger"]),
            "simons": float(r["simons"]),
            "piotroski": float(r["piotroski"]),
            "altman": float(r["altman"]),
            "rsi": float(r["rsi"]),
            "ev_ebit": float(r["ev_ebit"]),
            "sh_yield": float(r["sh_yield"]),
            "rev_growth": float(r["rev_growth"]),
            "short_int": float(r["short_int"]),
            "sector_value": float(r["sector_value"]),
            "insider_conf": float(r["insider_conf"]),
            "leverage": float(r["leverage"]),
            "accruals": float(r["accruals"]) if r["accruals"] is not None else 0.5,
            "asset_growth": float(r["asset_growth"]) if r["asset_growth"] is not None else 0.5
        }
        for r in rows
    ]

@app.get("/api/cape")
async def get_cape_values():
    # Start from stable per-region bands used by auto-regime thresholds.
    out = {k: dict(v) for k, v in MARKET_VAL_BASE.items()}
    source = {
        "US": "static",
        "Worldwide": "static",
        "Europe": "static",
        "Asia": "static",
        "Africa": "static",
    }

    # Live US CAPE from Multpl.
    try:
        us_live = await _fetch_us_cape_from_multpl()
        out["US"]["current"] = us_live
        source["US"] = "multpl_live"
    except Exception:
        pass

    # Optional operator overrides for regional CAPE if available from internal sources.
    # Example: CAPE_EUROPE=24.1 CAPE_ASIA=21.9 CAPE_AFRICA=14.2
    out["Europe"]["current"] = _env_float("CAPE_EUROPE", out["Europe"]["current"])
    out["Asia"]["current"] = _env_float("CAPE_ASIA", out["Asia"]["current"])
    out["Africa"]["current"] = _env_float("CAPE_AFRICA", out["Africa"]["current"])
    source["Europe"] = "env_override" if os.getenv("CAPE_EUROPE") else source["Europe"]
    source["Asia"] = "env_override" if os.getenv("CAPE_ASIA") else source["Asia"]
    source["Africa"] = "env_override" if os.getenv("CAPE_AFRICA") else source["Africa"]

    # Worldwide uses env override if provided, else average of regional currents.
    if os.getenv("CAPE_WORLDWIDE"):
        out["Worldwide"]["current"] = _env_float("CAPE_WORLDWIDE", out["Worldwide"]["current"])
        source["Worldwide"] = "env_override"
    else:
        regional_vals = [out["US"]["current"], out["Europe"]["current"], out["Asia"]["current"], out["Africa"]["current"]]
        out["Worldwide"]["current"] = sum(regional_vals) / len(regional_vals)
        source["Worldwide"] = "derived_avg"

    return {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "values": out,
        "source": source,
        "notes": "US is live from Multpl when available; other regions use env overrides or configured fallbacks."
    }

@app.get("/api/backtest")
async def get_backtest(
    region: str = "Worldwide",
    sector: str = "All",
    start_date: str = Query(default=None, description="ISO date/datetime, e.g. 2025-01-01"),
    end_date: str = Query(default=None, description="ISO date/datetime, e.g. 2025-12-31"),
    horizon_days: int = Query(default=63, ge=5, le=252),
    rebalance_days: int = Query(default=21, ge=5, le=63),
    quantiles: int = Query(default=5, ge=2, le=10),
    min_universe: int = Query(default=30, ge=10, le=5000),
):
    if DB_POOL is None:
        await startup_db_pool()

    now = datetime.utcnow()
    start_dt = _parse_iso_datetime(start_date) if start_date else now - timedelta(days=365)
    end_dt = _parse_iso_datetime(end_date) if end_date else now
    if start_dt >= end_dt:
        return {"error": "start_date must be earlier than end_date"}

    factor_start = start_dt - timedelta(days=120)
    price_end = end_dt + timedelta(days=horizon_days + 10)

    factors_params = [factor_start, end_dt]
    prices_params = [start_dt, price_end]
    factors_filters = []
    prices_filters = []

    if region != "Worldwide":
        factors_filters.append(f"s.region = ${len(factors_params) + 1}")
        factors_params.append(region)
        prices_filters.append(f"s.region = ${len(prices_params) + 1}")
        prices_params.append(region)
    if sector != "All":
        factors_filters.append(f"s.sector = ${len(factors_params) + 1}")
        factors_params.append(sector)
        prices_filters.append(f"s.sector = ${len(prices_params) + 1}")
        prices_params.append(sector)

    factors_where_clause = " AND ".join(factors_filters) if factors_filters else "TRUE"
    prices_where_clause = " AND ".join(prices_filters) if prices_filters else "TRUE"

    factors_query = f"""
        WITH latest_stocks AS (
            SELECT symbol, sector, region
            FROM stocks
            LATEST ON update_time PARTITION BY symbol
        )
        SELECT
            f.symbol,
            f.timestamp AS ts,
            f.composite_score AS score
        FROM factor_scores f
        JOIN latest_stocks s ON s.symbol = f.symbol
        WHERE f.timestamp BETWEEN $1 AND $2
          AND {factors_where_clause}
        ORDER BY f.symbol, f.timestamp
    """

    prices_query = f"""
        WITH latest_stocks AS (
            SELECT symbol, sector, region
            FROM stocks
            LATEST ON update_time PARTITION BY symbol
        )
        SELECT
            p.symbol,
            p.timestamp AS ts,
            p.close AS close
        FROM daily_prices p
        JOIN latest_stocks s ON s.symbol = p.symbol
        WHERE p.timestamp BETWEEN $1 AND $2
          AND {prices_where_clause}
        ORDER BY p.symbol, p.timestamp
    """

    try:
        async with DB_POOL.acquire() as conn:
            factor_rows = await conn.fetch(factors_query, *factors_params)
            price_rows = await conn.fetch(prices_query, *prices_params)

        factors = {}
        prices = {}

        for r in factor_rows:
            symbol = r["symbol"]
            score = float(r["score"]) if r["score"] is not None else None
            if score is None:
                continue
            ts = _to_naive_utc(r["ts"])
            factors.setdefault(symbol, []).append((ts, score))

        for r in price_rows:
            symbol = r["symbol"]
            close = float(r["close"]) if r["close"] is not None else None
            if close is None or close <= 0:
                continue
            ts = _to_naive_utc(r["ts"])
            prices.setdefault(symbol, []).append((ts, close))

        symbols = sorted(set(factors.keys()) & set(prices.keys()))
        if len(symbols) < min_universe:
            return {
                "start_date": start_dt.isoformat(),
                "end_date": end_dt.isoformat(),
                "horizon_days": horizon_days,
                "rebalance_days": rebalance_days,
                "quantiles": quantiles,
                "num_symbols": len(symbols),
                "num_rebalances": 0,
                "message": f"Not enough symbols for backtest (need >= {min_universe})",
                "rebalances": [],
                "aggregate": {}
            }

        rebalance_dates = []
        cursor = start_dt
        last_rebalance = end_dt - timedelta(days=horizon_days)
        while cursor <= last_rebalance:
            rebalance_dates.append(cursor)
            cursor += timedelta(days=rebalance_days)

        rebalance_results = []
        per_quantile_returns = {i: [] for i in range(1, quantiles + 1)}

        for rebalance_dt in rebalance_dates:
            horizon_dt = rebalance_dt + timedelta(days=horizon_days)
            scored = []

            for symbol in symbols:
                score = _latest_at_or_before(factors[symbol], rebalance_dt)
                p0 = _latest_at_or_before(prices[symbol], rebalance_dt)
                p1 = _first_at_or_after(prices[symbol], horizon_dt)
                if score is None or p0 is None or p1 is None or p0 <= 0:
                    continue
                ret = (p1 / p0) - 1.0
                scored.append((symbol, score, ret))

            if len(scored) < min_universe:
                continue

            scored.sort(key=lambda x: x[1], reverse=True)
            buckets = _split_into_quantiles(scored, quantiles)

            q_returns = []
            q_counts = []
            for idx, bucket in enumerate(buckets, start=1):
                if not bucket:
                    q_ret = 0.0
                    q_n = 0
                else:
                    q_n = len(bucket)
                    q_ret = sum(x[2] for x in bucket) / q_n
                q_returns.append(q_ret)
                q_counts.append(q_n)
                per_quantile_returns[idx].append(q_ret)

            rebalance_results.append({
                "date": rebalance_dt.isoformat(),
                "forward_to": horizon_dt.isoformat(),
                "universe_size": len(scored),
                "quantile_returns": q_returns,
                "quantile_counts": q_counts,
                "top_minus_bottom": q_returns[0] - q_returns[-1]
            })

        def _mean(vals):
            return (sum(vals) / len(vals)) if vals else 0.0

        def _compound(vals):
            total = 1.0
            for v in vals:
                total *= (1.0 + v)
            return total - 1.0

        aggregate = {
            "mean_return_by_quantile": [_mean(per_quantile_returns[i]) for i in range(1, quantiles + 1)],
            "compound_return_by_quantile": [_compound(per_quantile_returns[i]) for i in range(1, quantiles + 1)],
            "mean_top_minus_bottom": _mean([r["top_minus_bottom"] for r in rebalance_results]),
        }

        message = None
        if len(rebalance_results) == 0:
            message = (
                "No valid rebalance windows found. Try a shorter horizon/rebalance, "
                "a more recent date range, or ensure more historical daily_prices coverage."
            )

        return {
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "horizon_days": horizon_days,
            "rebalance_days": rebalance_days,
            "quantiles": quantiles,
            "num_symbols": len(symbols),
            "num_rebalances": len(rebalance_results),
            "message": message,
            "rebalances": rebalance_results,
            "aggregate": aggregate,
        }
    except Exception as exc:
        return {"error": f"backtest_failed: {type(exc).__name__}: {exc}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
