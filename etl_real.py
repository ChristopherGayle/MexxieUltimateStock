import asyncio
import aiohttp
import asyncpg
import numpy as np
from datetime import datetime
import os
import time
from collections import deque

FINNHUB_KEY = os.getenv("FINNHUB_KEY", "d6ebj19r01qloir5vun0d6ebj19r01qloir5vung")
QUESTDB_HOST = os.getenv("QUESTDB_HOST", "localhost")
QUESTDB_PORT = int(os.getenv("QUESTDB_PORT", "8812"))
SYMBOLS_FILE = "symbols.txt"
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "1000"))
FINNHUB_CALLS_PER_MINUTE = int(os.getenv("FINNHUB_CALLS_PER_MINUTE", "50"))
CONCURRENCY = int(os.getenv("ETL_CONCURRENCY", "20"))
BATCH_SIZE = int(os.getenv("ETL_BATCH_SIZE", "40"))

class RateLimiter:
    def __init__(self, calls_per_minute=50, period_seconds=60.0):
        self.calls_per_minute = max(1, calls_per_minute)
        self.period_seconds = period_seconds
        self.call_times = deque()
        self.lock = asyncio.Lock()
        
    async def wait(self):
        while True:
            async with self.lock:
                now = time.monotonic()
                cutoff = now - self.period_seconds
                while self.call_times and self.call_times[0] <= cutoff:
                    self.call_times.popleft()

                if len(self.call_times) < self.calls_per_minute:
                    self.call_times.append(now)
                    return

                sleep_for = max(0.01, self.period_seconds - (now - self.call_times[0]) + 0.01)
            await asyncio.sleep(sleep_for)

rate_limiter = RateLimiter(FINNHUB_CALLS_PER_MINUTE)

def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0):
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def get_first_metric(m, keys, default=0.0):
    for key in keys:
        if key in m and m.get(key) is not None:
            return safe_float(m.get(key), default)
    return default

async def fetch_quote(session, symbol):
    await rate_limiter.wait()
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_KEY}"
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            return {}
    except Exception:
        return {}

async def fetch_metrics(session, symbol):
    await rate_limiter.wait()
    url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={FINNHUB_KEY}"
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            return {}
    except Exception:
        return {}

async def fetch_insider(session, symbol):
    await rate_limiter.wait()
    url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={symbol}&token={FINNHUB_KEY}"
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            return {}
    except Exception:
        return {}

def calculate_greenblatt(m, price):
    ebit = safe_float(m.get('ebitTTM'))
    enterprise_value = safe_float(m.get('enterpriseValue'))
    
    if ebit and ebit > 0 and enterprise_value and enterprise_value > 0:
        earnings_yield = ebit / enterprise_value
    else:
        pe = safe_float(m.get('peTTM'))
        if pe and pe > 0:
            earnings_yield = 1.0 / pe
        else:
            earnings_yield = 0.05
    
    roic = safe_float(m.get('roicTTM'))
    if roic and roic > 0:
        roic = roic / 100.0
    else:
        roe = safe_float(m.get('roeTTM'))
        if roe and roe > 0:
            roic = (roe / 100.0) * 0.8
        else:
            roic = 0.10
    
    earnings_score = min(1.0, max(0.0, (earnings_yield - 0.02) / 0.15))
    roic_score = min(1.0, max(0.0, (roic - 0.05) / 0.25))
    
    return (earnings_score * 0.5 + roic_score * 0.5)

def calculate_tweedy(m, insider_data):
    score = 0.0
    count = 0
    
    pb = safe_float(m.get('pbAnnual'))
    if pb and pb > 0:
        pb_score = 1.0 - min(1.0, (pb - 0.5) / 4.0)
        score += pb_score
        count += 1
    
    div_yield = safe_float(m.get('dividendYieldIndicatedAnnual'))
    if div_yield and div_yield > 0:
        div_score = min(1.0, div_yield / 0.06)
        score += div_score
        count += 1
    
    fcf_yield = safe_float(m.get('freeCashFlowYieldTTM'))
    if fcf_yield and fcf_yield > 0:
        fcf_score = min(1.0, fcf_yield / 0.10)
        score += fcf_score
        count += 1
    
    if insider_data and 'data' in insider_data:
        insider_net = 0
        for tx in insider_data['data'][:10]:
            insider_net += safe_float(tx.get('transactionValue'), 0)
        if insider_net > 0:
            score += 0.2
            count += 1
        elif insider_net < 0:
            score -= 0.1
            count += 1
    
    if count > 0:
        return score / count
    return 0.5

def calculate_munger(m):
    score = 0.0
    count = 0
    
    gross_margin = safe_float(m.get('grossMarginTTM'))
    if gross_margin and gross_margin > 0:
        gm_score = min(1.0, gross_margin / 0.7)
        score += gm_score
        count += 1
    
    op_margin = safe_float(m.get('operatingMarginTTM'))
    if op_margin and op_margin > 0:
        op_score = min(1.0, op_margin / 0.4)
        score += op_score
        count += 1
    
    roe = safe_float(m.get('roeTTM'))
    if roe and roe > 0:
        roe_score = min(1.0, (roe / 100.0) / 0.30)
        score += roe_score
        count += 1
    
    if count > 0:
        return min(0.95, score / count)
    return 0.5

def calculate_piotroski(m):
    fscore = safe_int(m.get('piotroskiScore'))
    if fscore and fscore > 0:
        return fscore / 9.0
    
    score = 0.5
    
    roe = safe_float(m.get('roeTTM'))
    if roe and roe > 0:
        score += 0.1
    
    pb = safe_float(m.get('pbAnnual'))
    if pb and pb < 3:
        score += 0.1
    
    op_margin = safe_float(m.get('operatingMarginTTM'))
    if op_margin and op_margin > 0.15:
        score += 0.1
    
    return min(1.0, score)

def calculate_druckenmiller_region_score(region):
    macro = MACRO_DATA.get(region, MACRO_DATA['US'])
    
    score = 0.5
    
    stance = macro.get('central_bank_stance', 'neutral')
    if stance == 'accommodative':
        score += 0.15
    elif stance == 'neutral':
        score += 0.05
    
    rate_trend = macro.get('interest_rate_trend', 'stable')
    if rate_trend == 'down':
        score += 0.1
    elif rate_trend == 'stable':
        score += 0.05
    
    growth = macro.get('economic_growth', 2.0)
    if growth > 3.0:
        score += 0.15
    elif growth > 2.0:
        score += 0.1
    elif growth > 1.0:
        score += 0.05
    
    inflation = macro.get('inflation', 2.0)
    if 1.5 < inflation < 3.0:
        score += 0.1
    elif inflation > 5.0:
        score -= 0.1
    
    sentiment = macro.get('market_sentiment', 0.5)
    score += (sentiment - 0.5) * 0.2
    
    liquidity = macro.get('liquidity', 0.5)
    score += (liquidity - 0.5) * 0.2
    
    return min(1.0, max(0.0, score))

def calculate_druckenmiller_stock_score(m, sector, region, wk52):
    score = 0.5
    macro = MACRO_DATA.get(region, MACRO_DATA['US'])
    
    sector_leaders = macro.get('sector_leaders', [])
    if sector in sector_leaders:
        score += 0.2
    elif any(leader in sector for leader in sector_leaders):
        score += 0.1
    
    rev_growth = safe_float(m.get('revenueGrowthQoq', 0)) * 4
    if rev_growth > 0.20:
        score += 0.15
    elif rev_growth > 0.10:
        score += 0.1
    elif rev_growth > 0.05:
        score += 0.05
    
    roic = safe_float(m.get('roicTTM', 0)) / 100.0
    if roic > 0.20:
        score += 0.15
    elif roic > 0.12:
        score += 0.1
    elif roic > 0.08:
        score += 0.05
    
    if wk52 > 0.90:
        score += 0.15
    elif wk52 > 0.80:
        score += 0.1
    elif wk52 > 0.70:
        score += 0.05
    
    risk_appetite = macro.get('risk_appetite', 0.5)
    beta = safe_float(m.get('beta'), 1.0)
    beta_score = 1.0 - abs(beta - risk_appetite * 1.5) / 2.0
    score += beta_score * 0.1
    
    return min(1.0, max(0.0, score))

def calculate_ev_ebit(m):
    ev_ebit = safe_float(m.get('enterpriseValueOverEbitTTM'))
    if ev_ebit and ev_ebit > 0:
        if ev_ebit < 8: return 1.0
        elif ev_ebit < 12: return 0.9
        elif ev_ebit < 16: return 0.7
        elif ev_ebit < 20: return 0.5
        elif ev_ebit < 25: return 0.3
        else: return 0.2
    
    pe = safe_float(m.get('peTTM'))
    if pe and pe > 0:
        if pe < 10: return 0.9
        elif pe < 15: return 0.7
        elif pe < 20: return 0.5
        elif pe < 25: return 0.3
        else: return 0.2
    return 0.5

def calculate_sh_yield(m):
    div_yield = safe_float(m.get('dividendYieldIndicatedAnnual'), 0)
    buyback = safe_float(m.get('buybackYield'), 0)
    total_yield = div_yield + buyback
    if total_yield > 0.08: return 1.0
    elif total_yield > 0.06: return 0.9
    elif total_yield > 0.04: return 0.7
    elif total_yield > 0.02: return 0.5
    elif total_yield > 0.01: return 0.3
    else: return 0.2

def calculate_rev_growth(m):
    rev_growth = safe_float(m.get('revenueGrowthQoq'))
    if rev_growth:
        annualized = rev_growth * 4
        if annualized > 0.25: return 1.0
        elif annualized > 0.20: return 0.9
        elif annualized > 0.15: return 0.8
        elif annualized > 0.10: return 0.7
        elif annualized > 0.05: return 0.5
        elif annualized > 0.0: return 0.3
        else: return 0.1
    return 0.5

def calculate_short_int(m):
    short_pct = safe_float(m.get('shortPercentOfFloat'))
    if short_pct:
        if short_pct < 0.02: return 0.4
        elif short_pct < 0.05: return 0.7
        elif short_pct < 0.08: return 0.9
        elif short_pct < 0.12: return 0.8
        elif short_pct < 0.15: return 0.6
        elif short_pct < 0.20: return 0.4
        else: return 0.2
    return 0.5

def calculate_altman_proxy(m, leverage_score):
    """Proxy for distress quality when full Altman inputs are unavailable."""
    current_ratio = get_first_metric(m, ["currentRatioAnnual", "currentRatioTTM"], 1.5)
    operating_margin = safe_float(m.get("operatingMarginTTM"), 0.12)
    roa = get_first_metric(m, ["roaTTM", "roaRfy"], 0.06) / 100.0

    liquidity_score = min(1.0, max(0.0, (current_ratio - 0.8) / 2.0))
    profitability_score = min(1.0, max(0.0, (operating_margin - 0.02) / 0.25))
    roa_score = min(1.0, max(0.0, (roa - 0.01) / 0.10))
    return min(1.0, max(0.0, 0.35 * liquidity_score + 0.25 * profitability_score + 0.20 * roa_score + 0.20 * leverage_score))

def calculate_leverage(m):
    debt_to_equity = get_first_metric(
        m,
        ["totalDebtToEquityQuarterly", "totalDebtToEquityAnnual", "longTermDebtEquityAnnual"],
        100.0
    )
    interest_coverage = get_first_metric(m, ["interestCoverageTTM", "interestCoverageAnnual"], 3.0)

    if debt_to_equity <= 30:
        debt_score = 1.0
    elif debt_to_equity <= 60:
        debt_score = 0.85
    elif debt_to_equity <= 100:
        debt_score = 0.65
    elif debt_to_equity <= 180:
        debt_score = 0.40
    else:
        debt_score = 0.20

    if interest_coverage >= 10:
        ic_score = 1.0
    elif interest_coverage >= 6:
        ic_score = 0.85
    elif interest_coverage >= 3:
        ic_score = 0.65
    elif interest_coverage >= 1.5:
        ic_score = 0.40
    else:
        ic_score = 0.20
    return min(1.0, max(0.0, 0.65 * debt_score + 0.35 * ic_score))

def calculate_insider_confidence(insider_data):
    txs = insider_data.get("data", []) if isinstance(insider_data, dict) else []
    if not txs:
        return 0.5

    buy_value = 0.0
    sell_value = 0.0
    buy_count = 0
    sell_count = 0

    for tx in txs[:30]:
        val = abs(safe_float(tx.get("transactionValue"), 0.0))
        change = safe_float(tx.get("share"), 0.0)
        tx_type = str(tx.get("transactionCode", "")).upper()
        if change > 0 or tx_type in {"P", "A"}:
            buy_value += val
            buy_count += 1
        elif change < 0 or tx_type in {"S", "D"}:
            sell_value += val
            sell_count += 1

    total = buy_value + sell_value
    if total <= 0:
        return 0.5

    value_bias = (buy_value - sell_value) / total
    count_total = buy_count + sell_count
    count_bias = ((buy_count - sell_count) / count_total) if count_total > 0 else 0.0
    score = 0.5 + 0.35 * value_bias + 0.15 * count_bias
    return min(1.0, max(0.0, score))

def calculate_rsi_score_from_closes(closes, period=14):
    if len(closes) < period + 1:
        return 0.5

    deltas = np.diff(np.array(closes[-(period + 1):], dtype=float))
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()

    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # Prefer healthy trend (roughly 45-65) over extremes.
    if 45 <= rsi <= 65:
        return 1.0
    if 35 <= rsi < 45 or 65 < rsi <= 75:
        return 0.7
    if 25 <= rsi < 35 or 75 < rsi <= 85:
        return 0.4
    return 0.2

def calculate_vcp_score_from_closes(closes):
    if len(closes) < 25:
        return 0.5

    arr = np.array(closes[-25:], dtype=float)
    rets = np.diff(arr) / np.where(arr[:-1] == 0, 1.0, arr[:-1])
    if len(rets) < 20:
        return 0.5

    vol20 = np.std(rets[-20:])
    vol10 = np.std(rets[-10:])
    vol5 = np.std(rets[-5:])
    if vol20 <= 0:
        return 0.5

    contraction = 1.0 if (vol5 < vol10 < vol20) else 0.4
    compression_ratio = max(0.0, min(1.0, 1.0 - (vol5 / vol20)))
    breakout_proximity = max(0.0, min(1.0, arr[-1] / max(arr[-25:]) if max(arr[-25:]) > 0 else 0.0))
    return min(1.0, max(0.0, 0.45 * contraction + 0.35 * compression_ratio + 0.20 * breakout_proximity))

def calculate_accruals_score(m):
    # Lower accrual intensity is typically preferred.
    cfo_ps = get_first_metric(m, ["cashFlowPerShareTTM", "cashFlowPerShareAnnual"], 0.0)
    eps = get_first_metric(m, ["epsTTM", "epsBasicExclExtraItemsTTM"], 0.0)
    assets_per_share = get_first_metric(m, ["bookValuePerShareAnnual", "bookValuePerShareQuarterly"], 0.0)
    if assets_per_share <= 0:
        return 0.5

    accrual_ratio = (eps - cfo_ps) / assets_per_share
    if accrual_ratio < -0.02:
        return 1.0
    if accrual_ratio < 0.02:
        return 0.8
    if accrual_ratio < 0.05:
        return 0.6
    if accrual_ratio < 0.10:
        return 0.35
    return 0.15

def calculate_asset_growth_score(m):
    growth = get_first_metric(
        m,
        ["assetGrowthAnnual", "totalAssetsCagr5Y", "totalAssetsCagr10Y"],
        0.0
    )
    if growth > 1.5:  # likely percentage points already in whole numbers
        growth = growth / 100.0

    # Asset growth anomaly: lower/slower asset growth tends to outperform.
    if growth < 0.00:
        return 0.9
    if growth < 0.05:
        return 1.0
    if growth < 0.10:
        return 0.75
    if growth < 0.20:
        return 0.45
    return 0.20

def calculate_sector_value(m):
    return 0.6

# Region-specific macro data
MACRO_DATA = {
    'US': {
        'central_bank_stance': 'neutral',
        'interest_rate_trend': 'up',
        'economic_growth': 2.5,
        'inflation': 3.2,
        'market_sentiment': 0.65,
        'sector_leaders': ['Technology', 'Communication', 'Healthcare'],
        'risk_appetite': 0.7,
        'liquidity': 0.6
    },
    'Europe': {
        'central_bank_stance': 'accommodative',
        'interest_rate_trend': 'stable',
        'economic_growth': 1.2,
        'inflation': 2.8,
        'market_sentiment': 0.55,
        'sector_leaders': ['Consumer', 'Industrials', 'Healthcare'],
        'risk_appetite': 0.5,
        'liquidity': 0.7
    },
    'Asia': {
        'central_bank_stance': 'accommodative',
        'interest_rate_trend': 'down',
        'economic_growth': 4.5,
        'inflation': 2.1,
        'market_sentiment': 0.75,
        'sector_leaders': ['Technology', 'Consumer', 'Industrials'],
        'risk_appetite': 0.8,
        'liquidity': 0.8
    },
    'Africa': {
        'central_bank_stance': 'tightening',
        'interest_rate_trend': 'up',
        'economic_growth': 3.0,
        'inflation': 6.5,
        'market_sentiment': 0.45,
        'sector_leaders': ['Materials', 'Energy', 'Telecom'],
        'risk_appetite': 0.4,
        'liquidity': 0.3
    }
}

# Region-specific factor weights
REGION_WEIGHTS = {
    'US': {
        'greenblatt': 0.12,
        'tweedy': 0.08,
        'munger': 0.14,
        'simons': 0.14,
        'piotroski': 0.10,
        'druckenmiller': 0.14,
        'ev_ebit': 0.05,
        'sh_yield': 0.05,
        'rev_growth': 0.05,
        'short_int': 0.03,
        'sector_value': 0.02,
        'accruals': 0.04,
        'asset_growth': 0.04
    },
    'Europe': {
        'greenblatt': 0.15,
        'tweedy': 0.12,
        'munger': 0.12,
        'simons': 0.10,
        'piotroski': 0.12,
        'druckenmiller': 0.10,
        'ev_ebit': 0.06,
        'sh_yield': 0.06,
        'rev_growth': 0.06,
        'short_int': 0.03,
        'sector_value': 0.02,
        'accruals': 0.04,
        'asset_growth': 0.04
    },
    'Asia': {
        'greenblatt': 0.17,
        'tweedy': 0.14,
        'munger': 0.10,
        'simons': 0.08,
        'piotroski': 0.12,
        'druckenmiller': 0.06,
        'ev_ebit': 0.07,
        'sh_yield': 0.05,
        'rev_growth': 0.08,
        'short_int': 0.04,
        'sector_value': 0.03,
        'accruals': 0.04,
        'asset_growth': 0.04
    },
    'Africa': {
        'greenblatt': 0.20,
        'tweedy': 0.18,
        'munger': 0.08,
        'simons': 0.05,
        'piotroski': 0.12,
        'druckenmiller': 0.03,
        'ev_ebit': 0.08,
        'sh_yield': 0.04,
        'rev_growth': 0.08,
        'short_int': 0.05,
        'sector_value': 0.06,
        'accruals': 0.06,
        'asset_growth': 0.05
    }
}

async def process_symbol(symbol, name, sector, region, session, db_pool, existing_symbols, symbols_lock):
    try:
        quote, metrics, insider = await asyncio.gather(
            fetch_quote(session, symbol),
            fetch_metrics(session, symbol),
            fetch_insider(session, symbol),
            return_exceptions=True
        )
        
        if isinstance(quote, Exception) or not quote:
            return False
        
        price = safe_float(quote.get('c'), 0.0)
        if price == 0.0:
            return False
        
        m = metrics.get('metric', {}) if metrics and not isinstance(metrics, Exception) else {}
        insider_data = insider if insider and not isinstance(insider, Exception) else {}
        
        week52_high = safe_float(m.get('52WeekHigh'), price * 1.2)
        wk52 = price / week52_high if week52_high > 0 else 0.8
        
        greenblatt = calculate_greenblatt(m, price)
        tweedy = calculate_tweedy(m, insider_data)
        munger = calculate_munger(m)
        simons = wk52
        piotroski = calculate_piotroski(m)
        druckenmiller_macro = calculate_druckenmiller_region_score(region)
        druckenmiller_stock = calculate_druckenmiller_stock_score(m, sector, region, wk52)
        druckenmiller = druckenmiller_macro * 0.35 + druckenmiller_stock * 0.65
        
        ev_ebit = calculate_ev_ebit(m)
        sh_yield = calculate_sh_yield(m)
        rev_growth = calculate_rev_growth(m)
        short_int = calculate_short_int(m)
        leverage = calculate_leverage(m)
        insider_conf = calculate_insider_confidence(insider_data)
        altman = calculate_altman_proxy(m, leverage)
        accruals = calculate_accruals_score(m)
        asset_growth = calculate_asset_growth_score(m)
        sector_value = calculate_sector_value(m)
        
        weights = REGION_WEIGHTS.get(region, REGION_WEIGHTS['US'])
        
        composite = (
            greenblatt * weights['greenblatt'] +
            tweedy * weights['tweedy'] +
            munger * weights['munger'] +
            simons * weights['simons'] +
            piotroski * weights['piotroski'] +
            druckenmiller * weights['druckenmiller'] +
            ev_ebit * weights['ev_ebit'] +
            sh_yield * weights['sh_yield'] +
            rev_growth * weights['rev_growth'] +
            short_int * weights['short_int'] +
            sector_value * weights['sector_value'] +
            accruals * weights['accruals'] +
            asset_growth * weights['asset_growth']
        ) / sum(weights.values())
        
        now = datetime.now()
        
        async with db_pool.acquire() as conn:
            price_rows = await conn.fetch(
                """
                SELECT close
                FROM daily_prices
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 40
                """,
                symbol
            )
            historical_closes_desc = [safe_float(r["close"], 0.0) for r in price_rows if safe_float(r["close"], 0.0) > 0]
            historical_closes = list(reversed(historical_closes_desc))
            historical_closes.append(price)
            rsi = calculate_rsi_score_from_closes(historical_closes, period=14)
            vcp = calculate_vcp_score_from_closes(historical_closes)

            # Insert into stocks only if symbol is not already present to avoid duplicates.
            should_insert_stock = False
            async with symbols_lock:
                if symbol not in existing_symbols:
                    existing_symbols.add(symbol)
                    should_insert_stock = True

            if should_insert_stock:
                await conn.execute("""
                    INSERT INTO stocks (symbol, name, sector, region, update_time)
                    VALUES ($1, $2, $3, $4, $5)
                """, symbol, name, sector, region, now)
            
            await conn.execute("""
                INSERT INTO daily_prices (symbol, timestamp, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, symbol, now, price, price, price, price, 1000000)
            
            await conn.execute("""
                INSERT INTO factor_scores (
                    symbol, timestamp, greenblatt, tweedy, munger, simons, 
                    piotroski, altman, rsi, ev_ebit, sh_yield, rev_growth,
                    short_int, wk52, vcp, sector_value, insider_conf, leverage, accruals, asset_growth, composite_score
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                    $13, $14, $15, $16, $17, $18, $19, $20, $21
                )
            """, symbol, now, 
                greenblatt, tweedy, munger, simons,
                piotroski, altman, rsi, ev_ebit, sh_yield, rev_growth,
                short_int, wk52, vcp, sector_value, insider_conf, leverage, accruals, asset_growth, composite)
        
        print(f"    ✅ {symbol}", end="")
        return True
        
    except Exception as e:
        print(f"    ❌ {symbol}: {str(e)}")
        return False

async def dedupe_stocks_table(db_pool):
    print("🧹 Deduplicating stocks table (keeping latest row per symbol)...")
    async with db_pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS stocks_dedup")
        await conn.execute("""
            CREATE TABLE stocks_dedup AS (
                SELECT symbol, name, sector, region, update_time
                FROM stocks
                LATEST ON update_time PARTITION BY symbol
            )
        """)
        await conn.execute("TRUNCATE TABLE stocks")
        await conn.execute("""
            INSERT INTO stocks (symbol, name, sector, region, update_time)
            SELECT symbol, name, sector, region, update_time
            FROM stocks_dedup
        """)
        await conn.execute("DROP TABLE stocks_dedup")
    print("✅ Stocks table deduplicated")

async def ensure_factor_scores_columns(db_pool):
    print("🧱 Ensuring factor_scores has accruals and asset_growth columns...")
    statements = [
        "ALTER TABLE factor_scores ADD COLUMN accruals DOUBLE",
        "ALTER TABLE factor_scores ADD COLUMN asset_growth DOUBLE",
    ]
    async with db_pool.acquire() as conn:
        for stmt in statements:
            try:
                await conn.execute(stmt)
                print(f"  ✅ {stmt}")
            except Exception as exc:
                msg = str(exc).lower()
                if "exists" in msg or "duplicate" in msg:
                    print(f"  ℹ️ Column already exists for: {stmt}")
                else:
                    print(f"  ⚠️ Could not run '{stmt}': {exc}")

async def load_existing_symbols(db_pool):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT symbol FROM stocks")
    return {r["symbol"] for r in rows}

async def process_symbol_with_semaphore(item, session, db_pool, semaphore, existing_symbols, symbols_lock):
    sym, name, sector, region = item
    async with semaphore:
        return await process_symbol(sym, name, sector, region, session, db_pool, existing_symbols, symbols_lock)

async def process_batch(batch, batch_num, total_batches, session, db_pool, existing_symbols, symbols_lock):
    print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} stocks)")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    tasks = [
        process_symbol_with_semaphore(item, session, db_pool, semaphore, existing_symbols, symbols_lock)
        for item in batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success_count = sum(1 for r in results if r is True)
    
    print(f"\n  ✅ Batch complete: {success_count}/{len(batch)} successful")
    return success_count

async def main():
    # Read symbols and remove duplicate symbol lines while preserving order.
    symbols = []
    seen_symbols = set()
    with open(SYMBOLS_FILE) as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 4:
                symbol_row = parts[:4]
                symbol = symbol_row[0]
                if symbol not in seen_symbols:
                    seen_symbols.add(symbol)
                    symbols.append(symbol_row)

    symbols = symbols[:MAX_SYMBOLS]
    
    print(f"✅ Loaded {len(symbols)} unique symbols (MAX_SYMBOLS={MAX_SYMBOLS})")
    print(f"\nStarting ETL for {len(symbols)} symbols...")
    print(f"⚙️ Finnhub throttle: {FINNHUB_CALLS_PER_MINUTE} calls/min")
    print(f"⚙️ Concurrency: {CONCURRENCY} (batch size: {BATCH_SIZE})")
    est_minutes = (len(symbols) * 3) / max(1, FINNHUB_CALLS_PER_MINUTE)
    print(f"⏱️ Estimated API time floor: {est_minutes:.1f} minutes (3 calls/symbol)")
    
    batch_size = BATCH_SIZE
    success_count = 0
    total_batches = (len(symbols) + batch_size - 1) // batch_size

    db_pool = await asyncpg.create_pool(
        host=QUESTDB_HOST,
        port=QUESTDB_PORT,
        user='admin',
        password='quest',
        database='qdb',
        min_size=1,
        max_size=10
    )

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            await dedupe_stocks_table(db_pool)
            await ensure_factor_scores_columns(db_pool)
            existing_symbols = await load_existing_symbols(db_pool)
            symbols_lock = asyncio.Lock()
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                current_batch = i//batch_size + 1
                
                batch_success = await process_batch(
                    batch, current_batch, total_batches, session, db_pool, existing_symbols, symbols_lock
                )
                success_count += batch_success
                
                pct = min(i + batch_size, len(symbols)) / len(symbols) * 100
                print(f"  📊 Overall progress: {min(i+batch_size, len(symbols))}/{len(symbols)} ({pct:.1f}%)")
        finally:
            await db_pool.close()
    
    print(f"\n✅ ETL Complete! Successfully processed {success_count}/{len(symbols)} symbols")

if __name__ == "__main__":
    asyncio.run(main())
