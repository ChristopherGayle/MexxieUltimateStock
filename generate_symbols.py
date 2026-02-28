#!/usr/bin/env python3
"""
Generate an expanded symbol list with 300+ stocks from major global indices
"""

# US Stocks (150) - S&P 500 top holdings by market cap
US_SYMBOLS = [
    # Technology (30)
    "AAPL", "MSFT", "GOOGL", "GOOG", "META", "NVDA", "AMD", "INTC", "CSCO", "ORCL",
    "CRM", "ADBE", "ACN", "TXN", "QCOM", "AVGO", "INTU", "NOW", "UBER", "LYFT",
    "SNOW", "PLTR", "NET", "DDOG", "MDB", "ZS", "PANW", "CRWD", "FTNT", "OKTA",
    
    # Consumer (25)
    "AMZN", "TSLA", "NFLX", "DIS", "CMCSA", "NKE", "SBUX", "MCD", "BKNG", "ABNB",
    "LULU", "MAR", "HLT", "YUM", "DASH", "EBAY", "ETSY", "ROST", "TJX", "TGT",
    "COST", "WMT", "HD", "LOW", "KR",
    
    # Healthcare (20)
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "TMO", "DHR", "UNH", "CVS", "CI",
    "AMGN", "GILD", "REGN", "VRTX", "ISRG", "SYK", "BDX", "ZTS", "BMY", "ABT",
    
    # Finance (20)
    "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "BLK", "V", "MA",
    "AXP", "DFS", "COF", "USB", "PNC", "TFC", "MET", "PRU", "AIG", "ALL",
    
    # Energy (15)
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "OXY", "MPC", "PSX", "VLO",
    "KMI", "WMB", "OKE", "HAL", "BKR",
    
    # Industrials (20)
    "GE", "HON", "BA", "CAT", "DE", "MMM", "UPS", "FDX", "LMT", "NOC",
    "RTX", "GD", "TXT", "WM", "RSG", "UNP", "CSX", "NSC", "CHRW", "ODFL",
    
    # Materials (10)
    "LIN", "APD", "SHW", "ECL", "NUE", "FCX", "DOW", "DD", "PPG", "LYB",
    
    # Real Estate (10)
    "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SPG", "WELL", "AVB", "EQR"
]

# European Stocks (75)
EUROPE_SYMBOLS = [
    # UK (20)
    "HSBC", "BP", "SHEL", "AZN", "GSK", "ULVR", "RIO", "BHP", "GLEN", "BATS",
    "DGE", "REL", "LSEG", "PRU", "LLOY", "BARCL", "VOD", "BT", "IMB", "AAL",
    
    # Germany (15)
    "SAP", "SIE", "DTE", "ALV", "BAS", "BAYN", "VOW3", "BMW", "MBG", "ADS",
    "DB1", "HEI", "MRK", "FRE", "IFX",
    
    # France (15)
    "MC", "OR", "AC", "SU", "BNP", "SAN", "ENGI", "AIR", "SAF", "KER",
    "RI", "CAP", "STM", "VIV", "WLN",
    
    # Switzerland (10)
    "NESN", "ROG", "NOVN", "UBSG", "ZURN", "ABBN", "LONN", "CFR", "GIVN", "SREN",
    
    # Netherlands (5)
    "ASML", "REN", "AD", "INGA", "PHIA",
    
    # Italy (5)
    "ENEL", "ENI", "ISP", "UCG", "STLA",
    
    # Spain (5)
    "SAN", "BBVA", "TEF", "IBE", "FER"
]

# Asian Stocks (75)
ASIA_SYMBOLS = [
    # Japan (25)
    "TM", "SONY", "HMC", "NSANY", "MUFG", "SMFG", "NMR", "SFTBY", "TCEHY", "BABA",
    "7203.T", "9984.T", "6758.T", "6861.T", "6501.T", "6954.T", "8035.T", "9432.T", "8411.T", "8306.T",
    "7267.T", "6902.T", "7741.T", "7751.T", "4502.T",
    
    # China (25)
    "BABA", "TCEHY", "BIDU", "JD", "NTES", "PDD", "LI", "NIO", "XPEV", "BILI",
    "YUMC", "TAL", "EDU", "IQ", "WB", "MOMO", "VIPS", "ATHM", "BZUN", "QTT",
    "0700.HK", "9988.HK", "3690.HK", "1810.HK", "9618.HK",
    
    # India (15)
    "RELIANCE", "TCS", "HDB", "IBN", "INFY", "WIT", "HCLTECH", "BHARTIARTL", "ITC", "SBIN",
    "KOTAKBANK", "AXISBANK", "BAJFINANCE", "MARUTI", "TATAMOTORS",
    
    # South Korea (10)
    "005930.KS", "000660.KS", "035420.KS", "051910.KS", "006400.KS", "005380.KS", "012450.KS", "055550.KS", "032640.KS", "036570.KS"
]

# African Stocks (20)
AFRICA_SYMBOLS = [
    # South Africa (15)
    "NPN", "SBK", "FSR", "SOL", "GFI", "ANG", "IMP", "MTN", "VOD", "ABG",
    "CPI", "CLS", "BVT", "ARI", "AMS",
    
    # Nigeria (3)
    "DANGCEM", "NB", "GUINNESS",
    
    # Egypt (2)
    "COMI", "TMG",
    
    # Kenya (2)
    "SCOM", "EQBNK"
]

# Commodities/ETFs (20)
ETF_SYMBOLS = [
    # Broad Market
    "SPY", "QQQ", "DIA", "IWM", "VTI",
    
    # International
    "EFA", "EEM", "VEA", "VWO", "FXI",
    
    # Sectors
    "XLF", "XLK", "XLE", "XLV", "XLI",
    
    # Bonds
    "TLT", "AGG", "BND", "SHY", "LQD"
]

# Combine all symbols
all_symbols = []

for sym in US_SYMBOLS:
    all_symbols.append(f"{sym}|{sym} Inc.|Technology|US")

for sym in EUROPE_SYMBOLS:
    all_symbols.append(f"{sym}|{sym} SA|Unknown|Europe")

for sym in ASIA_SYMBOLS:
    all_symbols.append(f"{sym}|{sym} Co.|Unknown|Asia")

for sym in AFRICA_SYMBOLS:
    all_symbols.append(f"{sym}|{sym} Ltd|Unknown|Africa")

for sym in ETF_SYMBOLS:
    all_symbols.append(f"{sym}|{sym} ETF|ETF|Worldwide")

# Write to file
with open('symbols.txt', 'w') as f:
    for line in all_symbols:
        f.write(line + '\n')

print(f"✅ Generated {len(all_symbols)} symbols")
print(f"   US: {len(US_SYMBOLS)}")
print(f"   Europe: {len(EUROPE_SYMBOLS)}")
print(f"   Asia: {len(ASIA_SYMBOLS)}")
print(f"   Africa: {len(AFRICA_SYMBOLS)}")
print(f"   ETFs: {len(ETF_SYMBOLS)}")
