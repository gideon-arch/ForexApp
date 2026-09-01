# test_api.py
# Test API connectivity — Kraken, Deriv, and local endpoints

import aiohttp
import asyncio


async def test_kraken():
    print("\nTesting Kraken (BTCUSD)...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.kraken.com/0/public/Time", timeout=10) as response:
                print(f"  Status: {response.status}")
                data = await response.json()
                print(f"  Server Time: {data.get('result', {}).get('unixtime', 'N/A')}")
                return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


async def test_kraken_eur():
    print("\nTesting Kraken (EURUSD fallback)...")
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.kraken.com/0/public/OHLC"
            params = {"pair": "EURUSD", "interval": 5}
            async with session.get(url, params=params, timeout=15) as response:
                print(f"  Status: {response.status}")
                data = await response.json()
                if "error" in data and data["error"]:
                    print(f"  API Error: {data['error']}")
                    return False
                if "result" in data:
                    key = list(data["result"].keys())[0]
                    print(f"  Candles available: {len(data['result'][key])}")
                    return True
                return False
    except Exception as e:
        print(f"  Failed: {e}")
        return False


async def test_deriv():
    print("\nTesting Deriv (EURUSD)...")
    try:
        import websockets
        import json

        async with websockets.connect("wss://ws.derivws.com/websockets/v3?app_id=1089", ping_interval=20, ping_timeout=10) as ws:
            req = {
                "ticks_history": "frxEURUSD",
                "adjust_start_time": 1,
                "count": 10,
                "end": "latest",
                "start": 1,
                "style": "candles",
                "granularity": 300
            }
            await ws.send(json.dumps(req))
            response = await asyncio.wait_for(ws.recv(), timeout=15.0)
            data = json.loads(response)

            if data.get("msg_type") == "candles":
                candles = data.get("candles", [])
                print(f"  Status: OK | Candles: {len(candles)}")
                if candles:
                    print(f"  Latest: O={candles[-1]['open']} C={candles[-1]['close']}")
                return True
            else:
                print(f"  Error: {data.get('error', 'Unknown')}")
                return False
    except Exception as e:
        print(f"  Failed: {e}")
        return False


async def test_local_api():
    print("\nTesting Local API Endpoints...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/", timeout=5) as response:
                print(f"  Root: {response.status}")
                data = await response.json()
                print(f"  Response: {data}")

            async with session.get("http://localhost:8000/health", timeout=5) as response:
                print(f"  Health: {response.status}")
                data = await response.json()
                print(f"  Response: {data}")

            async with session.get("http://localhost:8000/api/signals", timeout=5) as response:
                print(f"  Signals: {response.status}")
                data = await response.json()
                print(f"  Response: {data}")

            async with session.get("http://localhost:8000/analyze", timeout=30) as response:
                print(f"  /analyze: {response.status}")
                data = await response.json()
                print(f"  Triggered: {data.get('triggered_at')}")
                for asset, res in data.get('results', {}).items():
                    print(f"    {asset}: {res.get('signal')} — {res.get('reason', 'N/A')[:60]}")

            return True
    except Exception as e:
        print(f"  Local API failed (is server running?): {e}")
        return False


async def main():
    print("=" * 50)
    print("API Connectivity Test v5.0")
    print("=" * 50)

    kraken_btc = await test_kraken()
    kraken_eur = await test_kraken_eur()
    deriv_eur = await test_deriv()
    api_ok = await test_local_api()

    print("\n" + "=" * 50)
    print("Results:")
    print(f"  Kraken BTCUSD: {'OK' if kraken_btc else 'FAILED'}")
    print(f"  Kraken EURUSD: {'OK' if kraken_eur else 'FAILED'}")
    print(f"  Deriv EURUSD:  {'OK' if deriv_eur else 'FAILED'}")
    print(f"  Local API:     {'OK' if api_ok else 'FAILED'}")
    print("=" * 50)


asyncio.run(main())