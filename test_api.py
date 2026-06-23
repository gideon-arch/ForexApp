import aiohttp
import asyncio

async def test_google():
    print("Testing Google...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://www.google.com", timeout=10) as response:
                print(f"Google Status: {response.status}")
    except Exception as e:
        print(f"Google failed: {e}")

async def test_kraken():
    print("\nTesting Kraken...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.kraken.com/0/public/Time", timeout=10) as response:
                print(f"Kraken Status: {response.status}")
                data = await response.json()
                print(f"Kraken Data: {data}")
    except Exception as e:
        print(f"Kraken failed: {e}")

async def test_deriv():
    print("\nTesting Deriv...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.deriv.com/api/ticks_history?ticks_history=BTC/USD&granularity=300&count=10&end=latest&style=candles", timeout=10) as response:
                print(f"Deriv Status: {response.status}")
                data = await response.json()
                print(f"Deriv Data keys: {data.keys()}")
    except Exception as e:
        print(f"Deriv failed: {e}")

async def main():
    await test_google()
    await test_kraken()
    await test_deriv()
    print("\nDone!")

asyncio.run(main())