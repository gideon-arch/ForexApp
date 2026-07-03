# test_api.py
# Test API connectivity and endpoints

import aiohttp
import asyncio


async def test_kraken():
    print("\nTesting Kraken...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.kraken.com/0/public/Time", timeout=10) as response:
                print(f"Kraken Status: {response.status}")
                data = await response.json()
                print(f"Kraken Server Time: {data.get('result', {}).get('unixtime', 'N/A')}")
                return True
    except Exception as e:
        print(f"Kraken failed: {e}")
        return False


async def test_local_api():
    print("\nTesting Local API Endpoints...")
    try:
        async with aiohttp.ClientSession() as session:
            # Test root
            async with session.get("http://localhost:8000/", timeout=5) as response:
                print(f"Root Status: {response.status}")
                data = await response.json()
                print(f"Root Response: {data}")

            # Test health
            async with session.get("http://localhost:8000/health", timeout=5) as response:
                print(f"Health Status: {response.status}")
                data = await response.json()
                print(f"Health Response: {data}")

            # Test signals
            async with session.get("http://localhost:8000/api/signals", timeout=5) as response:
                print(f"Signals Status: {response.status}")
                data = await response.json()
                print(f"Signals Response: {data}")

            return True
    except Exception as e:
        print(f"Local API test failed (is the server running?): {e}")
        return False


async def main():
    print("="*50)
    print("API Connectivity Test")
    print("="*50)

    kraken_ok = await test_kraken()
    api_ok = await test_local_api()

    print("\n" + "="*50)
    print("Results:")
    print(f"  Kraken API: {'OK' if kraken_ok else 'FAILED'}")
    print(f"  Local API:  {'OK' if api_ok else 'FAILED (server may not be running)'}")
    print("="*50)


asyncio.run(main())

