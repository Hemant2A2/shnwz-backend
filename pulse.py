import time
import board, busio, adafruit_bme680
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
import requests
import os

# Load your Flask server’s base URL via env or hardcode
API_URL = os.environ.get('HEALTH_BACKEND_URL', 'http://<BACKEND_IP>:5000')
INGEST_ENDPOINT = f"{API_URL}/api/sensor-data"

# I2C + sensor init…
i2c = busio.I2C(board.SCL, board.SDA)
bme = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=0x76)
ads = ADS1115(i2c)
chan = AnalogIn(ads, ADS1115.P0)

def read_sensors():
    return {
        "temperature": round(bme.temperature, 2),
        "humidity":    round(bme.humidity,    2),
        "pressure":    round(bme.pressure,    2),
        "gasLevel":    int(bme.gas),
        # convert chan.value (raw ADC) → BPM however you like:
        "heartRate":   int(chan.value),
        "timestamp":   datetime.datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    while True:
        payload = read_sensors()
        try:
            resp = requests.post(INGEST_ENDPOINT, json=payload, timeout=5)
            resp.raise_for_status()
        except Exception as e:
            print("Failed to push sensor data:", e)
        time.sleep(1)  # send once per second (or whatever interval)