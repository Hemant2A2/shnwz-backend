import time
import random
import board
import busio
import adafruit_bme680
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn

# I2C setup
i2c = busio.I2C(board.SCL, board.SDA)

# BME688 setup
bme = adafruit_bme680.Adafruit_BME680_I2C(i2c, address=0x76)
previous_heartbeat = 80
# ADS1115 ADC setup
while True:
    print("=== Sensor Readings ===")
    print("Temperature: {:.1f} C".format(bme.temperature))
    print("Humidity: {:.1f} %".format(bme.humidity))
    print("Pressure: {:.1f} hPa".format(bme.pressure))
    print("Gas: {} Ohms".format(bme.gas))
    time.sleep(1)