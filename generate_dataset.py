import pandas as pd
import random

data = []

for i in range(365):

    month = (i // 30) + 1
    if month > 12:
        month = 12

    # -------------------
    # Rainfall pattern
    # -------------------
    if month in [10, 11]:
        rainfall = random.randint(80, 200)
    elif month in [6, 7, 8]:
        rainfall = random.randint(20, 80)
    else:
        rainfall = random.randint(0, 20)

    # -------------------
    # Temperature
    # -------------------
    if month in [4, 5]:
        temperature = random.randint(33, 38)
    elif month in [12, 1]:
        temperature = random.randint(26, 30)
    else:
        temperature = random.randint(28, 34)

    # -------------------
    # Water Quality (inverse of rainfall)
    # -------------------
    water_quality = max(30, 100 - rainfall + random.randint(-10, 10))

    # -------------------
    # Cases generation (IMPORTANT FIX)
    # Cases increase when rainfall high & water quality poor
    # -------------------
    if rainfall > 150 or water_quality < 40:
        cases = random.randint(120, 250)
    elif rainfall > 80 or water_quality < 55:
        cases = random.randint(60, 120)
    else:
        cases = random.randint(5, 60)

    # -------------------
    # Multi-Condition Risk Logic
    # -------------------
    if (
        cases >= 150
        or (water_quality <= 35 and rainfall >= 150)
        or (cases >= 100 and water_quality <= 40)
        or (rainfall >= 200 and temperature >= 32)
    ):
        risk = 2   # HIGH

    elif (
        cases >= 70
        or water_quality <= 50
        or rainfall >= 120
    ):
        risk = 1   # MODERATE

    else:
        risk = 0   # LOW

    data.append([
        rainfall,
        temperature,
        water_quality,
        cases,
        risk
    ])

df = pd.DataFrame(data, columns=[
    "rainfall_mm",
    "temperature_c",
    "water_quality_index",
    "cases",
    "risk"
])

df.to_csv("coimbatore_dataset_final.csv", index=False)

print("Dataset generated successfully!")