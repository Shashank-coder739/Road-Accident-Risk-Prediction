from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# ==========================================
# LOAD MODEL + DATASET
# ==========================================

MODEL_FILE = "test1.pkl"
DATA_FILE = "accidents_india.csv"

model = pickle.load(open(MODEL_FILE, "rb"))
df = pd.read_csv(DATA_FILE).dropna().copy()


# ==========================================
# ENCODERS
# ==========================================

day_encoder = LabelEncoder()
day_encoder.fit(df["Day_of_Week"])

light_encoder = LabelEncoder()
light_encoder.fit(df["Light_Conditions"])

severity_encoder = LabelEncoder()
severity_encoder.fit(df["Accident_Severity"])


# ==========================================
# OPTIONS
# ==========================================

SEX_OPTIONS = [
    ("Male", "1"),
    ("Female", "0")
]

VEHICLE_OPTIONS = [
    ("Car", "1"),
    ("Bus", "2"),
    ("Bike", "3"),
    ("Truck", "4"),
    ("Auto Rickshaw", "5"),
    ("Van", "6"),
    ("SUV", "7"),
    ("Taxi", "8"),
    ("Motorcycle", "9"),
    ("Bicycle", "10"),
    ("Other", "11")
]

WEATHER_OPTIONS = [
    "Clear",
    "Rainy",
    "Foggy",
    "Cloudy",
    "Stormy",
    "Snowy"
]

ROAD_OPTIONS = [
    "Good",
    "Moderate",
    "Poor",
    "Wet",
    "Damaged"
]

TIME_OPTIONS = [
    "Morning",
    "Afternoon",
    "Evening",
    "Night"
]

TRAFFIC_OPTIONS = [
    "Low",
    "Medium",
    "High",
    "Very High"
]

EXPERIENCE_OPTIONS = [
    "Beginner",
    "Intermediate",
    "Experienced"
]


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return render_template(
        "t.html",

        sex_options=SEX_OPTIONS,
        vehicle_options=VEHICLE_OPTIONS,
        weather_options=WEATHER_OPTIONS,
        road_options=ROAD_OPTIONS,
        time_options=TIME_OPTIONS,
        traffic_options=TRAFFIC_OPTIONS,
        experience_options=EXPERIENCE_OPTIONS
    )


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # --------------------------------------
        # GET FORM VALUES
        # --------------------------------------

        age = int(request.form["age"])

        sex_text = request.form["sex"]

        vehicle_text = request.form["vehicle"]

        weather = request.form["weather"]

        road_condition = request.form["road_condition"]

        time_of_day = request.form["time"]

        speed = int(request.form["speed"])

        traffic = request.form["traffic"]

        experience = request.form["experience"]

        seatbelt = request.form["seatbelt"]

        alcohol = request.form["alcohol"]


        # --------------------------------------
        # SEX CONVERSION
        # --------------------------------------

        sex_mapping = {
            "Male": 1,
            "Female": 0
        }

        sex = sex_mapping.get(sex_text, 1)


        # --------------------------------------
        # VEHICLE CONVERSION
        # --------------------------------------

        vehicle_mapping = {
            "Car": 1,
            "Bus": 2,
            "Bike": 3,
            "Truck": 4,
            "Auto Rickshaw": 5,
            "Van": 6,
            "SUV": 7,
            "Taxi": 8,
            "Motorcycle": 9,
            "Bicycle": 10,
            "Other": 11
        }

        vehicle = vehicle_mapping.get(vehicle_text, 11)


        # --------------------------------------
        # ROAD TYPE
        # --------------------------------------
        # Model expects Road_Type as numeric.
        # We convert the new UI road condition
        # into a simple road category.

        road_mapping = {
            "Good": 1,
            "Moderate": 2,
            "Poor": 3,
            "Wet": 4,
            "Damaged": 5
        }

        road = road_mapping.get(road_condition, 1)


        # --------------------------------------
        # DAY
        # --------------------------------------
        # Current UI doesn't ask day.
        # Use Monday as default because the model
        # requires a Day feature.

        default_day = "Monday"

        day_value = int(
            day_encoder.transform([default_day])[0]
        )


        # --------------------------------------
        # LIGHT CONDITION
        # --------------------------------------
        # Convert time of day to Daylight/Darkness.

        if time_of_day in ["Morning", "Afternoon", "Evening"]:
            light_text = "Daylight"
        else:
            light_text = "Darkness"


        # Check available values in dataset
        if light_text in list(light_encoder.classes_):

            light_value = int(
                light_encoder.transform([light_text])[0]
            )

        else:

            light_value = 0


        # --------------------------------------
        # PASSENGERS
        # --------------------------------------
        # Current UI doesn't ask passengers.
        # Model requires this feature.

        passengers = 0


        # ======================================
        # MODEL INPUT
        # ======================================

        input_data = pd.DataFrame([{

            "Sex_Of_Driver": sex,

            "Vehicle_Type": vehicle,

            "Speed_limit": speed,

            "Road_Type": road,

            "Number_of_Pasengers": passengers,

            "Day": day_value,

            "Light": light_value

        }])


        # --------------------------------------
        # EXACT MODEL FEATURE ORDER
        # --------------------------------------

        if hasattr(model, "feature_names_in_"):

            input_data = input_data[
                list(model.feature_names_in_)
            ]


        # ======================================
        # PREDICTION
        # ======================================

        prediction = model.predict(input_data)[0]


        # ======================================
        # PROBABILITY
        # ======================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = round(
                float(max(probabilities)) * 100,
                1
            )

        else:

            probabilities = None
            confidence = 0


        # ======================================
        # PREDICTION LABEL
        # ======================================

        try:

            prediction_label = severity_encoder.inverse_transform(
                [prediction]
            )[0]

        except Exception:

            prediction_label = str(prediction)


        # ======================================
        # EXPLANATION
        # ======================================

        if prediction_label == "Serious":

            explanation = (
                "The model classified the entered "
                "conditions as Serious based on patterns "
                "learned from the accident dataset."
            )

        else:

            explanation = (
                "The model classified the entered "
                "conditions as Slight based on patterns "
                "learned from the accident dataset."
            )


        # ======================================
        # RETURN RESULT
        # ======================================

        return render_template(

            "t.html",

            prediction=prediction_label,

            pred=prediction_label,

            explanation=explanation,

            confidence=confidence,

            selected_age=str(age),

            selected_sex=sex_text,

            selected_vehicle=vehicle_text,

            selected_weather=weather,

            selected_road=road_condition,

            selected_time=time_of_day,

            selected_speed=str(speed),

            selected_traffic=traffic,

            selected_experience=experience,

            selected_seatbelt=seatbelt,

            selected_alcohol=alcohol,

            sex_options=SEX_OPTIONS,

            vehicle_options=VEHICLE_OPTIONS,

            weather_options=WEATHER_OPTIONS,

            road_options=ROAD_OPTIONS,

            time_options=TIME_OPTIONS,

            traffic_options=TRAFFIC_OPTIONS,

            experience_options=EXPERIENCE_OPTIONS

        )


    # ======================================
    # ERROR
    # ======================================

    except Exception as e:

        print("PREDICTION ERROR:", e)

        return render_template(

            "t.html",

            error="Prediction could not be completed.",

            error_detail=str(e),

            sex_options=SEX_OPTIONS,

            vehicle_options=VEHICLE_OPTIONS,

            weather_options=WEATHER_OPTIONS,

            road_options=ROAD_OPTIONS,

            time_options=TIME_OPTIONS,

            traffic_options=TRAFFIC_OPTIONS,

            experience_options=EXPERIENCE_OPTIONS

        )


# ==========================================
# GRAPHS
# ==========================================

@app.route("/Graphs")
def graphs():

    data = pd.read_csv(
        DATA_FILE
    ).dropna()

    severity_counts = (
        data["Accident_Severity"]
        .value_counts()
        .to_dict()
    )

    speed_counts = (
        data["Speed_limit"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    day_counts = (
        data["Day_of_Week"]
        .value_counts()
        .to_dict()
    )

    return render_template(
        "graph.html",
        severity_counts=severity_counts,
        speed_counts=speed_counts,
        day_counts=day_counts
    )


# ==========================================
# PIE
# ==========================================

@app.route("/Pie")
def pie():

    data = pd.read_csv(
        DATA_FILE
    ).dropna()

    severity_counts = (
        data["Accident_Severity"]
        .value_counts()
        .to_dict()
    )

    light_counts = (
        data["Light_Conditions"]
        .value_counts()
        .to_dict()
    )

    return render_template(
        "pie.html",
        severity_counts=severity_counts,
        light_counts=light_counts
    )


# ==========================================
# MAP PAGES
# ==========================================

@app.route("/Map")
def map1():

    return render_template("map.html")


@app.route("/Map1")
def map2():

    return render_template("ur.html")


@app.route("/Map2")
def map3():

    return render_template("bs.html")


@app.route("/Map3")
def map4():

    return render_template("hm.html")


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)