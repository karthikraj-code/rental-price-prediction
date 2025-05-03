import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import turtle
import time


def display_text(t, text, x, y):
    t.penup()
    t.goto(x, y)
    t.pendown()
    t.write(text, align="center", font=("Arial", 14, "normal"))


def get_user_input():
    turtle.clearscreen()
    turtle.bgcolor("lightblue")
    turtle.title("Rental Price Predictor")
    
    display_text(turtle, "Welcome to the Rental Price Predictor", 0, 100)
    turtle.hideturtle()

    bedrooms = int(turtle.numinput("Input", "Enter number of bedrooms (1-4):", minval=1, maxval=4))
    bathrooms = int(turtle.numinput("Input", "Enter number of bathrooms (1-2):", minval=1, maxval=2))
    size_sqft = int(turtle.numinput("Input", "Enter size in sqft (500-2000):", minval=500, maxval=2000))
    location = turtle.textinput("Input", "Enter location (VIJ, GUN, VIJAG, NEL, ONG, KDP):")
    
    # New Features
    property_type = turtle.textinput("Input", "Enter property type (Apartment, Villa, Studio):")
    age_of_property = int(turtle.numinput("Input", "Enter age of property (in years):", minval=0, maxval=50))
    close_to_transport = turtle.textinput("Input", "Close to public transport? (yes/no):").strip().lower() == 'yes'
    nearby_schools = turtle.textinput("Input", "Nearby schools? (yes/no):").strip().lower() == 'yes'
    
    swimming_pool = turtle.textinput("Input", "Swimming pool? (yes/no):").strip().lower() == 'yes'
    gym = turtle.textinput("Input", "Gym? (yes/no):").strip().lower() == 'yes'
    parking = turtle.textinput("Input", "Parking? (yes/no):").strip().lower() == 'yes'

    return (bedrooms, bathrooms, size_sqft, location, property_type, age_of_property,
            close_to_transport, nearby_schools, swimming_pool, gym, parking)


def plot_average_price(df):
    all_locations = ['VIJ', 'GUN', 'VIJAG', 'NEL', 'ONG', 'KDP']
    location_averages = {}

    
    for loc in all_locations:
        loc_column = 'location_' + loc
        if loc_column in df.columns:  
            average_price = df[df[loc_column] == 1]['price'].mean()
            
            location_averages[loc] = average_price if average_price > 0 else np.random.randint(20000, 40000)
        else:
            
            location_averages[loc] = np.random.randint(20000, 40000)


    amenities_averages = {
        'Swimming Pool': df[df['swimming_pool'] == 1]['price'].mean() or 0,
        'Gym': df[df['gym'] == 1]['price'].mean() or 0,
        'Parking': df[df['parking'] == 1]['price'].mean() or 0,
        'No Swimming Pool': df[df['swimming_pool'] == 0]['price'].mean() or 0,
        'No Gym': df[df['gym'] == 0]['price'].mean() or 0,
        'No Parking': df[df['parking'] == 0]['price'].mean() or 0
    }

  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   
    ax1.bar(location_averages.keys(), location_averages.values(), color='skyblue')
    ax1.set_title('Average Rental Price by Location')
    ax1.set_ylabel('Price (in ₹)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

   
    ax2.bar(amenities_averages.keys(), amenities_averages.values(), color=['green', 'orange', 'blue', 'red', 'purple', 'yellow'])
    ax2.set_title('Average Rental Price by Amenities')
    ax2.set_ylabel('Price (in ₹)')
    ax2.set_xticklabels(amenities_averages.keys(), rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show() 


def run_model(user_inputs):
    
    (bedrooms, bathrooms, size_sqft, location, property_type, age_of_property,
     close_to_transport, nearby_schools, swimming_pool, gym, parking) = user_inputs

    
    num_samples = 1000
    data = {
        'bedrooms': np.random.randint(1, 5, size=num_samples),
        'bathrooms': np.random.randint(1, 3, size=num_samples),
        'size_sqft': np.random.randint(500, 2000, size=num_samples),
        'location': np.random.choice(['VIJ', 'GUN', 'VIJAG', 'NEL', 'ONG', 'KDP'], size=num_samples),
        'property_type': np.random.choice(['Apartment', 'Villa', 'Studio'], size=num_samples),
        'age_of_property': np.random.randint(0, 51, size=num_samples),
        'close_to_transport': np.random.randint(0, 2, size=num_samples),
        'nearby_schools': np.random.randint(0, 2, size=num_samples),
        'swimming_pool': np.random.randint(0, 2, size=num_samples),
        'gym': np.random.randint(0, 2, size=num_samples),
        'parking': np.random.randint(0, 2, size=num_samples),
    }

    
    df = pd.DataFrame(data)

    
    df['price'] = (
        df['bedrooms'] * 1000 +
        df['bathrooms'] * 500 +
        df['size_sqft'] * 10 +  
        df['property_type'].map({
            'Apartment': np.random.randint(20000, 40001),
            'Villa': np.random.randint(40000, 60001),
            'Studio': np.random.randint(15000, 20001)
        }) +
        df['age_of_property'] * -100 + 
        df['close_to_transport'] * 250 +  
        df['nearby_schools'] * 200 +  
        df['swimming_pool'] * 350 +  
        df['gym'] * 1000 +  
        df['parking'] * 2000 +  
        np.random.normal(0, 5000, num_samples) 
    )

   
    df = pd.get_dummies(df, columns=['location', 'property_type'], drop_first=True)

 
    X = df.drop('price', axis=1)
    y = df['price']

  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

   
    new_data = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'size_sqft': size_sqft,
        'location': location,
        'property_type': property_type,
        'age_of_property': age_of_property,
        'close_to_transport': int(close_to_transport),
        'nearby_schools': int(nearby_schools),
        'swimming_pool': int(swimming_pool),
        'gym': int(gym),
        'parking': int(parking)
    }])

    new_data = pd.get_dummies(new_data, columns=['location', 'property_type'])

    
    for col in X.columns:
        if col not in new_data.columns:
            new_data[col] = 0

    new_data = new_data[X.columns]

    
    predicted_price = model.predict(new_data)[0]
    
    
    turtle.clearscreen()
    turtle.bgcolor("lightblue")
    turtle.title("Predicted Rental Price")
    display_text(turtle, f"Predicted Rental Price: ₹{predicted_price:.2f}", 0, 0)
    
    
    plot_average_price(df)

    
    time.sleep(10)
    turtle.done()

def main():
    user_inputs = get_user_input()
    run_model(user_inputs)

if __name__ == "__main__":
    main()

