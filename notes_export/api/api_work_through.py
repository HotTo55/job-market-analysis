import requests
import pandas as pd

api_key = "deecea7a52b0b8af4f4bf27c9d92235c"

city_name = "paris"
state_code = "il"
country_code = "us"

geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},{state_code},{country_code}&limit=1&appid={api_key}"

