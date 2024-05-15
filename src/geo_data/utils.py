import os
import googlemaps
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

def get_geo_data(location:str) -> List[Dict]:
    """given an address or location, this function uses google's api to 
    
    ### Keyword arguments:
    location (str) : the location of a place or landmark
    Return: a list of metadata pertaining to te location
    """
    
    gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAP_API"))

    # Geocode the address
    geocode_result = gmaps.geocode(location)
    return geocode_result

    
if __name__ == "__main__":
    x = get_geo_data("Royal Palaces of Abomey")
    c = 0