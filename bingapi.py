from PIL import Image
import io
import requests

def get_birdseye_map(center_point, zoom_level=20, map_size="900,700", direction=0, bing_maps_key= "AtgABjMeqhR8ZF3Y9hqx~UmwKvUJxP0-_1pXbtiQ3hg~Aga1It2Oo1dq6mcvgRsWYDZnevj_XtsgBBHrGooJQ0ue4SdcWBPrCaiMnzJmzsnz"):
    """
    Get a Bird's Eye map using Bing Maps API

    Args:
        - center_point (str): The center point of the map in the format "latitude,longitude", e.g., "37.802297,-122.405844".
        - zoom_level (int): The zoom level of the map. Default is 20.
        - map_size (str): The size of the map in pixels in the format "width,height". Default is "900,700".
        - direction (int): The orientation of view for Bird's Eye imagery. Default is 0 (North).
                           Valid values are from 0 to 360, where 0 = North, 90 = East, 180 = South, and 270 = West.
        - bing_maps_key (str): Bing Maps API key. If not provided, None is used as the default.

    Returns:
        - image_data (bytes): The image data in bytes.
    """

    # Construct the request URL
    api_url = "https://dev.virtualearth.net/REST/V1/Imagery/Map/Birdseye/{centerPoint}/{zoomLevel}?dir={direction}&ms={mapSize}&key={BingMapsKey}"

    # Replace the parameters
    api_url = api_url.format(centerPoint=center_point, zoomLevel=zoom_level, direction=direction, mapSize=map_size,
                             BingMapsKey=bing_maps_key)

    # Send the request
    response = requests.get(api_url)

    # Process the response
    if response.status_code == 200:
        # Successful response, return the image data
        return Image.open(io.BytesIO(response.content))

    else:
        # Failed response, raise an exception
        raise Exception("API request failed with status code {}: {}".format(response.status_code, response.text))


def get_static_map_image(center_point, zoom_level=15, map_size="1024,1024", imagery_set="Aerial", bing_maps_key= "AtgABjMeqhR8ZF3Y9hqx~UmwKvUJxP0-_1pXbtiQ3hg~Aga1It2Oo1dq6mcvgRsWYDZnevj_XtsgBBHrGooJQ0ue4SdcWBPrCaiMnzJmzsnz"):
    """
    Get a static satellite image using Bing Maps API

    Args:
        - center_point (str): The center point of the map in the format "latitude,longitude", e.g., "47.610,-122.107".
        - zoom_level (int): The zoom level of the map. Default is 10.
        - map_size (str): The size of the map in pixels in the format "width,height". Default is "350,350".
        - imagery_set (str): The type of imagery. Default is "Aerial".
                             Possible values:
                             - "Aerial": Satellite imagery.
                             - "AerialWithLabels": Satellite imagery with road overlay.
                             - "AerialWithLabelsOnDemand": Satellite imagery with on-demand road overlay.
                             - "Streetside": Street-level imagery.
                             - "BirdsEye": Bird's eye (oblique-angle) imagery.
                             - "BirdsEyeWithLabels": Bird's eye imagery with road overlay.
                             - "Road": Road map without additional imagery.
                             - "CanvasDark": A dark version of the road maps.
                             - "CanvasLight": A lighter version of the road maps with some details disabled, such as hill shading.
                             - "CanvasGray": A grayscale version of the road maps.
        - bing_maps_key (str): Bing Maps API key. If not provided, None is used as the default.

    Returns:
        - image_data (bytes): The image data in bytes.
    """
    # Construct the request URL
    api_url = "https://dev.virtualearth.net/REST/v1/Imagery/Map/{imagerySet}/{centerPoint}/{zoomLevel}?mapSize={mapSize}&key={BingMapsKey}"

    # Replace the parameters
    api_url = api_url.format(imagerySet=imagery_set, centerPoint=center_point, zoomLevel=zoom_level, mapSize=map_size,
                             BingMapsKey=bing_maps_key)

    # Send the request
    response = requests.get(api_url)

    # Process the response
    if response.status_code == 200:
        # Successful response, return the image data
        return Image.open(io.BytesIO(response.content))
    else:
        # Failed response, raise an exception
        raise Exception("API request failed with status code {}: {}".format(response.status_code, response.text))
