from visual_genome import api as vg
from visual_genome.models import Region
from matplotlib import pyplot as plt
from PIL import Image as PIL_Image
import requests
from io import BytesIO 


class dataset:


    def __init__(self) -> None:

        self.image_ids = {
            "trains": [22, 61576, 61602, 107935, 107943, 107947, 150265, 150276, 150298],
            "sports": [61515, 61519, 61524, 61525, 61570, 61583, 61600, 107921, 150354],
            "giraffes": [61511, 61528, 61532, 61535, 61542, 61587, 107928, 107938, 107945],
            "buildings": [2579, 61531, 107971, 150311, 150318, 2417970, 2417839, 2417702, 2417636],
            "computers": [3, 10, 12, 13, 14, 15, 2417706, 2417296, 2417294]
            }

        self.test_image_ids = {
            "trains":150300,
            "sports":2417857,
            "giraffes":107974,
            "buildings": 2417489,
            "computers": 2416824
            }

        self.additional_annotations = {
            22 : "This is a 4-car tram composition, drawing power from the 600V overhead line with a pantograph",
            22 : "This tram has doors that open automatically at every stop",
            61576 : "British Rail Mark 3 carriages stopped at a station",
            61576 : "BREL Mark 3 carriages entered service in 1975 and are in use by 7 companies today. They are 23 m long and reach a maximum speed of 200 km/h",
            61602 : "A BREL Mark 3 carriage being shunted by a diesel locomotive",
            61602 : "A passenger coach with two 2-axis bogies",
            107935 : "A renfe class 319 diesel locomotive",
            107935 : "This locomotives are known as retales and were built in 1984",
            107943 : "Steel build carriages of the metro train",
            107943 : "Chinese metro train coming out of the tunnel",
            107947 : "Three BNSF locomotives pulling a boxcar freight train",
            107947 : "This is a ES44DC diesel locomotive",
            150276 : "This is a class 1116 locomotive built by Siemens from 2000",
            150276 : "This is a Taurus locomotive which can reach up to 230 km/h top speed",
            150298 : "This is a regional train built by Siemens, Bombardier, and DWA from 1999",
            150298 : "The top speed of this aluminium train is 140 km/h"
            }       


    def get_all_data(self):

        data_dict = {}

        for cat in self.image_ids:
            for el in self.image_ids[cat]:

                descriptions = vg.get_region_descriptions_of_image(id=el)
                descriptions = [descriptions[x].phrase for x in range(len(descriptions))]
                image = (cat, el)
                data_dict.update({image: descriptions})

        return data_dict


    def add_annotation(self, image_id, phrase: str, data_dict: dict):

        descriptions = vg.get_region_descriptions_of_image(id=image_id)
        image = vg.get_image_data(id=image_id)
        descriptions.append(Region(9999, image, phrase, 0, 0, 0, 0))
        descriptions = [descriptions[x].phrase for x in range(len(descriptions))]
        image =(self.get_image_label(image_id), image_id)
        data_dict.update({image: descriptions})
        return data_dict


    def get_image_label(self, image_id):

        for cat in self.image_ids:
            for el in self.image_ids[cat]:
                if el == image_id:
                    return cat
                else:
                    pass


    def visualise(self, image_id):

        image = vg.get_image_data(id=image_id)
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        response = requests.get(image.url)
        img = PIL_Image.open(BytesIO(response.content))
        plt.imshow(img)
        plt.show()