from visual_genome import api as vg
from visual_genome import models

class utils:
    def __init__(self) -> None:
        pass

    def get_image(self, image_id):

        image = vg.get_image_data(id=image_id)

        descriptions = vg.get_region_descriptions_of_image(id=image_id)

        return image, descriptions

    def add_annotation(self, image_id):
        pass



class dataset:
    def __init__(self) -> None:

        self.image_ids = {
            "trains": [22, 61576, 61602, 107935, 107943, 107947, 150265, 150276, 150298, 150300],
            "sports": [61515, 61519, 61524, 61525, 61570, 61583, 61600, 107921, 150354, 2417857],
            "giraffes": [61511, 61528, 61532, 61535, 61542, 61587, 107928, 107938, 107945, 107974],
            "buildings": [2579, 61531, 107971, 150311, 150318, 2417970, 2417839, 2417702, 2417636, 2417489],
            "computers": [3, 10, 12, 13, 14, 15, 2417706, 2417296, 2417294, 2416824]
            }

        self.utils = utils()
    
    def get_all_data(self):

        data_dict = {"image":"data"}

        for cat in self.image_ids:
            for el in self.image_ids[cat]:

                image, data = self.utils.get_image(el)
                data_dict.update({image: data})

        return data_dict


if __name__ == "__main__":
    dset = dataset()

    data = dset.get_data()

    print(data)
        

        