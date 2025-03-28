from typing import Dict, Union
import math

class ObjectStatus:
    def __init__(self, **kwargs):
        self.x = 0
        self.y = 0
        self.v = 0
        self.a = 0
        self.yaw = 0
        self.width = 0
        self.length = 0
        self.extern_obj = ExternObject()

        self.update(**kwargs)
    
    def __str__(self):
        return str(vars(self))
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key == 'yaw':
                    value = value % (2 * math.pi)
                setattr(self, key, round(value, 3))

    def update_extern_object(self, externObjData):
        self.extern_obj.update(externObjData)

    def get_extern_object_info(self):
        return {key: vars(value) for key, value in self.extern_obj.externObjDict.items()}

class EgoStatus(ObjectStatus):
    def __init__(self, **kwargs):
        self.a = 0
        self.rot = 0
        super().__init__()
        self.update(**kwargs)

class Observation():
    def __init__(self):
        self.ego_info = EgoStatus()
        self.object_info: Dict[str, Dict[str, ObjectStatus]] = {
            'vehicle': {},
            'bicycle': {},
            'pedestrian': {},
        }
        self.light_info = ""
        self.test_info = {
            "t": 0.00, 
            "dt": 0.00,
            "end": -1,
        }

    def __str__(self):
        result = ""
        result += f"- ego_info: {str(self.ego_info)}\n"
        result += "- object_info: \n"
        for category, objects in self.object_info.items():
            if objects:
                result += f"  + \"{category}\":\n"
                for obj_name, obj_status in objects.items():
                    result += f"      \"{obj_name}\" - {str(obj_status)}\n"
        result += f"- light_info: {self.light_info}\n"
        result += f"- test_info: {self.test_info}\n"
        return result
    
    def update_ego_info(self, **kwargs):
        self.ego_info.update(**kwargs)

    def update_light_info(self, light_info: str=""):
        self.light_info = light_info

    def update_test_info(self, **kwargs):
        self.test_info.update(**kwargs)

    def erase_object_info(self):
        self.object_info = {
            'vehicle': {},
            'bicycle': {},
            'pedestrian': {},
        }

    def update_object_info(self, category: str, obj_name: str, **kwargs):
        if category in self.object_info.keys():
            obj_name = str(obj_name)
            if obj_name not in self.object_info[category].keys():
                self.object_info[category][obj_name] = ObjectStatus()
            self.object_info[category][obj_name].update(**kwargs)


class AvStruct:
    def __init__(self, name):
        self.name = name
        self.speed = None
        self.angle = None
        self.frameId = None
        self.pos = []
        self.length = None
        self.width = None
        self.bound = []

    def update(self, value):
        self.speed = value['speed']
        self.angle = value['courseAngle']
        self.frameId = value['frameId']
        self.pos = value['tessngPos']
        self.length = value['length'] / 100 + 2
        self.width = value['width'] / 100 + 1.5
        self.bound = self.calculateVehicleRectangleVertices(
            self.pos[0], self.pos[1], self.angle, self.length, self.width
        )

    def calculateVehicleRectangleVertices(self, center_x, center_y, heading, length, width):
        angle = math.radians(90 - heading)
        dx, dy = length / 2, width / 2

        return [
            (center_x - dx * math.cos(angle) - dy * math.sin(angle),
             center_y - dx * math.sin(angle) + dy * math.cos(angle)),

            (center_x - dx * math.cos(angle) + dy * math.sin(angle),
             center_y - dx * math.sin(angle) - dy * math.cos(angle)),

            (center_x + dx * math.cos(angle) - dy * math.sin(angle),
             center_y + dx * math.sin(angle) + dy * math.cos(angle)),

            (center_x + dx * math.cos(angle) + dy * math.sin(angle),
             center_y + dx * math.sin(angle) - dy * math.cos(angle)),
        ]


class ExternObject:
    def __init__(self):
        self.externObjDict = {}

    def update(self, externObjData):
        if not externObjData:
            return

        for channel, vehicleInfos in externObjData.items():
            if channel != "type" and vehicleInfos:
                value1 = vehicleInfos["value"]
                myKey = list(value1.keys())[-1]
                value = value1[myKey]

                av = AvStruct(channel)
                av.update(value)
                self.externObjDict[channel] = av

if __name__ == "__main__":
    observation = Observation()
    print(observation)
