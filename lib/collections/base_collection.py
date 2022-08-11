from typing import Any, List, Type

from lib.collections.base_prop import BaseProp


class BaseCollection(object):
    NAME = ""
    PROPS = []
    UNKNOWN_PROP = None

    def get_by_name(self, name, throw_exception=False) -> Type[BaseProp]:
        for prop in self.PROPS:
            for keyword in prop.KEYWORDS:
                if keyword.lower() == name.lower():
                    return prop
        if throw_exception:
            raise ValueError("Cannot find {} in collection: {}".format(name, self.NAME))

        print("Warning: Cannot find name: {} in collection: {}".format(name, self.NAME))
        return self.UNKNOWN_PROP

    def get_by_id(self, label_id) -> BaseProp:
        for prop in self.PROPS:
            if prop.LABEL_ID == label_id:
                return prop
        print("Warning: Cannot find id: {}".format(label_id))
        return self.UNKNOWN_PROP

    def get_by_index(self, index: int) -> BaseProp:
        return self.PROPS[index]

    def is_unknown_prop(self, prop: BaseProp) -> bool:
        return issubclass(prop, self.UNKNOWN_PROP)

    def get_all_props_dict(self) -> List[Any]:
        res = []
        for prop in [*self.PROPS, self.UNKNOWN_PROP]:
            res.append({
                "name": prop.NAME,
                "label_id": prop.LABEL_ID,
                "color": prop.COLOR
            })
        return res

    def get_names(self):
        return [prop.NAME for prop in self.PROPS + [self.UNKNOWN_PROP]]

    @property
    def label_count(self):
        return len(self.PROPS)
