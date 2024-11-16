from intbase import InterpreterBase


# Enumerated type for our different language data types
class Type:
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    NIL = "nil"
    VOID = "void"

################ VALUE ################
# Represents a value, which has a type and its value
class Value:
    def __init__(self, type, value=None):
        self.t = type
        self.v = value

    def value(self):
        if self.t == Type.VOID:
            return None # VOID doesn't have any meaningful value
        return self.v

    def type(self):
        return self.t

def create_value(val):
    if val == InterpreterBase.TRUE_DEF:
        return Value(Type.BOOL, True)
    elif val == InterpreterBase.FALSE_DEF:
        return Value(Type.BOOL, False)
    elif val == InterpreterBase.NIL_DEF:
        return Value(Type.NIL, None)
    elif val == InterpreterBase.VOID_DEF:
        return Value(Type.VOID, None)
    elif isinstance(val, str):
        return Value(Type.STRING, val)
    elif isinstance(val, int):
        return Value(Type.INT, val)
    else:
        raise ValueError("Unknown value type")


def get_printable(val):
    if val.type() == Type.INT:
        return str(val.value())
    if val.type() == Type.STRING:
        return val.value()
    if val.type() == Type.BOOL:
        if val.value() is True:
            return "true"
        return "false"
    if val.type() == Type.VOID:
        return "void"
    if val.type() == Type.NIL:
        return "nil"
    return None

################ STRUCT ################
# represents a user-defined struct type <struct name> and <dictionary of fields/types>
class StructType:
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields  # dict of field names and their types

# represents a struct value (instance of a struct type)
class StructValue:
    def __init__(self, struct_type, struct_def_list):
        self.struct_type = struct_type
        self.fields = {
            field: self.get_default_value(field_type, struct_def_list)
            for field, field_type in struct_type.fields.items()
        }

    def get_default_value(self, field_type, struct_def_list):
        # Initialize fields with default values based on their type
        if field_type == "int":
            return Value(Type.INT, 0)
        elif field_type == "bool":
            return Value(Type.BOOL, False)
        elif field_type == "string":
            return Value(Type.STRING, "")
        elif field_type in struct_def_list:  # 🍅 user-defined/nested struct (passed in list)
            return create_value(InterpreterBase.NIL_DEF)
        else:
            raise ValueError(f"Unknown field type: {field_type}")

    def get_field(self, field_name):
        if field_name not in self.fields:
            raise ValueError(f"Field '{field_name}' does not exist in struct '{self.struct_type.name}'")
        return self.fields[field_name]

    def set_field(self, field_name, value):
        if field_name not in self.fields:
            raise ValueError(f"Field '{field_name}' does not exist in struct '{self.struct_type.name}'")
        self.fields[field_name] = value

    def type(self):
        return self.struct_type