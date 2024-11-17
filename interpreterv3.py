# document that we won't have a return inside the init/update of a for loop

import copy
from enum import Enum

from brewparse import parse_program
from env_v2 import EnvironmentManager
from intbase import InterpreterBase, ErrorType
from type_valuev2 import Type, Value, create_value, get_printable
from type_valuev2 import StructType, StructValue


class ExecStatus(Enum):
    CONTINUE = 1
    RETURN = 2


# Main interpreter class
class Interpreter(InterpreterBase):
    # constants
    NIL_VALUE = create_value(InterpreterBase.NIL_DEF)
    TRUE_VALUE = create_value(InterpreterBase.TRUE_DEF)
    VOID_VALUE = create_value(InterpreterBase.VOID_DEF)
    BIN_OPS = {"+", "-", "*", "/", "==", "!=", ">", ">=", "<", "<=", "||", "&&"}

    # methods
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.__setup_ops()
        self.struct_definitions = {}  # dict of all user-defined struct types

    def run(self, program):
        ast = parse_program(program)
        self.__set_up_struct_definitions(ast)
        self.__set_up_function_table(ast)
        # print(f"🏡: self.struct_definitions {self.struct_definitions}")
        self.env = EnvironmentManager()
        self.__call_func_aux("main", [])

    def __set_up_function_table(self, ast):
        self.func_name_to_ast = {}
        for func_def in ast.get("functions"):
            func_name = func_def.get("name")
            num_params = len(func_def.get("args"))
            return_type = func_def.get("return_type")
            param_types = [arg.get("var_type") for arg in func_def.get("args")]

            # validate before adding to the self.func_name_to_ast
            self.__validate_func_types(param_types, return_type)

            if func_name not in self.func_name_to_ast:
                self.func_name_to_ast[func_name] = {}

            # add func def w its param types and return type
            self.func_name_to_ast[func_name][num_params] = {
                "def": func_def,
                "param_types": param_types,
                "return_type": return_type,
            }

    def __get_func_by_name(self, name, num_params):
        if name not in self.func_name_to_ast:
            super().error(ErrorType.NAME_ERROR, f"Function {name} not found")
        candidate_funcs = self.func_name_to_ast[name]
        if num_params not in candidate_funcs:
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {name} taking {num_params} params not found",
            )
        return candidate_funcs[num_params]

    def __run_statements(self, statements, return_type=None):
        self.env.push_block()
        for statement in statements:
            if self.trace_output:
                print(statement)
            status, return_val = self.__run_statement(statement, return_type)
            if status == ExecStatus.RETURN:
                self.env.pop_block()
                return (status, return_val)

        self.env.pop_block()
        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __run_statement(self, statement, return_type=None): # 🍅: added an optional return type parameter
        status = ExecStatus.CONTINUE
        return_val = None
        if statement.elem_type == InterpreterBase.FCALL_NODE:
            self.__call_func(statement)
        elif statement.elem_type == "=":
            self.__assign(statement)
        elif statement.elem_type == InterpreterBase.VAR_DEF_NODE:
            self.__var_def(statement)
        elif statement.elem_type == InterpreterBase.RETURN_NODE:
            status, return_val = self.__do_return(statement, return_type)
        elif statement.elem_type == Interpreter.IF_NODE:
            status, return_val = self.__do_if(statement)
        elif statement.elem_type == Interpreter.FOR_NODE:
            status, return_val = self.__do_for(statement)

        return (status, return_val)
    
    def __call_func(self, call_node):
        func_name = call_node.get("name")
        actual_args = call_node.get("args")
        return self.__call_func_aux(func_name, actual_args)

    def __call_func_aux(self, func_name, actual_args):
        if func_name == "print":
            return self.__call_print(actual_args)
        if func_name == "inputi" or func_name == "inputs":
            return self.__call_input(func_name, actual_args)
        
        func_info = self.__get_func_by_name(func_name, len(actual_args))
        func_ast = func_info["def"]
        formal_args = func_ast.get("args")
        param_types = func_info["param_types"]
        return_type = func_info["return_type"]

        # prepare args to pass to the func (by type checking)
        prepared_args = {}
        for i, (formal_arg, actual_arg) in enumerate(zip(formal_args, actual_args)):
            actual_value = self.__eval_expr(actual_arg)
            actual_type = actual_value.type()
            expected_type = param_types[i]
            arg_name = formal_arg.get("name")

            # TYPE CHECKING:
            # CASE 1: same type, pass directly
            if actual_type == expected_type or (
                expected_type in self.struct_definitions and 
                isinstance(actual_value, StructValue) and
                actual_value.struct_type.name in self.struct_definitions and 
                actual_value.struct_type.name == expected_type
            ):
                if expected_type in ["int", "string", "bool"]:  # primitive
                    prepared_args[arg_name] = copy.copy(actual_value)
                else:  # struct (pass by reference)
                    prepared_args[arg_name] = actual_value

            # CASE 2: coercion: int -> bool
            elif expected_type == Type.BOOL:
                actual_value = self.coerce_int_to_bool(actual_value)
                # TYPE CHECK: after possible coercion
                if actual_value.type() != expected_type:
                    super().error(
                        ErrorType.TYPE_ERROR,
                        f"Type mismatch: Expected {expected_type}, got {actual_value.type()} for argument '{arg_name}'")
                prepared_args[arg_name] = actual_value

            # CASE 3: nil can be assigned to struct param
            elif expected_type in self.struct_definitions and actual_type == Type.NIL:
                prepared_args[arg_name] = actual_value
            # Type Mismatch
            else:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Type mismatch: Expected {expected_type}, got {actual_value.type()} for argument '{arg_name}'")
                
        # CHECK: func w/ arg len exists
        if len(actual_args) != len(formal_args):
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {func_ast.get('name')} with {len(actual_args)} args not found")

        # create a new func scope
        self.env.push_func()

        # bind formal params to the prepared args
        for arg_name, value in prepared_args.items():
            self.env.create(arg_name, value)

        # execute rest of func statements
        status, return_val_obj = self.__run_statements(func_ast.get("statements"), return_type)
        self.env.pop_func()

        # handle default ret val if the func runs to completion
        if status != ExecStatus.RETURN:
            return_val_obj = self.__get_default_val("return", return_type)

        # CHECK: return type
        if return_type != "void":
            if isinstance(return_val_obj, StructValue):
                if return_val_obj.struct_type.name != return_type:
                    super().error(
                        ErrorType.TYPE_ERROR,
                        f"Type mismatch in return value: Expected {return_type}, got {return_val_obj.struct_type.name}")
            elif return_val_obj.type() == Type.NIL and return_type in self.struct_definitions:
                # allow nil to be returned for a struct type
                pass
            elif return_val_obj.value() == Interpreter.VOID_VALUE.value() or return_val_obj.type() != return_type:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Type mismatch in return value: Expected {return_type}, got {return_val_obj.type()}")


        return return_val_obj if return_type != "void" else Interpreter.VOID_VALUE

    def __call_print(self, args):
        output = ""
        for arg in args:
            result = self.__eval_expr(arg)  # result is a Value object

            # CHECK: val is of type nil
            if result.type() == Type.NIL:
                output += "nil"
            else:
                output = output + get_printable(result)
        super().output(output)
        # return Interpreter.NIL_VALUE # bc print is now VOID!!

    def __call_input(self, name, args):
        if args is not None and len(args) == 1:
            result = self.__eval_expr(args[0])
            super().output(get_printable(result))
        elif args is not None and len(args) > 1:
            super().error(
                ErrorType.NAME_ERROR, "No inputi() function that takes > 1 parameter"
            )
        inp = super().get_input()
        if name == "inputi":
            return Value(Type.INT, int(inp))
        if name == "inputs":
            return Value(Type.STRING, inp)
            
    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        value_obj = self.__eval_expr(assign_ast.get("expression"))

        # handle if var name is a struct field (contains '.')
        if "." in var_name:
            field_path = var_name.split(".")
            struct_instance = self.__resolve_nested_struct(field_path[:-1])
            field_name = field_path[-1]

            # CHECK: if the struct instance is valid
            if struct_instance is None or struct_instance.type() == Type.NIL:
                super().error(ErrorType.FAULT_ERROR, f"Attempted to assign a field on a nil reference '{field_path[:-1]}'.")
            if not isinstance(struct_instance, StructValue):
                super().error(ErrorType.TYPE_ERROR, f"Variable '{field_path[:-1]}' is not a struct.")

            # get the expected field type
            struct_type = struct_instance.struct_type
            if field_name not in struct_type.fields:
                super().error(ErrorType.NAME_ERROR, f"Field '{field_name}' does not exist in struct '{struct_type.name}'.")

            expected_field_type = struct_type.fields[field_name]
            value_type = value_obj.type()

            if isinstance(value_obj, StructValue):
                if value_obj.struct_type.name != expected_field_type:
                    super().error(ErrorType.TYPE_ERROR, f"Type mismatch: Cannot assign struct instance of type '{value_obj.struct_type.name}' to field '{field_name}' of type '{expected_field_type}'.")
            
            # coercion: int -> bool
            elif expected_field_type == Type.BOOL:
                value_obj = self.coerce_int_to_bool(value_obj)
            
            elif value_type != expected_field_type and not (expected_field_type in self.struct_definitions and value_type == Type.NIL):
                super().error(ErrorType.TYPE_ERROR, f"Type mismatch: Cannot assign '{value_type}' to field '{field_name}' of type '{expected_field_type}'.")

            # perform the field assignment
            struct_instance.set_field(field_name, value_obj)
            return

        # regular var assignment (non-struct)
        # target is the var being changed
        target_value = self.env.get(var_name)
        if target_value is None:
            super().error(ErrorType.NAME_ERROR, f"Undefined variable '{var_name}' in assignment")

        target_type = target_value.type()
        source_type = value_obj.type()

        # CASE 1: same type assignment (primitive or struct)
        if target_type == source_type:
            self.env.set(var_name, value_obj)
            return

        # CASE 2: struct assignment (assigning a new struct instance to a var intialized w nil)
        if (target_type in self.struct_definitions or target_type == Type.NIL) and isinstance(value_obj, StructValue):
            # ensure the struct types match
            if value_obj.struct_type.name != target_type and target_type != Type.NIL:
                super().error(ErrorType.TYPE_ERROR, f"Type mismatch: Cannot assign struct instance of type '{value_obj.struct_type.name}' to variable of type '{target_type}'.")
            self.env.set(var_name, value_obj)
            return

        # CASE 3: struct can be assigned nil
        if target_type in self.struct_definitions or source_type == Type.NIL:
            self.env.set(var_name, value_obj)
            return

        # CASE 4: coercion from int -> bool
        if target_type == Type.BOOL and source_type == Type.INT:
            coerced_value = self.coerce_int_to_bool(value_obj)
            # ⭐️ ⭐️ ⭐️ regular type-checking logic
            if target_type != coerced_value.type():
                super().error(ErrorType.TYPE_ERROR, f"Type mismatch: Cannot assign '{source_type}' to '{target_type}'")

            self.env.set(var_name, coerced_value)
            return

        # mismatch type error
        super().error(ErrorType.TYPE_ERROR, f"Type mismatch: Cannot assign '{source_type}' to '{target_type}'")
        
    def __var_def(self, var_ast):
        var_name = var_ast.get("name")
        var_type = var_ast.get("var_type")

        # validate the var type
        valid_types = ["int", "string", "bool"] + list(self.struct_definitions.keys())
        if var_type not in valid_types:
            super().error(ErrorType.TYPE_ERROR, f"Invalid type '{var_type}' for variable '{var_name}'")

        # get default val based on type
        default_value = self.__get_default_val(var_name, var_type)

        # create the var in the env w the default val
        if not self.env.create(var_name, default_value):
            super().error(ErrorType.NAME_ERROR, f"Duplicate definition for variable '{var_name}'")

    def __eval_expr(self, expr_ast):
        if expr_ast.elem_type == InterpreterBase.NIL_NODE:
            return Interpreter.NIL_VALUE
        if expr_ast.elem_type == InterpreterBase.INT_NODE:
            return Value(Type.INT, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.STRING_NODE:
            return Value(Type.STRING, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.BOOL_NODE:
            return Value(Type.BOOL, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.VAR_NODE:
            var_name = expr_ast.get("name")

            # 🍅: DO I NEED THIS FOR THE STRUCT FIELD ASSIGNMENT??
            # check if var name is a struct field
            if "." in var_name:
                field_path = var_name.split(".")
                struct_instance = self.__resolve_nested_struct(field_path[:-1])
                field_name = field_path[-1]

                if struct_instance is None or struct_instance.type() == Type.NIL:
                    super().error(ErrorType.FAULT_ERROR, f"Attempted to access a field on a nil reference: '{field_path[:-1]}'.")
                if not isinstance(struct_instance, StructValue):
                    super().error(ErrorType.TYPE_ERROR, f"Variable '{field_path[:-1]}' is not a struct.")

                return struct_instance.get_field(field_name)

            # regular var access
            val = self.env.get(var_name)
            if val is None:
                super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
            return val
        if expr_ast.elem_type == InterpreterBase.FCALL_NODE:
            return self.__call_func(expr_ast)
        if expr_ast.elem_type in Interpreter.BIN_OPS:
            return self.__eval_op(expr_ast)
        if expr_ast.elem_type == Interpreter.NEG_NODE:
            return self.__eval_unary(expr_ast, Type.INT, lambda x: -1 * x)
        if expr_ast.elem_type == Interpreter.NOT_NODE:
            return self.__eval_unary(expr_ast, Type.BOOL, lambda x: not x)
        if expr_ast.elem_type == InterpreterBase.NEW_NODE:
            struct_type_name = expr_ast.get("var_type")
            if struct_type_name not in self.struct_definitions:
                super().error(ErrorType.TYPE_ERROR, f"Unknown struct type: '{struct_type_name}'")
            # create a new instance of struct w default vals
            struct_type = self.struct_definitions[struct_type_name]
            struct_instance = StructValue(struct_type, self.struct_definitions)
            return struct_instance

    def __eval_op(self, arith_ast):
        left_value_obj = self.__eval_expr(arith_ast.get("op1"))
        right_value_obj = self.__eval_expr(arith_ast.get("op2"))
        op = arith_ast.elem_type

        # CHECK: invalid comparison w/ VOID
        if left_value_obj.type() == "void" or right_value_obj.type() == "void":
            super().error(ErrorType.TYPE_ERROR, "Invalid comparison involving void type")

        if op in ["&&", "||"]:
            # coerce int -> bool for logical op
            left_value_obj = self.coerce_int_to_bool(left_value_obj)
            right_value_obj = self.coerce_int_to_bool(right_value_obj)

        if op in ["==", "!="]:
            # coerce int -> bool for equality
            if left_value_obj.type() == Type.INT and right_value_obj.type() == Type.BOOL:
                left_value_obj = self.coerce_int_to_bool(left_value_obj)
            elif left_value_obj.type() == Type.BOOL and right_value_obj.type() == Type.INT:
                right_value_obj = self.coerce_int_to_bool(right_value_obj)

            # CHECK: invalid comparison w/ nil
            if (left_value_obj.type() == Type.NIL and right_value_obj.type() not in self.struct_definitions and right_value_obj.type() != Type.NIL) or \
                (right_value_obj.type() == Type.NIL and left_value_obj.type() not in self.struct_definitions and left_value_obj.type() != Type.NIL):
                    super().error(ErrorType.TYPE_ERROR, "Invalid comparison between nil and non-struct type")


        # ⭐️ ⭐️ ⭐️: ERROR: unsuporrted coercions (ex: false < 5)
        if op not in ["==", "!=", "&&", "||"] and (
            (left_value_obj.type() == Type.BOOL and right_value_obj.type() == Type.INT) or
            (left_value_obj.type() == Type.INT and right_value_obj.type() == Type.BOOL)
        ):
            super().error(ErrorType.TYPE_ERROR, f"Invalid coercion int -> bool for op {op}")


        if not self.__compatible_types(
            op, left_value_obj, right_value_obj
        ):
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible types for {op} operation",
            )
        if op not in self.op_to_lambda[left_value_obj.type()]:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible operator {op} for type {left_value_obj.type()}",
            )
        f = self.op_to_lambda[left_value_obj.type()][op]
        return f(left_value_obj, right_value_obj)

    def __compatible_types(self, oper, obj1, obj2):
        # DOCUMENT: allow comparisons ==/!= of anything against anything
        if oper in ["==", "!="]:
            return True
        return obj1.type() == obj2.type()

    def __eval_unary(self, arith_ast, t, f):
        value_obj = self.__eval_expr(arith_ast.get("op1"))
        if value_obj.type() != t:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible type for {arith_ast.elem_type} operation",
            )
        return Value(t, f(value_obj.value()))

    def __setup_ops(self):
        self.op_to_lambda = {}
        # set up operations on integers
        self.op_to_lambda[Type.INT] = {}
        self.op_to_lambda[Type.INT]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.INT]["-"] = lambda x, y: Value(
            x.type(), x.value() - y.value()
        )
        self.op_to_lambda[Type.INT]["*"] = lambda x, y: Value(
            x.type(), x.value() * y.value()
        )
        self.op_to_lambda[Type.INT]["/"] = lambda x, y: Value(
            x.type(), x.value() // y.value()
        )
        self.op_to_lambda[Type.INT]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.INT]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )
        self.op_to_lambda[Type.INT]["<"] = lambda x, y: Value(
            Type.BOOL, x.value() < y.value()
        )
        self.op_to_lambda[Type.INT]["<="] = lambda x, y: Value(
            Type.BOOL, x.value() <= y.value()
        )
        self.op_to_lambda[Type.INT][">"] = lambda x, y: Value(
            Type.BOOL, x.value() > y.value()
        )
        self.op_to_lambda[Type.INT][">="] = lambda x, y: Value(
            Type.BOOL, x.value() >= y.value()
        )
        #  set up operations on strings
        self.op_to_lambda[Type.STRING] = {}
        self.op_to_lambda[Type.STRING]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.STRING]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.STRING]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )
        #  set up operations on bools
        self.op_to_lambda[Type.BOOL] = {}
        self.op_to_lambda[Type.BOOL]["&&"] = lambda x, y: Value(
            x.type(), x.value() and y.value()
        )
        self.op_to_lambda[Type.BOOL]["||"] = lambda x, y: Value(
            x.type(), x.value() or y.value()
        )
        self.op_to_lambda[Type.BOOL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.BOOL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

        #  set up operations on nil
        self.op_to_lambda[Type.NIL] = {}
        self.op_to_lambda[Type.NIL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.NIL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

    def __do_if(self, if_ast):
        cond_ast = if_ast.get("condition")
        result = self.__eval_expr(cond_ast)
        # ⭐️ ⭐️ ⭐️ coerce int -> bool if needed
        result = self.coerce_int_to_bool(result)
        if result.type() != Type.BOOL:
            super().error(
                ErrorType.TYPE_ERROR,
                "Incompatible type for if condition",
            )
        if result.value():
            statements = if_ast.get("statements")
            status, return_val = self.__run_statements(statements)
            return (status, return_val)
        else:
            else_statements = if_ast.get("else_statements")
            if else_statements is not None:
                status, return_val = self.__run_statements(else_statements)
                return (status, return_val)

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_for(self, for_ast):
        init_ast = for_ast.get("init") 
        cond_ast = for_ast.get("condition")
        update_ast = for_ast.get("update") 

        self.__run_statement(init_ast)  # initialize counter variable
        run_for = Interpreter.TRUE_VALUE
        while run_for.value():
            run_for = self.__eval_expr(cond_ast)  # check for-loop condition
            # ⭐️ ⭐️ ⭐️: coerce int -> bool if needed
            run_for = self.coerce_int_to_bool(run_for)
            if run_for.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    "Incompatible type for for condition",
                )
            if run_for.value():
                statements = for_ast.get("statements")
                status, return_val = self.__run_statements(statements)
                if status == ExecStatus.RETURN:
                    return status, return_val
                self.__run_statement(update_ast)  # update counter variable

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_return(self, return_ast, return_type):
        expr_ast = return_ast.get("expression")

        # if there is no return expression, return the default value for the return type
        if expr_ast is None:
            return (ExecStatus.RETURN, self.__get_default_val("return", return_type))

        return_val_obj = copy.copy(self.__eval_expr(expr_ast))
        return_val_type = return_val_obj.type()

        # TYPE CHECKING
        # CASE 1: exact type match
        if return_val_type == return_type:
            return (ExecStatus.RETURN, return_val_obj)
        
        # CASE 2: return struct instance (by obj ref)
        if return_type in self.struct_definitions and isinstance(return_val_obj, StructValue):
            if return_val_obj.struct_type.name == return_type:
                return (ExecStatus.RETURN, return_val_obj)

        # CASE 3: coercion from int -> bool
        if return_type == Type.BOOL and return_val_type == Type.INT:
            coerced_value = Value(Type.BOOL, return_val_obj.value() != 0)
            return (ExecStatus.RETURN, coerced_value)

        # CASE 4: return nil for a user-defined structure
        if return_type in self.struct_definitions and return_val_type == Type.NIL:
            return (ExecStatus.RETURN, return_val_obj)

        # if none of the cases match, raise a type error
        super().error(
            ErrorType.TYPE_ERROR,
            f"Type mismatch in return value: Expected {return_type}, got {return_val_type}")

        return (ExecStatus.RETURN, return_val_obj)

    def __validate_func_types(self, param_types, return_type):
        # 🍅 🍅 🍅 🍅 🍅 🍅 🍅 🍅 🍅 🍅 IMPLEMENT STRUCTS LATER
        valid_types = ["int", "string", "bool", "void"] + list(self.struct_definitions.keys())

        # validate parameter types
        for param_type in param_types:
            if param_type not in valid_types:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Invalid parameter type: {param_type}")

        # validate return type
        if return_type not in valid_types:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Invalid return type: {return_type}")

    def __get_default_val(self, var_name, var_type):
        if var_type == "int":
            default_value = Value(Type.INT, 0)
        elif var_type == "string":
            default_value = Value(Type.STRING, "")
        elif var_type == "bool":
            default_value = Value(Type.BOOL, False)
        elif var_type in self.struct_definitions:
            # 🍅: should be fine actually?
            default_value = Interpreter.NIL_VALUE
        elif var_type == "void":
            return Interpreter.VOID_VALUE
        else:
            # This case should be unreachable because of the earlier type check
            super().error(ErrorType.TYPE_ERROR, f"Unknown type '{var_type}' for variable '{var_name}'")
        return default_value

    def __set_up_struct_definitions(self, ast):
        for struct_node in ast.get("structs"):
            struct_name = struct_node.get("name")
            fields = {}
            for field_node in struct_node.get("fields"):
                field_name = field_node.get("name")
                field_type = field_node.get("var_type")
                fields[field_name] = field_type
            self.struct_definitions[struct_name] = StructType(struct_name, fields)

    def __resolve_nested_struct(self, field_path):
        struct_instance = self.env.get(field_path[0])

        # traverse each field in the path
        for field_name in field_path[1:]:
            # CHECK: if the struct instance is nil before accessing the field
            if struct_instance is None or struct_instance.type() == Type.NIL:
                super().error(ErrorType.FAULT_ERROR, f"Attempted to access a field on a nil reference: '{field_name}'.")
            
            if not isinstance(struct_instance, StructValue):
                super().error(ErrorType.TYPE_ERROR, f"Variable '{field_name}' is not a struct.")

            struct_instance = struct_instance.get_field(field_name)

        return struct_instance

    def coerce_int_to_bool(self, value_obj):
        if value_obj.type() == Type.INT:
            return Value(Type.BOOL, value_obj.value() != 0)
        return value_obj

def main():
  program = """
struct person {
  name : string;
}
func incorrect() : int{
  var x : int;
  return 9;
}
func correct() : person{
  print("i should print");
  return nil;
}
func main() : void{
  print("hi");
  correct();
  incorrect();
}

/*
*OUT*
hi
i should print
*OUT*
*/
                """
  interpreter = Interpreter()
  interpreter.run(program)


if __name__ == "__main__":
    main()