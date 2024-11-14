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
    BIN_OPS = {"+", "-", "*", "/", "==", "!=", ">", ">=", "<", "<=", "||", "&&"}

    # methods
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.__setup_ops()
        self.struct_definitions = {}  # dict of all user-defined struct types

    def run(self, program):
        ast = parse_program(program)
        self.__set_up_function_table(ast)
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

    def __run_statements(self, statements):
        self.env.push_block()
        for statement in statements:
            if self.trace_output:
                print(statement)
            status, return_val = self.__run_statement(statement)
            if status == ExecStatus.RETURN:
                self.env.pop_block()
                return (status, return_val)

        self.env.pop_block()
        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __run_statement(self, statement):
        status = ExecStatus.CONTINUE
        return_val = None
        if statement.elem_type == InterpreterBase.FCALL_NODE:
            self.__call_func(statement)
        elif statement.elem_type == "=":
            self.__assign(statement)
        elif statement.elem_type == InterpreterBase.VAR_DEF_NODE:
            self.__var_def(statement)
        elif statement.elem_type == InterpreterBase.RETURN_NODE:
            status, return_val = self.__do_return(statement)
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

        # CHECK: func w/ arg len exists
        if len(actual_args) != len(formal_args):
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {func_ast.get('name')} with {len(actual_args)} args not found")

        # CHECK: if arg type match expected param types
        for i, (formal_arg, actual_arg) in enumerate(zip(formal_args, actual_args)):
            actual_value = copy.copy(self.__eval_expr(actual_arg)) # create new Value obj that references the same underlying data
            expected_type = param_types[i]

            if actual_value.type() != expected_type:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Type mismatch: Expected {expected_type}, got {actual_value.type()} for argument '{formal_arg.get('name')}'")

        # create new activation record and bind formal arguments to actual values
        self.env.push_func()
        for formal_arg, actual_arg in zip(formal_args, actual_args):
            arg_name = formal_arg.get("name")
            arg_value = copy.copy(self.__eval_expr(actual_arg)) # 🍅: CHECK BACK IF EVAL EXPR TWICE IS AN ISSUE
            self.env.create(arg_name, arg_value)

        # execute func statements
        status, return_val = self.__run_statements(func_ast.get("statements"))
        self.env.pop_func()

        # CHECK: return type
        if (return_type != "void" and (return_val is None or return_val.type() != return_type)):
            super().error(
                ErrorType.TYPE_ERROR,
                f"Type mismatch in return value: Expected {return_type}, got {return_val.type() if return_val else 'nil'}")

        # CHECK: return type void
        if return_type == "void" and return_val.type() != "nil": # 🍅 🍅 🍅: what should return val be if void??
            super().error(
                ErrorType.TYPE_ERROR,
                f"Type mismatch in return value: Expected {return_type}, got {return_val.type() if return_val else 'nil'}")

        return return_val if return_type != "void" else Interpreter.NIL_VALUE

    def __call_print(self, args):
        output = ""
        for arg in args:
            result = self.__eval_expr(arg)  # result is a Value object
            output = output + get_printable(result)
        super().output(output)
        return Interpreter.NIL_VALUE

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
        if not self.env.set(var_name, value_obj):
            super().error(
                ErrorType.NAME_ERROR, f"Undefined variable {var_name} in assignment")
            
    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        value_obj = self.__eval_expr(assign_ast.get("expression"))

        # get the target var's curr val from the env
        target_value = self.env.get(var_name)
        if target_value is None:
            super().error(ErrorType.NAME_ERROR, f"Undefined variable '{var_name}' in assignment")

        target_type = target_value.type()
        source_type = value_obj.type()

        # CASE 1: same type assignment (primitive or struct)
        if target_type == source_type:
            self.env.set(var_name, value_obj)
            return

        # CASE 2: struct can be assigned nil
        if target_type in self.struct_definitions and source_type == Type.NIL: # 🍅 🍅 🍅: or? not necessarily nil i think
            self.env.set(var_name, value_obj)
            return

        # CASE 3: coercion from int -> bool
        if target_type == Type.BOOL and source_type == Type.INT:
            coerced_value = Value(Type.BOOL, value_obj.value() != 0)
            self.env.set(var_name, coerced_value)
            return

        # If none of the valid cases apply, raise a type error
        super().error(ErrorType.TYPE_ERROR, f"Type mismatch: Cannot assign '{source_type}' to '{target_type}'")

          
    def __var_def(self, var_ast):
        var_name = var_ast.get("name")
        var_type = var_ast.get("var_type")

        # validate the var type
        valid_types = ["int", "string", "bool"] + list(self.struct_definitions.keys())
        if var_type not in valid_types:
            super().error(ErrorType.TYPE_ERROR, f"Invalid type '{var_type}' for variable '{var_name}'")

        # get default val based on type
        default_value = self.get_default_val(var_name, var_type)

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

    def __eval_op(self, arith_ast):
        left_value_obj = self.__eval_expr(arith_ast.get("op1"))
        right_value_obj = self.__eval_expr(arith_ast.get("op2"))
        if not self.__compatible_types(
            arith_ast.elem_type, left_value_obj, right_value_obj
        ):
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible types for {arith_ast.elem_type} operation",
            )
        if arith_ast.elem_type not in self.op_to_lambda[left_value_obj.type()]:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible operator {arith_ast.elem_type} for type {left_value_obj.type()}",
            )
        f = self.op_to_lambda[left_value_obj.type()][arith_ast.elem_type]
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

    def __do_return(self, return_ast):
        expr_ast = return_ast.get("expression")
        if expr_ast is None:
            return (ExecStatus.RETURN, Interpreter.NIL_VALUE)
        value_obj = copy.copy(self.__eval_expr(expr_ast))
        return (ExecStatus.RETURN, value_obj)

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
            
    def define_struct(self, struct_name, fields):
        if struct_name in self.struct_definitions:
            raise ValueError(f"Struct '{struct_name}' is already defined.")
        self.struct_definitions[struct_name] = StructType(struct_name, fields)

    def create_struct_instance(self, struct_name):
        if struct_name not in self.struct_definitions:
            raise ValueError(f"Struct '{struct_name}' is not defined.")
        struct_type = self.struct_definitions[struct_name]
        return StructValue(struct_type)

    def get_struct_field(self, struct_instance, field_name):
        if not isinstance(struct_instance, StructValue):
            raise TypeError("Expected a struct instance.")
        return struct_instance.get_field(field_name)

    def set_struct_field(self, struct_instance, field_name, value):
        if not isinstance(struct_instance, StructValue):
            raise TypeError("Expected a struct instance.")
        struct_instance.set_field(field_name, value)

    def get_default_val(self, var_name, var_type):
        if var_type == "int":
            default_value = Value(Type.INT, 0)
        elif var_type == "string":
            default_value = Value(Type.STRING, "")
        elif var_type == "bool":
            default_value = Value(Type.BOOL, False)
        elif var_type in self.struct_definitions:
            default_value = Interpreter.NIL_VALUE
        else:
            # This case should be unreachable because of the earlier type check
            super().error(ErrorType.TYPE_ERROR, f"Unknown type '{var_type}' for variable '{var_name}'")
        return default_value


def main():
  program = """
func foo(coerced:bool): void {
    print(coerced);
}

func main() : void {
    var x: int;
    var y: bool;
    x = 10;         /* Valid (int → int) */
    y = x;          /* Valid (int → bool coercion) */
    print("y = ", y);
    print ("x = ", x);
    y = "hello";    /* Error (string → bool) */
}
                """
  interpreter = Interpreter()
  interpreter.run(program)


if __name__ == "__main__":
    main()