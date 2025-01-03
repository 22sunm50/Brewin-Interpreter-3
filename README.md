# Brewin++ Interpreter
This repository contains the implementation of an interpreter for the Brewin++ programming language. Brewin++ extends the original Brewin language with static typing, user-defined structures, and enhanced runtime features.

## Features
### Brewin ++
- **Static Typing**: All variables, parameters, and functions have explicit types.
- **User-Defined Structures**: Support for creating and using custom data types (e.g., structs).
- **Default Return Values**: Functions automatically return a type-specific default if no explicit return statement is provided.
- **Type Coercion**: Limited coercion from integers to booleans.
- **Enhanced Control Flow**: Support for if, else, for loops, and nested scopes.
### Core Capabilities
- **Basic Data Types**: Supports integers, booleans, strings, and custom structures.
- **Expressions**: Arithmetic, logical, and comparison operations.
- **Function Calls**: Parameter passing by value or reference with recursion support.
- **Error Handling**: Comprehensive runtime checks for type mismatches, undefined variables, and invalid operations.

## Repository Structure
- **interpreterv3.py**: Main interpreter implementation for Brewin++.
- **element.py**: Abstract Syntax Tree (AST) node definitions.
- **Supporting Modules**: Additional Python files for variable management, statement handling, and type checking.
- **README.md**: This documentation file.

## Usage
### Prerequisites
- Python 3.11
- Ensure intbase.py, brewparse.py, and brewlex.py are in the project directory.

### Running the Interpreter
To run a Brewin program within `interpreterv1.py` or `interpreterv2.py`:
1. Create an instance of the Interpreter class.
2. Pass a Brewin program as a string to the run() method.
   For Example:
  ```
  def main():
    program = """
  func foo(a) {
    print("a: ", a);
    return a + 1;
  }
  
  func bar(b) {
    print(b);
  }
  
  func main() {
    var x;
    x = foo(5);
    bar(x);
  }
  
  /*
  *OUT*
  a: 5
  6
  *OUT*
  */
                  """
    interpreter = Interpreter()
    interpreter.run(program)
  
  
  if __name__ == "__main__":
      main()
  ```
## Test Cases
For running test cases, please refer to this repository: https://github.com/22sunm50/Brewin-Interpreter-Tests
