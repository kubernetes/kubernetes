package hil

import (
	"strconv"

	"github.com/hashicorp/hil/ast"
)

// NOTE: All builtins are tested in engine_test.go

func registerBuiltins(scope *ast.BasicScope) *ast.BasicScope {
	if scope == nil {
		scope = new(ast.BasicScope)
	}
	if scope.FuncMap == nil {
		scope.FuncMap = make(map[string]ast.Function)
	}

	// Implicit conversions
	scope.FuncMap["__builtin_FloatToInt"] = builtinFloatToInt()
	scope.FuncMap["__builtin_FloatToString"] = builtinFloatToString()
	scope.FuncMap["__builtin_IntToFloat"] = builtinIntToFloat()
	scope.FuncMap["__builtin_IntToString"] = builtinIntToString()
	scope.FuncMap["__builtin_StringToInt"] = builtinStringToInt()
	scope.FuncMap["__builtin_StringToFloat"] = builtinStringToFloat()

	// Math operations
	scope.FuncMap["__builtin_IntMath"] = builtinIntMath()
	scope.FuncMap["__builtin_FloatMath"] = builtinFloatMath()
	return scope
}

func builtinFloatMath() ast.Function {
	return ast.Function{
		ArgTypes:     []ast.Type{ast.TypeInt},
		Variadic:     true,
		VariadicType: ast.TypeFloat,
		ReturnType:   ast.TypeFloat,
		Callback: func(args []interface{}) (interface{}, error) {
			op := args[0].(ast.ArithmeticOp)
			result := args[1].(float64)
			for _, raw := range args[2:] {
				arg := raw.(float64)
				switch op {
				case ast.ArithmeticOpAdd:
					result += arg
				case ast.ArithmeticOpSub:
					result -= arg
				case ast.ArithmeticOpMul:
					result *= arg
				case ast.ArithmeticOpDiv:
					result /= arg
				}
			}

			return result, nil
		},
	}
}

func builtinIntMath() ast.Function {
	return ast.Function{
		ArgTypes:     []ast.Type{ast.TypeInt},
		Variadic:     true,
		VariadicType: ast.TypeInt,
		ReturnType:   ast.TypeInt,
		Callback: func(args []interface{}) (interface{}, error) {
			op := args[0].(ast.ArithmeticOp)
			result := args[1].(int)
			for _, raw := range args[2:] {
				arg := raw.(int)
				switch op {
				case ast.ArithmeticOpAdd:
					result += arg
				case ast.ArithmeticOpSub:
					result -= arg
				case ast.ArithmeticOpMul:
					result *= arg
				case ast.ArithmeticOpDiv:
					result /= arg
				case ast.ArithmeticOpMod:
					result = result % arg
				}
			}

			return result, nil
		},
	}
}

func builtinFloatToInt() ast.Function {
	return ast.Function{
		ArgTypes:   []ast.Type{ast.TypeFloat},
		ReturnType: ast.TypeInt,
		Callback: func(args []interface{}) (interface{}, error) {
			return int(args[0].(float64)), nil
		},
	}
}

func builtinFloatToString() ast.Function {
	return ast.Function{
		ArgTypes:   []ast.Type{ast.TypeFloat},
		ReturnType: ast.TypeString,
		Callback: func(args []interface{}) (interface{}, error) {
			return strconv.FormatFloat(
				args[0].(float64), 'g', -1, 64), nil
		},
	}
}

func builtinIntToFloat() ast.Function {
	return ast.Function{
		ArgTypes:   []ast.Type{ast.TypeInt},
		ReturnType: ast.TypeFloat,
		Callback: func(args []interface{}) (interface{}, error) {
			return float64(args[0].(int)), nil
		},
	}
}

func builtinIntToString() ast.Function {
	return ast.Function{
		ArgTypes:   []ast.Type{ast.TypeInt},
		ReturnType: ast.TypeString,
		Callback: func(args []interface{}) (interface{}, error) {
			return strconv.FormatInt(int64(args[0].(int)), 10), nil
		},
	}
}

func builtinStringToInt() ast.Function {
	return ast.Function{
		ArgTypes:   []ast.Type{ast.TypeInt},
		ReturnType: ast.TypeString,
		Callback: func(args []interface{}) (interface{}, error) {
			v, err := strconv.ParseInt(args[0].(string), 0, 0)
			if err != nil {
				return nil, err
			}

			return int(v), nil
		},
	}
}

func builtinStringToFloat() ast.Function {
	return ast.Function{
		ArgTypes:   []ast.Type{ast.TypeString},
		ReturnType: ast.TypeFloat,
		Callback: func(args []interface{}) (interface{}, error) {
			v, err := strconv.ParseFloat(args[0].(string), 64)
			if err != nil {
				return nil, err
			}

			return v, nil
		},
	}
}
