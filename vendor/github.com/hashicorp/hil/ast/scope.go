package ast

import (
	"fmt"
	"reflect"
)

// Scope is the interface used to look up variables and functions while
// evaluating. How these functions/variables are defined are up to the caller.
type Scope interface {
	LookupFunc(string) (Function, bool)
	LookupVar(string) (Variable, bool)
}

// Variable is a variable value for execution given as input to the engine.
// It records the value of a variables along with their type.
type Variable struct {
	Value interface{}
	Type  Type
}

// NewVariable creates a new Variable for the given value. This will
// attempt to infer the correct type. If it can't, an error will be returned.
func NewVariable(v interface{}) (result Variable, err error) {
	switch v := reflect.ValueOf(v); v.Kind() {
	case reflect.String:
		result.Type = TypeString
	default:
		err = fmt.Errorf("Unknown type: %s", v.Kind())
	}

	result.Value = v
	return
}

// String implements Stringer on Variable, displaying the type and value
// of the Variable.
func (v Variable) String() string {
	return fmt.Sprintf("{Variable (%s): %+v}", v.Type, v.Value)
}

// Function defines a function that can be executed by the engine.
// The type checker will validate that the proper types will be called
// to the callback.
type Function struct {
	// ArgTypes is the list of types in argument order. These are the
	// required arguments.
	//
	// ReturnType is the type of the returned value. The Callback MUST
	// return this type.
	ArgTypes   []Type
	ReturnType Type

	// Variadic, if true, says that this function is variadic, meaning
	// it takes a variable number of arguments. In this case, the
	// VariadicType must be set.
	Variadic     bool
	VariadicType Type

	// Callback is the function called for a function. The argument
	// types are guaranteed to match the spec above by the type checker.
	// The length of the args is strictly == len(ArgTypes) unless Varidiac
	// is true, in which case its >= len(ArgTypes).
	Callback func([]interface{}) (interface{}, error)
}

// BasicScope is a simple scope that looks up variables and functions
// using a map.
type BasicScope struct {
	FuncMap map[string]Function
	VarMap  map[string]Variable
}

func (s *BasicScope) LookupFunc(n string) (Function, bool) {
	if s == nil {
		return Function{}, false
	}

	v, ok := s.FuncMap[n]
	return v, ok
}

func (s *BasicScope) LookupVar(n string) (Variable, bool) {
	if s == nil {
		return Variable{}, false
	}

	v, ok := s.VarMap[n]
	return v, ok
}
