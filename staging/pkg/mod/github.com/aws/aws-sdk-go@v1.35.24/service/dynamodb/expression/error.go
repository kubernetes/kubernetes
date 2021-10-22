package expression

import (
	"fmt"
)

// InvalidParameterError is returned if invalid parameters are encountered. This
// error specifically refers to situations where parameters are non-empty but
// have an invalid syntax/format. The error message includes the function
// that returned the error originally and the parameter type that was deemed
// invalid.
//
// Example:
//
//     // err is of type InvalidParameterError
//     _, err := expression.Name("foo..bar").BuildOperand()
type InvalidParameterError struct {
	parameterType string
	functionName  string
}

func (ipe InvalidParameterError) Error() string {
	return fmt.Sprintf("%s error: invalid parameter: %s", ipe.functionName, ipe.parameterType)
}

func newInvalidParameterError(funcName, paramType string) InvalidParameterError {
	return InvalidParameterError{
		parameterType: paramType,
		functionName:  funcName,
	}
}

// UnsetParameterError is returned if parameters are empty and uninitialized.
// This error is returned if opaque structs (ConditionBuilder, NameBuilder,
// Builder, etc) are initialized outside of functions in the package, since all
// structs in the package are designed to be initialized with functions.
//
// Example:
//
//     // err is of type UnsetParameterError
//     _, err := expression.Builder{}.Build()
//     _, err := expression.NewBuilder().
//                 WithCondition(expression.ConditionBuilder{}).
//                 Build()
type UnsetParameterError struct {
	parameterType string
	functionName  string
}

func (upe UnsetParameterError) Error() string {
	return fmt.Sprintf("%s error: unset parameter: %s", upe.functionName, upe.parameterType)
}

func newUnsetParameterError(funcName, paramType string) UnsetParameterError {
	return UnsetParameterError{
		parameterType: paramType,
		functionName:  funcName,
	}
}
