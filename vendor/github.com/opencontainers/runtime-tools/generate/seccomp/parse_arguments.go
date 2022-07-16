package seccomp

import (
	"fmt"
	"strconv"

	rspec "github.com/opencontainers/runtime-spec/specs-go"
)

// parseArguments takes a list of arguments (delimArgs). It parses and fills out
// the argument information and returns a slice of arg structs
func parseArguments(delimArgs []string) ([]rspec.LinuxSeccompArg, error) {
	nilArgSlice := []rspec.LinuxSeccompArg{}
	numberOfArgs := len(delimArgs)

	// No parameters passed with syscall
	if numberOfArgs == 1 {
		return nilArgSlice, nil
	}

	// Correct number of parameters passed with syscall
	if numberOfArgs == 5 {
		syscallIndex, err := strconv.ParseUint(delimArgs[1], 10, 0)
		if err != nil {
			return nilArgSlice, err
		}

		syscallValue, err := strconv.ParseUint(delimArgs[2], 10, 64)
		if err != nil {
			return nilArgSlice, err
		}

		syscallValueTwo, err := strconv.ParseUint(delimArgs[3], 10, 64)
		if err != nil {
			return nilArgSlice, err
		}

		syscallOp, err := parseOperator(delimArgs[4])
		if err != nil {
			return nilArgSlice, err
		}

		argStruct := rspec.LinuxSeccompArg{
			Index:    uint(syscallIndex),
			Value:    syscallValue,
			ValueTwo: syscallValueTwo,
			Op:       syscallOp,
		}

		argSlice := []rspec.LinuxSeccompArg{}
		argSlice = append(argSlice, argStruct)
		return argSlice, nil
	}

	return nilArgSlice, fmt.Errorf("incorrect number of arguments passed with syscall: %d", numberOfArgs)
}

func parseOperator(operator string) (rspec.LinuxSeccompOperator, error) {
	operators := map[string]rspec.LinuxSeccompOperator{
		"NE": rspec.OpNotEqual,
		"LT": rspec.OpLessThan,
		"LE": rspec.OpLessEqual,
		"EQ": rspec.OpEqualTo,
		"GE": rspec.OpGreaterEqual,
		"GT": rspec.OpGreaterThan,
		"ME": rspec.OpMaskedEqual,
	}
	o, ok := operators[operator]
	if !ok {
		return "", fmt.Errorf("unrecognized operator: %s", operator)
	}
	return o, nil
}
