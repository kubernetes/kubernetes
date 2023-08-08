// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package interpreter

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"unicode"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

type typeVerifier func(int64, ...*types.TypeValue) (bool, error)

// InterpolateFormattedString checks the syntax and cardinality of any string.format calls present in the expression and reports
// any errors at compile time.
func InterpolateFormattedString(verifier typeVerifier) InterpretableDecorator {
	return func(inter Interpretable) (Interpretable, error) {
		call, ok := inter.(InterpretableCall)
		if !ok {
			return inter, nil
		}
		if call.OverloadID() != "string_format" {
			return inter, nil
		}
		args := call.Args()
		if len(args) != 2 {
			return nil, fmt.Errorf("wrong number of arguments to string.format (expected 2, got %d)", len(args))
		}
		fmtStrInter, ok := args[0].(InterpretableConst)
		if !ok {
			return inter, nil
		}
		var fmtArgsInter InterpretableConstructor
		fmtArgsInter, ok = args[1].(InterpretableConstructor)
		if !ok {
			return inter, nil
		}
		if fmtArgsInter.Type() != types.ListType {
			// don't necessarily return an error since the list may be DynType
			return inter, nil
		}
		formatStr := fmtStrInter.Value().Value().(string)
		initVals := fmtArgsInter.InitVals()

		formatCheck := &formatCheck{
			args:     initVals,
			verifier: verifier,
		}
		// use a placeholder locale, since locale doesn't affect syntax
		_, err := ParseFormatString(formatStr, formatCheck, formatCheck, "en_US")
		if err != nil {
			return nil, err
		}
		seenArgs := formatCheck.argsRequested
		if len(initVals) > seenArgs {
			return nil, fmt.Errorf("too many arguments supplied to string.format (expected %d, got %d)", seenArgs, len(initVals))
		}
		return inter, nil
	}
}

type formatCheck struct {
	args                []Interpretable
	argsRequested       int
	curArgIndex         int64
	enableCheckArgTypes bool
	verifier            typeVerifier
}

func (c *formatCheck) String(arg ref.Val, locale string) (string, error) {
	valid, err := verifyString(c.args[c.curArgIndex], c.verifier)
	if err != nil {
		return "", err
	}
	if !valid {
		return "", errors.New("string clause can only be used on strings, bools, bytes, ints, doubles, maps, lists, types, durations, and timestamps")
	}
	return "", nil
}

func (c *formatCheck) Decimal(arg ref.Val, locale string) (string, error) {
	id := c.args[c.curArgIndex].ID()
	valid, err := c.verifier(id, types.IntType, types.UintType)
	if err != nil {
		return "", err
	}
	if !valid {
		return "", errors.New("integer clause can only be used on integers")
	}
	return "", nil
}

func (c *formatCheck) Fixed(precision *int) func(ref.Val, string) (string, error) {
	return func(arg ref.Val, locale string) (string, error) {
		id := c.args[c.curArgIndex].ID()
		// we allow StringType since "NaN", "Infinity", and "-Infinity" are also valid values
		valid, err := c.verifier(id, types.DoubleType, types.StringType)
		if err != nil {
			return "", err
		}
		if !valid {
			return "", errors.New("fixed-point clause can only be used on doubles")
		}
		return "", nil
	}
}

func (c *formatCheck) Scientific(precision *int) func(ref.Val, string) (string, error) {
	return func(arg ref.Val, locale string) (string, error) {
		id := c.args[c.curArgIndex].ID()
		valid, err := c.verifier(id, types.DoubleType, types.StringType)
		if err != nil {
			return "", err
		}
		if !valid {
			return "", errors.New("scientific clause can only be used on doubles")
		}
		return "", nil
	}
}

func (c *formatCheck) Binary(arg ref.Val, locale string) (string, error) {
	id := c.args[c.curArgIndex].ID()
	valid, err := c.verifier(id, types.IntType, types.UintType, types.BoolType)
	if err != nil {
		return "", err
	}
	if !valid {
		return "", errors.New("only integers and bools can be formatted as binary")
	}
	return "", nil
}

func (c *formatCheck) Hex(useUpper bool) func(ref.Val, string) (string, error) {
	return func(arg ref.Val, locale string) (string, error) {
		id := c.args[c.curArgIndex].ID()
		valid, err := c.verifier(id, types.IntType, types.UintType, types.StringType, types.BytesType)
		if err != nil {
			return "", err
		}
		if !valid {
			return "", errors.New("only integers, byte buffers, and strings can be formatted as hex")
		}
		return "", nil
	}
}

func (c *formatCheck) Octal(arg ref.Val, locale string) (string, error) {
	id := c.args[c.curArgIndex].ID()
	valid, err := c.verifier(id, types.IntType, types.UintType)
	if err != nil {
		return "", err
	}
	if !valid {
		return "", errors.New("octal clause can only be used on integers")
	}
	return "", nil
}

func (c *formatCheck) Arg(index int64) (ref.Val, error) {
	c.argsRequested++
	c.curArgIndex = index
	// return a dummy value - this is immediately passed to back to us
	// through one of the FormatCallback functions, so anything will do
	return types.Int(0), nil
}

func (c *formatCheck) ArgSize() int64 {
	return int64(len(c.args))
}

func verifyString(sub Interpretable, verifier typeVerifier) (bool, error) {
	subVerified, err := verifier(sub.ID(),
		types.ListType, types.MapType, types.IntType, types.UintType, types.DoubleType,
		types.BoolType, types.StringType, types.TimestampType, types.BytesType, types.DurationType, types.TypeType, types.NullType)
	if err != nil {
		return false, err
	}
	if !subVerified {
		return false, nil
	}
	con, ok := sub.(InterpretableConstructor)
	if ok {
		members := con.InitVals()
		for _, m := range members {
			// recursively verify if we're dealing with a list/map
			verified, err := verifyString(m, verifier)
			if err != nil {
				return false, err
			}
			if !verified {
				return false, nil
			}
		}
	}
	return true, nil

}

// FormatStringInterpolator is an interface that allows user-defined behavior
// for formatting clause implementations, as well as argument retrieval.
// Each function is expected to support the appropriate types as laid out in
// the string.format documentation, and to return an error if given an inappropriate type.
type FormatStringInterpolator interface {
	// String takes a ref.Val and a string representing the current locale identifier
	// and returns the Val formatted as a string, or an error if one occurred.
	String(ref.Val, string) (string, error)

	// Decimal takes a ref.Val and a string representing the current locale identifier
	// and returns the Val formatted as a decimal integer, or an error if one occurred.
	Decimal(ref.Val, string) (string, error)

	// Fixed takes an int pointer representing precision (or nil if none was given) and
	// returns a function operating in a similar manner to String and Decimal, taking a
	// ref.Val and locale and returning the appropriate string. A closure is returned
	// so precision can be set without needing an additional function call/configuration.
	Fixed(*int) func(ref.Val, string) (string, error)

	// Scientific functions identically to Fixed, except the string returned from the closure
	// is expected to be in scientific notation.
	Scientific(*int) func(ref.Val, string) (string, error)

	// Binary takes a ref.Val and a string representing the current locale identifier
	// and returns the Val formatted as a binary integer, or an error if one occurred.
	Binary(ref.Val, string) (string, error)

	// Hex takes a boolean that, if true, indicates the hex string output by the returned
	// closure should use uppercase letters for A-F.
	Hex(bool) func(ref.Val, string) (string, error)

	// Octal takes a ref.Val and a string representing the current locale identifier and
	// returns the Val formatted in octal, or an error if one occurred.
	Octal(ref.Val, string) (string, error)
}

// FormatList is an interface that allows user-defined list-like datatypes to be used
// for formatting clause implementations.
type FormatList interface {
	// Arg returns the ref.Val at the given index, or an error if one occurred.
	Arg(int64) (ref.Val, error)
	// ArgSize returns the length of the argument list.
	ArgSize() int64
}

type clauseImpl func(ref.Val, string) (string, error)

// ParseFormatString formats a string according to the string.format syntax, taking the clause implementations
// from the provided FormatCallback and the args from the given FormatList.
func ParseFormatString(formatStr string, callback FormatStringInterpolator, list FormatList, locale string) (string, error) {
	i := 0
	argIndex := 0
	var builtStr strings.Builder
	for i < len(formatStr) {
		if formatStr[i] == '%' {
			if i+1 < len(formatStr) && formatStr[i+1] == '%' {
				err := builtStr.WriteByte('%')
				if err != nil {
					return "", fmt.Errorf("error writing format string: %w", err)
				}
				i += 2
				continue
			} else {
				argAny, err := list.Arg(int64(argIndex))
				if err != nil {
					return "", err
				}
				if i+1 >= len(formatStr) {
					return "", errors.New("unexpected end of string")
				}
				if int64(argIndex) >= list.ArgSize() {
					return "", fmt.Errorf("index %d out of range", argIndex)
				}
				numRead, val, refErr := parseAndFormatClause(formatStr[i:], argAny, callback, list, locale)
				if refErr != nil {
					return "", refErr
				}
				_, err = builtStr.WriteString(val)
				if err != nil {
					return "", fmt.Errorf("error writing format string: %w", err)
				}
				i += numRead
				argIndex++
			}
		} else {
			err := builtStr.WriteByte(formatStr[i])
			if err != nil {
				return "", fmt.Errorf("error writing format string: %w", err)
			}
			i++
		}
	}
	return builtStr.String(), nil
}

// parseAndFormatClause parses the format clause at the start of the given string with val, and returns
// how many characters were consumed and the substituted string form of val, or an error if one occurred.
func parseAndFormatClause(formatStr string, val ref.Val, callback FormatStringInterpolator, list FormatList, locale string) (int, string, error) {
	i := 1
	read, formatter, err := parseFormattingClause(formatStr[i:], callback)
	i += read
	if err != nil {
		return -1, "", fmt.Errorf("could not parse formatting clause: %s", err)
	}

	valStr, err := formatter(val, locale)
	if err != nil {
		return -1, "", fmt.Errorf("error during formatting: %s", err)
	}
	return i, valStr, nil
}

func parseFormattingClause(formatStr string, callback FormatStringInterpolator) (int, clauseImpl, error) {
	i := 0
	read, precision, err := parsePrecision(formatStr[i:])
	i += read
	if err != nil {
		return -1, nil, fmt.Errorf("error while parsing precision: %w", err)
	}
	r := rune(formatStr[i])
	i++
	switch r {
	case 's':
		return i, callback.String, nil
	case 'd':
		return i, callback.Decimal, nil
	case 'f':
		return i, callback.Fixed(precision), nil
	case 'e':
		return i, callback.Scientific(precision), nil
	case 'b':
		return i, callback.Binary, nil
	case 'x', 'X':
		return i, callback.Hex(unicode.IsUpper(r)), nil
	case 'o':
		return i, callback.Octal, nil
	default:
		return -1, nil, fmt.Errorf("unrecognized formatting clause \"%c\"", r)
	}
}

func parsePrecision(formatStr string) (int, *int, error) {
	i := 0
	if formatStr[i] != '.' {
		return i, nil, nil
	}
	i++
	var buffer strings.Builder
	for {
		if i >= len(formatStr) {
			return -1, nil, errors.New("could not find end of precision specifier")
		}
		if !isASCIIDigit(rune(formatStr[i])) {
			break
		}
		buffer.WriteByte(formatStr[i])
		i++
	}
	precision, err := strconv.Atoi(buffer.String())
	if err != nil {
		return -1, nil, fmt.Errorf("error while converting precision to integer: %w", err)
	}
	return i, &precision, nil
}

func isASCIIDigit(r rune) bool {
	return r <= unicode.MaxASCII && unicode.IsDigit(r)
}
