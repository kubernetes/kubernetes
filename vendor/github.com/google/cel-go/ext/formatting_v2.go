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

package ext

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

type clauseImplV2 func(ref.Val) (string, error)

type appendingFormatterV2 struct {
	buf []byte
}

type formattedMapEntryV2 struct {
	key string
	val string
}

func (af *appendingFormatterV2) format(arg ref.Val) error {
	switch arg.Type() {
	case types.BoolType:
		argBool, ok := arg.Value().(bool)
		if !ok {
			return fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.BoolType)
		}
		af.buf = strconv.AppendBool(af.buf, argBool)
		return nil
	case types.IntType:
		argInt, ok := arg.Value().(int64)
		if !ok {
			return fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.IntType)
		}
		af.buf = strconv.AppendInt(af.buf, argInt, 10)
		return nil
	case types.UintType:
		argUint, ok := arg.Value().(uint64)
		if !ok {
			return fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.UintType)
		}
		af.buf = strconv.AppendUint(af.buf, argUint, 10)
		return nil
	case types.DoubleType:
		argDbl, ok := arg.Value().(float64)
		if !ok {
			return fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.DoubleType)
		}
		if math.IsNaN(argDbl) {
			af.buf = append(af.buf, "NaN"...)
			return nil
		}
		if math.IsInf(argDbl, -1) {
			af.buf = append(af.buf, "-Infinity"...)
			return nil
		}
		if math.IsInf(argDbl, 1) {
			af.buf = append(af.buf, "Infinity"...)
			return nil
		}
		af.buf = strconv.AppendFloat(af.buf, argDbl, 'f', -1, 64)
		return nil
	case types.BytesType:
		argBytes, ok := arg.Value().([]byte)
		if !ok {
			return fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.BytesType)
		}
		af.buf = append(af.buf, argBytes...)
		return nil
	case types.StringType:
		argStr, ok := arg.Value().(string)
		if !ok {
			return fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.StringType)
		}
		af.buf = append(af.buf, argStr...)
		return nil
	case types.DurationType:
		argDur, ok := arg.Value().(time.Duration)
		if !ok {
			return fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.DurationType)
		}
		af.buf = strconv.AppendFloat(af.buf, argDur.Seconds(), 'f', -1, 64)
		af.buf = append(af.buf, "s"...)
		return nil
	case types.TimestampType:
		argTime, ok := arg.Value().(time.Time)
		if !ok {
			return fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.TimestampType)
		}
		af.buf = argTime.UTC().AppendFormat(af.buf, time.RFC3339Nano)
		return nil
	case types.NullType:
		af.buf = append(af.buf, "null"...)
		return nil
	case types.TypeType:
		argType, ok := arg.Value().(string)
		if !ok {
			return fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.TypeType)
		}
		af.buf = append(af.buf, argType...)
		return nil
	case types.ListType:
		argList, ok := arg.(traits.Lister)
		if !ok {
			return fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.ListType)
		}
		argIter := argList.Iterator()
		af.buf = append(af.buf, "["...)
		if argIter.HasNext() == types.True {
			if err := af.format(argIter.Next()); err != nil {
				return err
			}
			for argIter.HasNext() == types.True {
				af.buf = append(af.buf, ", "...)
				if err := af.format(argIter.Next()); err != nil {
					return err
				}
			}
		}
		af.buf = append(af.buf, "]"...)
		return nil
	case types.MapType:
		argMap, ok := arg.(traits.Mapper)
		if !ok {
			return fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.MapType)
		}
		argIter := argMap.Iterator()
		ents := []formattedMapEntryV2{}
		for argIter.HasNext() == types.True {
			key := argIter.Next()
			val, ok := argMap.Find(key)
			if !ok {
				return fmt.Errorf("key missing from map: '%s'", key)
			}
			keyStr, err := formatStringV2(key)
			if err != nil {
				return err
			}
			valStr, err := formatStringV2(val)
			if err != nil {
				return err
			}
			ents = append(ents, formattedMapEntryV2{keyStr, valStr})
		}
		sort.SliceStable(ents, func(x, y int) bool {
			return ents[x].key < ents[y].key
		})
		af.buf = append(af.buf, "{"...)
		for i, e := range ents {
			if i > 0 {
				af.buf = append(af.buf, ", "...)
			}
			af.buf = append(af.buf, e.key...)
			af.buf = append(af.buf, ": "...)
			af.buf = append(af.buf, e.val...)
		}
		af.buf = append(af.buf, "}"...)
		return nil
	default:
		return stringFormatErrorV2(runtimeID, arg.Type().TypeName())
	}
}

func formatStringV2(arg ref.Val) (string, error) {
	var fmter appendingFormatterV2
	if err := fmter.format(arg); err != nil {
		return "", err
	}
	return string(fmter.buf), nil
}

type stringFormatterV2 struct{}

// String implements formatStringInterpolatorV2.String.
func (c *stringFormatterV2) String(arg ref.Val) (string, error) {
	return formatStringV2(arg)
}

// Decimal implements formatStringInterpolatorV2.Decimal.
func (c *stringFormatterV2) Decimal(arg ref.Val) (string, error) {
	switch arg.Type() {
	case types.IntType:
		argInt, ok := arg.Value().(int64)
		if !ok {
			return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.IntType)
		}
		return strconv.FormatInt(argInt, 10), nil
	case types.UintType:
		argUint, ok := arg.Value().(uint64)
		if !ok {
			return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.UintType)
		}
		return strconv.FormatUint(argUint, 10), nil
	case types.DoubleType:
		argDbl, ok := arg.Value().(float64)
		if !ok {
			return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.DoubleType)
		}
		if math.IsNaN(argDbl) {
			return "NaN", nil
		}
		if math.IsInf(argDbl, -1) {
			return "-Infinity", nil
		}
		if math.IsInf(argDbl, 1) {
			return "Infinity", nil
		}
		return strconv.FormatFloat(argDbl, 'f', -1, 64), nil
	default:
		return "", decimalFormatErrorV2(runtimeID, arg.Type().TypeName())
	}
}

// Fixed implements formatStringInterpolatorV2.Fixed.
func (c *stringFormatterV2) Fixed(precision int) func(ref.Val) (string, error) {
	return func(arg ref.Val) (string, error) {
		fmtStr := fmt.Sprintf("%%.%df", precision)
		switch arg.Type() {
		case types.IntType:
			argInt, ok := arg.Value().(int64)
			if !ok {
				return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.IntType)
			}
			return fmt.Sprintf(fmtStr, argInt), nil
		case types.UintType:
			argUint, ok := arg.Value().(uint64)
			if !ok {
				return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.UintType)
			}
			return fmt.Sprintf(fmtStr, argUint), nil
		case types.DoubleType:
			argDbl, ok := arg.Value().(float64)
			if !ok {
				return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.DoubleType)
			}
			if math.IsNaN(argDbl) {
				return "NaN", nil
			}
			if math.IsInf(argDbl, -1) {
				return "-Infinity", nil
			}
			if math.IsInf(argDbl, 1) {
				return "Infinity", nil
			}
			return fmt.Sprintf(fmtStr, argDbl), nil
		default:
			return "", fixedPointFormatErrorV2(runtimeID, arg.Type().TypeName())
		}
	}
}

// Scientific implements formatStringInterpolatorV2.Scientific.
func (c *stringFormatterV2) Scientific(precision int) func(ref.Val) (string, error) {
	return func(arg ref.Val) (string, error) {
		fmtStr := fmt.Sprintf("%%1.%de", precision)
		switch arg.Type() {
		case types.IntType:
			argInt, ok := arg.Value().(int64)
			if !ok {
				return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.IntType)
			}
			return fmt.Sprintf(fmtStr, argInt), nil
		case types.UintType:
			argUint, ok := arg.Value().(uint64)
			if !ok {
				return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.UintType)
			}
			return fmt.Sprintf(fmtStr, argUint), nil
		case types.DoubleType:
			argDbl, ok := arg.Value().(float64)
			if !ok {
				return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.DoubleType)
			}
			if math.IsNaN(argDbl) {
				return "NaN", nil
			}
			if math.IsInf(argDbl, -1) {
				return "-Infinity", nil
			}
			if math.IsInf(argDbl, 1) {
				return "Infinity", nil
			}
			return fmt.Sprintf(fmtStr, argDbl), nil
		default:
			return "", scientificFormatErrorV2(runtimeID, arg.Type().TypeName())
		}
	}
}

// Binary implements formatStringInterpolatorV2.Binary.
func (c *stringFormatterV2) Binary(arg ref.Val) (string, error) {
	switch arg.Type() {
	case types.BoolType:
		argBool, ok := arg.Value().(bool)
		if !ok {
			return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.BoolType)
		}
		if argBool {
			return "1", nil
		}
		return "0", nil
	case types.IntType:
		argInt, ok := arg.Value().(int64)
		if !ok {
			return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.IntType)
		}
		return strconv.FormatInt(argInt, 2), nil
	case types.UintType:
		argUint, ok := arg.Value().(uint64)
		if !ok {
			return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.UintType)
		}
		return strconv.FormatUint(argUint, 2), nil
	default:
		return "", binaryFormatErrorV2(runtimeID, arg.Type().TypeName())
	}
}

// Hex implements formatStringInterpolatorV2.Hex.
func (c *stringFormatterV2) Hex(useUpper bool) func(ref.Val) (string, error) {
	return func(arg ref.Val) (string, error) {
		var fmtStr string
		if useUpper {
			fmtStr = "%X"
		} else {
			fmtStr = "%x"
		}
		switch arg.Type() {
		case types.IntType:
			argInt, ok := arg.Value().(int64)
			if !ok {
				return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.IntType)
			}
			return fmt.Sprintf(fmtStr, argInt), nil
		case types.UintType:
			argUint, ok := arg.Value().(uint64)
			if !ok {
				return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.UintType)
			}
			return fmt.Sprintf(fmtStr, argUint), nil
		case types.StringType:
			argStr, ok := arg.Value().(string)
			if !ok {
				return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.StringType)
			}
			return fmt.Sprintf(fmtStr, argStr), nil
		case types.BytesType:
			argBytes, ok := arg.Value().([]byte)
			if !ok {
				return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.BytesType)
			}
			return fmt.Sprintf(fmtStr, argBytes), nil
		default:
			return "", hexFormatErrorV2(runtimeID, arg.Type().TypeName())
		}
	}
}

// Octal implements formatStringInterpolatorV2.Octal.
func (c *stringFormatterV2) Octal(arg ref.Val) (string, error) {
	switch arg.Type() {
	case types.IntType:
		argInt, ok := arg.Value().(int64)
		if !ok {
			return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.IntType)
		}
		return strconv.FormatInt(argInt, 8), nil
	case types.UintType:
		argUint, ok := arg.Value().(uint64)
		if !ok {
			return "", fmt.Errorf("type conversion error from '%s' to '%s'", arg.Type(), types.UintType)
		}
		return strconv.FormatUint(argUint, 8), nil
	default:
		return "", octalFormatErrorV2(runtimeID, arg.Type().TypeName())
	}
}

// stringFormatValidatorV2 implements the cel.ASTValidator interface allowing for static validation
// of string.format calls.
type stringFormatValidatorV2 struct{}

// Name returns the name of the validator.
func (stringFormatValidatorV2) Name() string {
	return "cel.validator.string_format"
}

// Configure implements the ASTValidatorConfigurer interface and augments the list of functions to skip
// during homogeneous aggregate literal type-checks.
func (stringFormatValidatorV2) Configure(config cel.MutableValidatorConfig) error {
	functions := config.GetOrDefault(cel.HomogeneousAggregateLiteralExemptFunctions, []string{}).([]string)
	functions = append(functions, "format")
	return config.Set(cel.HomogeneousAggregateLiteralExemptFunctions, functions)
}

// Validate parses all literal format strings and type checks the format clause against the argument
// at the corresponding ordinal within the list literal argument to the function, if one is specified.
func (stringFormatValidatorV2) Validate(env *cel.Env, _ cel.ValidatorConfig, a *ast.AST, iss *cel.Issues) {
	root := ast.NavigateAST(a)
	formatCallExprs := ast.MatchDescendants(root, matchConstantFormatStringWithListLiteralArgs(a))
	for _, e := range formatCallExprs {
		call := e.AsCall()
		formatStr := call.Target().AsLiteral().Value().(string)
		args := call.Args()[0].AsList().Elements()
		formatCheck := &stringFormatCheckerV2{
			args: args,
			ast:  a,
		}
		// use a placeholder locale, since locale doesn't affect syntax
		_, err := parseFormatStringV2(formatStr, formatCheck, formatCheck)
		if err != nil {
			iss.ReportErrorAtID(getErrorExprID(e.ID(), err), "%v", err)
			continue
		}
		seenArgs := formatCheck.argsRequested
		if len(args) > seenArgs {
			iss.ReportErrorAtID(e.ID(),
				"too many arguments supplied to string.format (expected %d, got %d)", seenArgs, len(args))
		}
	}
}

// stringFormatCheckerV2 implements the formatStringInterpolater interface
type stringFormatCheckerV2 struct {
	args          []ast.Expr
	argsRequested int
	currArgIndex  int64
	ast           *ast.AST
}

// String implements formatStringInterpolatorV2.String.
func (c *stringFormatCheckerV2) String(arg ref.Val) (string, error) {
	formatArg := c.args[c.currArgIndex]
	valid, badID := c.verifyString(formatArg)
	if !valid {
		return "", stringFormatErrorV2(badID, c.typeOf(badID).TypeName())
	}
	return "", nil
}

// Decimal implements formatStringInterpolatorV2.Decimal.
func (c *stringFormatCheckerV2) Decimal(arg ref.Val) (string, error) {
	id := c.args[c.currArgIndex].ID()
	valid := c.verifyTypeOneOf(id, types.IntType, types.UintType, types.DoubleType)
	if !valid {
		return "", decimalFormatErrorV2(id, c.typeOf(id).TypeName())
	}
	return "", nil
}

// Fixed implements formatStringInterpolatorV2.Fixed.
func (c *stringFormatCheckerV2) Fixed(precision int) func(ref.Val) (string, error) {
	return func(arg ref.Val) (string, error) {
		id := c.args[c.currArgIndex].ID()
		valid := c.verifyTypeOneOf(id, types.IntType, types.UintType, types.DoubleType)
		if !valid {
			return "", fixedPointFormatErrorV2(id, c.typeOf(id).TypeName())
		}
		return "", nil
	}
}

// Scientific implements formatStringInterpolatorV2.Scientific.
func (c *stringFormatCheckerV2) Scientific(precision int) func(ref.Val) (string, error) {
	return func(arg ref.Val) (string, error) {
		id := c.args[c.currArgIndex].ID()
		valid := c.verifyTypeOneOf(id, types.IntType, types.UintType, types.DoubleType)
		if !valid {
			return "", scientificFormatErrorV2(id, c.typeOf(id).TypeName())
		}
		return "", nil
	}
}

// Binary implements formatStringInterpolatorV2.Binary.
func (c *stringFormatCheckerV2) Binary(arg ref.Val) (string, error) {
	id := c.args[c.currArgIndex].ID()
	valid := c.verifyTypeOneOf(id, types.BoolType, types.IntType, types.UintType)
	if !valid {
		return "", binaryFormatErrorV2(id, c.typeOf(id).TypeName())
	}
	return "", nil
}

// Hex implements formatStringInterpolatorV2.Hex.
func (c *stringFormatCheckerV2) Hex(useUpper bool) func(ref.Val) (string, error) {
	return func(arg ref.Val) (string, error) {
		id := c.args[c.currArgIndex].ID()
		valid := c.verifyTypeOneOf(id, types.IntType, types.UintType, types.StringType, types.BytesType)
		if !valid {
			return "", hexFormatErrorV2(id, c.typeOf(id).TypeName())
		}
		return "", nil
	}
}

// Octal implements formatStringInterpolatorV2.Octal.
func (c *stringFormatCheckerV2) Octal(arg ref.Val) (string, error) {
	id := c.args[c.currArgIndex].ID()
	valid := c.verifyTypeOneOf(id, types.IntType, types.UintType)
	if !valid {
		return "", octalFormatErrorV2(id, c.typeOf(id).TypeName())
	}
	return "", nil
}

// Arg implements formatListArgs.Arg.
func (c *stringFormatCheckerV2) Arg(index int64) (ref.Val, error) {
	c.argsRequested++
	c.currArgIndex = index
	// return a dummy value - this is immediately passed to back to us
	// through one of the FormatCallback functions, so anything will do
	return types.Int(0), nil
}

// Size implements formatListArgs.Size.
func (c *stringFormatCheckerV2) Size() int64 {
	return int64(len(c.args))
}

func (c *stringFormatCheckerV2) typeOf(id int64) *cel.Type {
	return c.ast.GetType(id)
}

func (c *stringFormatCheckerV2) verifyTypeOneOf(id int64, validTypes ...*cel.Type) bool {
	t := c.typeOf(id)
	if t == cel.DynType {
		return true
	}
	for _, vt := range validTypes {
		// Only check runtime type compatibility without delving deeper into parameterized types
		if t.Kind() == vt.Kind() {
			return true
		}
	}
	return false
}

func (c *stringFormatCheckerV2) verifyString(sub ast.Expr) (bool, int64) {
	paramA := cel.TypeParamType("A")
	paramB := cel.TypeParamType("B")
	subVerified := c.verifyTypeOneOf(sub.ID(),
		cel.ListType(paramA), cel.MapType(paramA, paramB),
		cel.IntType, cel.UintType, cel.DoubleType, cel.BoolType, cel.StringType,
		cel.TimestampType, cel.BytesType, cel.DurationType, cel.TypeType, cel.NullType)
	if !subVerified {
		return false, sub.ID()
	}
	switch sub.Kind() {
	case ast.ListKind:
		for _, e := range sub.AsList().Elements() {
			// recursively verify if we're dealing with a list/map
			verified, id := c.verifyString(e)
			if !verified {
				return false, id
			}
		}
		return true, sub.ID()
	case ast.MapKind:
		for _, e := range sub.AsMap().Entries() {
			// recursively verify if we're dealing with a list/map
			entry := e.AsMapEntry()
			verified, id := c.verifyString(entry.Key())
			if !verified {
				return false, id
			}
			verified, id = c.verifyString(entry.Value())
			if !verified {
				return false, id
			}
		}
		return true, sub.ID()
	default:
		return true, sub.ID()
	}
}

// helper routines for reporting common errors during string formatting static validation and
// runtime execution.

func binaryFormatErrorV2(id int64, badType string) error {
	return newFormatError(id, "only ints, uints, and bools can be formatted as binary, was given %s", badType)
}

func decimalFormatErrorV2(id int64, badType string) error {
	return newFormatError(id, "decimal clause can only be used on ints, uints, and doubles, was given %s", badType)
}

func fixedPointFormatErrorV2(id int64, badType string) error {
	return newFormatError(id, "fixed-point clause can only be used on ints, uints, and doubles, was given %s", badType)
}

func hexFormatErrorV2(id int64, badType string) error {
	return newFormatError(id, "only ints, uints, bytes, and strings can be formatted as hex, was given %s", badType)
}

func octalFormatErrorV2(id int64, badType string) error {
	return newFormatError(id, "octal clause can only be used on ints and uints, was given %s", badType)
}

func scientificFormatErrorV2(id int64, badType string) error {
	return newFormatError(id, "scientific clause can only be used on ints, uints, and doubles, was given %s", badType)
}

func stringFormatErrorV2(id int64, badType string) error {
	return newFormatError(id, "string clause can only be used on strings, bools, bytes, ints, doubles, maps, lists, types, durations, and timestamps, was given %s", badType)
}

// formatStringInterpolatorV2 is an interface that allows user-defined behavior
// for formatting clause implementations, as well as argument retrieval.
// Each function is expected to support the appropriate types as laid out in
// the string.format documentation, and to return an error if given an inappropriate type.
type formatStringInterpolatorV2 interface {
	// String takes a ref.Val and a string representing the current locale identifier
	// and returns the Val formatted as a string, or an error if one occurred.
	String(ref.Val) (string, error)

	// Decimal takes a ref.Val and a string representing the current locale identifier
	// and returns the Val formatted as a decimal integer, or an error if one occurred.
	Decimal(ref.Val) (string, error)

	// Fixed takes an int pointer representing precision (or nil if none was given) and
	// returns a function operating in a similar manner to String and Decimal, taking a
	// ref.Val and locale and returning the appropriate string. A closure is returned
	// so precision can be set without needing an additional function call/configuration.
	Fixed(int) func(ref.Val) (string, error)

	// Scientific functions identically to Fixed, except the string returned from the closure
	// is expected to be in scientific notation.
	Scientific(int) func(ref.Val) (string, error)

	// Binary takes a ref.Val and a string representing the current locale identifier
	// and returns the Val formatted as a binary integer, or an error if one occurred.
	Binary(ref.Val) (string, error)

	// Hex takes a boolean that, if true, indicates the hex string output by the returned
	// closure should use uppercase letters for A-F.
	Hex(bool) func(ref.Val) (string, error)

	// Octal takes a ref.Val and a string representing the current locale identifier and
	// returns the Val formatted in octal, or an error if one occurred.
	Octal(ref.Val) (string, error)
}

// parseFormatString formats a string according to the string.format syntax, taking the clause implementations
// from the provided FormatCallback and the args from the given FormatList.
func parseFormatStringV2(formatStr string, callback formatStringInterpolatorV2, list formatListArgs) (string, error) {
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
				if int64(argIndex) >= list.Size() {
					return "", fmt.Errorf("index %d out of range", argIndex)
				}
				numRead, val, refErr := parseAndFormatClauseV2(formatStr[i:], argAny, callback, list)
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
func parseAndFormatClauseV2(formatStr string, val ref.Val, callback formatStringInterpolatorV2, list formatListArgs) (int, string, error) {
	i := 1
	read, formatter, err := parseFormattingClauseV2(formatStr[i:], callback)
	i += read
	if err != nil {
		return -1, "", newParseFormatError("could not parse formatting clause", err)
	}

	valStr, err := formatter(val)
	if err != nil {
		return -1, "", newParseFormatError("error during formatting", err)
	}
	return i, valStr, nil
}

func parseFormattingClauseV2(formatStr string, callback formatStringInterpolatorV2) (int, clauseImplV2, error) {
	i := 0
	read, precision, err := parsePrecisionV2(formatStr[i:])
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

func parsePrecisionV2(formatStr string) (int, int, error) {
	i := 0
	if formatStr[i] != '.' {
		return i, defaultPrecision, nil
	}
	i++
	var buffer strings.Builder
	for {
		if i >= len(formatStr) {
			return -1, -1, errors.New("could not find end of precision specifier")
		}
		if !isASCIIDigit(rune(formatStr[i])) {
			break
		}
		buffer.WriteByte(formatStr[i])
		i++
	}
	precision, err := strconv.Atoi(buffer.String())
	if err != nil {
		return -1, -1, fmt.Errorf("error while converting precision to integer: %w", err)
	}
	if precision < 0 {
		return -1, -1, fmt.Errorf("negative precision: %d", precision)
	}
	return i, precision, nil
}
