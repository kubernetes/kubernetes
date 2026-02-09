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
	"unicode"

	"golang.org/x/text/language"
	"golang.org/x/text/message"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

type clauseImpl func(ref.Val, string) (string, error)

func clauseForType(argType ref.Type) (clauseImpl, error) {
	switch argType {
	case types.IntType, types.UintType:
		return formatDecimal, nil
	case types.StringType, types.BytesType, types.BoolType, types.NullType, types.TypeType:
		return FormatString, nil
	case types.TimestampType, types.DurationType:
		// special case to ensure timestamps/durations get printed as CEL literals
		return func(arg ref.Val, locale string) (string, error) {
			argStrVal := arg.ConvertToType(types.StringType)
			argStr := argStrVal.Value().(string)
			if arg.Type() == types.TimestampType {
				return fmt.Sprintf("timestamp(%q)", argStr), nil
			}
			if arg.Type() == types.DurationType {
				return fmt.Sprintf("duration(%q)", argStr), nil
			}
			return "", fmt.Errorf("cannot convert argument of type %s to timestamp/duration", arg.Type().TypeName())
		}, nil
	case types.ListType:
		return formatList, nil
	case types.MapType:
		return formatMap, nil
	case types.DoubleType:
		// avoid formatFixed so we can output a period as the decimal separator in order
		// to always be a valid CEL literal
		return func(arg ref.Val, locale string) (string, error) {
			argDouble, ok := arg.Value().(float64)
			if !ok {
				return "", fmt.Errorf("couldn't convert %s to float64", arg.Type().TypeName())
			}
			fmtStr := fmt.Sprintf("%%.%df", defaultPrecision)
			return fmt.Sprintf(fmtStr, argDouble), nil
		}, nil
	case types.TypeType:
		return func(arg ref.Val, locale string) (string, error) {
			return fmt.Sprintf("type(%s)", arg.Value().(string)), nil
		}, nil
	default:
		return nil, fmt.Errorf("no formatting function for %s", argType.TypeName())
	}
}

func formatList(arg ref.Val, locale string) (string, error) {
	argList := arg.(traits.Lister)
	argIterator := argList.Iterator()
	var listStrBuilder strings.Builder
	_, err := listStrBuilder.WriteRune('[')
	if err != nil {
		return "", fmt.Errorf("error writing to list string: %w", err)
	}
	for argIterator.HasNext() == types.True {
		member := argIterator.Next()
		memberFormat, err := clauseForType(member.Type())
		if err != nil {
			return "", err
		}
		unquotedStr, err := memberFormat(member, locale)
		if err != nil {
			return "", err
		}
		str := quoteForCEL(member, unquotedStr)
		_, err = listStrBuilder.WriteString(str)
		if err != nil {
			return "", fmt.Errorf("error writing to list string: %w", err)
		}
		if argIterator.HasNext() == types.True {
			_, err = listStrBuilder.WriteString(", ")
			if err != nil {
				return "", fmt.Errorf("error writing to list string: %w", err)
			}
		}
	}
	_, err = listStrBuilder.WriteRune(']')
	if err != nil {
		return "", fmt.Errorf("error writing to list string: %w", err)
	}
	return listStrBuilder.String(), nil
}

func formatMap(arg ref.Val, locale string) (string, error) {
	argMap := arg.(traits.Mapper)
	argIterator := argMap.Iterator()
	type mapPair struct {
		key   string
		value string
	}
	argPairs := make([]mapPair, argMap.Size().Value().(int64))
	i := 0
	for argIterator.HasNext() == types.True {
		key := argIterator.Next()
		var keyFormat clauseImpl
		switch key.Type() {
		case types.StringType, types.BoolType:
			keyFormat = FormatString
		case types.IntType, types.UintType:
			keyFormat = formatDecimal
		default:
			return "", fmt.Errorf("no formatting function for map key of type %s", key.Type().TypeName())
		}
		unquotedKeyStr, err := keyFormat(key, locale)
		if err != nil {
			return "", err
		}
		keyStr := quoteForCEL(key, unquotedKeyStr)
		value, found := argMap.Find(key)
		if !found {
			return "", fmt.Errorf("could not find key: %q", key)
		}
		valueFormat, err := clauseForType(value.Type())
		if err != nil {
			return "", err
		}
		unquotedValueStr, err := valueFormat(value, locale)
		if err != nil {
			return "", err
		}
		valueStr := quoteForCEL(value, unquotedValueStr)
		argPairs[i] = mapPair{keyStr, valueStr}
		i++
	}
	sort.SliceStable(argPairs, func(x, y int) bool {
		return argPairs[x].key < argPairs[y].key
	})
	var mapStrBuilder strings.Builder
	_, err := mapStrBuilder.WriteRune('{')
	if err != nil {
		return "", fmt.Errorf("error writing to map string: %w", err)
	}
	for i, entry := range argPairs {
		_, err = mapStrBuilder.WriteString(fmt.Sprintf("%s:%s", entry.key, entry.value))
		if err != nil {
			return "", fmt.Errorf("error writing to map string: %w", err)
		}
		if i < len(argPairs)-1 {
			_, err = mapStrBuilder.WriteString(", ")
			if err != nil {
				return "", fmt.Errorf("error writing to map string: %w", err)
			}
		}
	}
	_, err = mapStrBuilder.WriteRune('}')
	if err != nil {
		return "", fmt.Errorf("error writing to map string: %w", err)
	}
	return mapStrBuilder.String(), nil
}

// quoteForCEL takes a formatted, unquoted value and quotes it in a manner suitable
// for embedding directly in CEL.
func quoteForCEL(refVal ref.Val, unquotedValue string) string {
	switch refVal.Type() {
	case types.StringType:
		return fmt.Sprintf("%q", unquotedValue)
	case types.BytesType:
		return fmt.Sprintf("b%q", unquotedValue)
	case types.DoubleType:
		// special case to handle infinity/NaN
		num := refVal.Value().(float64)
		if math.IsInf(num, 1) || math.IsInf(num, -1) || math.IsNaN(num) {
			return fmt.Sprintf("%q", unquotedValue)
		}
		return unquotedValue
	default:
		return unquotedValue
	}
}

// FormatString returns the string representation of a CEL value.
//
// It is used to implement the %s specifier in the (string).format() extension function.
func FormatString(arg ref.Val, locale string) (string, error) {
	switch arg.Type() {
	case types.ListType:
		return formatList(arg, locale)
	case types.MapType:
		return formatMap(arg, locale)
	case types.IntType, types.UintType, types.DoubleType,
		types.BoolType, types.StringType, types.TimestampType, types.BytesType, types.DurationType, types.TypeType:
		argStrVal := arg.ConvertToType(types.StringType)
		argStr, ok := argStrVal.Value().(string)
		if !ok {
			return "", fmt.Errorf("could not convert argument %q to string", argStrVal)
		}
		return argStr, nil
	case types.NullType:
		return "null", nil
	default:
		return "", stringFormatError(runtimeID, arg.Type().TypeName())
	}
}

func formatDecimal(arg ref.Val, locale string) (string, error) {
	switch arg.Type() {
	case types.IntType:
		argInt, ok := arg.ConvertToType(types.IntType).Value().(int64)
		if !ok {
			return "", fmt.Errorf("could not convert \"%s\" to int64", arg.Value())
		}
		return fmt.Sprintf("%d", argInt), nil
	case types.UintType:
		argInt, ok := arg.ConvertToType(types.UintType).Value().(uint64)
		if !ok {
			return "", fmt.Errorf("could not convert \"%s\" to uint64", arg.Value())
		}
		return fmt.Sprintf("%d", argInt), nil
	default:
		return "", decimalFormatError(runtimeID, arg.Type().TypeName())
	}
}

func matchLanguage(locale string) (language.Tag, error) {
	matcher, err := makeMatcher(locale)
	if err != nil {
		return language.Und, err
	}
	tag, _ := language.MatchStrings(matcher, locale)
	return tag, nil
}

func makeMatcher(locale string) (language.Matcher, error) {
	tags := make([]language.Tag, 0)
	tag, err := language.Parse(locale)
	if err != nil {
		return nil, err
	}
	tags = append(tags, tag)
	return language.NewMatcher(tags), nil
}

type stringFormatter struct{}

// String implements formatStringInterpolator.String.
func (c *stringFormatter) String(arg ref.Val, locale string) (string, error) {
	return FormatString(arg, locale)
}

// Decimal implements formatStringInterpolator.Decimal.
func (c *stringFormatter) Decimal(arg ref.Val, locale string) (string, error) {
	return formatDecimal(arg, locale)
}

// Fixed implements formatStringInterpolator.Fixed.
func (c *stringFormatter) Fixed(precision *int) func(ref.Val, string) (string, error) {
	if precision == nil {
		precision = new(int)
		*precision = defaultPrecision
	}
	return func(arg ref.Val, locale string) (string, error) {
		strException := false
		if arg.Type() == types.StringType {
			argStr := arg.Value().(string)
			if argStr == "NaN" || argStr == "Infinity" || argStr == "-Infinity" {
				strException = true
			}
		}
		if arg.Type() != types.DoubleType && !strException {
			return "", fixedPointFormatError(runtimeID, arg.Type().TypeName())
		}
		argFloatVal := arg.ConvertToType(types.DoubleType)
		argFloat, ok := argFloatVal.Value().(float64)
		if !ok {
			return "", fmt.Errorf("could not convert \"%s\" to float64", argFloatVal.Value())
		}
		fmtStr := fmt.Sprintf("%%.%df", *precision)

		matchedLocale, err := matchLanguage(locale)
		if err != nil {
			return "", fmt.Errorf("error matching locale: %w", err)
		}
		return message.NewPrinter(matchedLocale).Sprintf(fmtStr, argFloat), nil
	}
}

// Scientific implements formatStringInterpolator.Scientific.
func (c *stringFormatter) Scientific(precision *int) func(ref.Val, string) (string, error) {
	if precision == nil {
		precision = new(int)
		*precision = defaultPrecision
	}
	return func(arg ref.Val, locale string) (string, error) {
		strException := false
		if arg.Type() == types.StringType {
			argStr := arg.Value().(string)
			if argStr == "NaN" || argStr == "Infinity" || argStr == "-Infinity" {
				strException = true
			}
		}
		if arg.Type() != types.DoubleType && !strException {
			return "", scientificFormatError(runtimeID, arg.Type().TypeName())
		}
		argFloatVal := arg.ConvertToType(types.DoubleType)
		argFloat, ok := argFloatVal.Value().(float64)
		if !ok {
			return "", fmt.Errorf("could not convert \"%v\" to float64", argFloatVal.Value())
		}
		matchedLocale, err := matchLanguage(locale)
		if err != nil {
			return "", fmt.Errorf("error matching locale: %w", err)
		}
		fmtStr := fmt.Sprintf("%%%de", *precision)
		return message.NewPrinter(matchedLocale).Sprintf(fmtStr, argFloat), nil
	}
}

// Binary implements formatStringInterpolator.Binary.
func (c *stringFormatter) Binary(arg ref.Val, locale string) (string, error) {
	switch arg.Type() {
	case types.IntType:
		argInt := arg.Value().(int64)
		// locale is intentionally unused as integers formatted as binary
		// strings are locale-independent
		return fmt.Sprintf("%b", argInt), nil
	case types.UintType:
		argInt := arg.Value().(uint64)
		return fmt.Sprintf("%b", argInt), nil
	case types.BoolType:
		argBool := arg.Value().(bool)
		if argBool {
			return "1", nil
		}
		return "0", nil
	default:
		return "", binaryFormatError(runtimeID, arg.Type().TypeName())
	}
}

// Hex implements formatStringInterpolator.Hex.
func (c *stringFormatter) Hex(useUpper bool) func(ref.Val, string) (string, error) {
	return func(arg ref.Val, locale string) (string, error) {
		fmtStr := "%x"
		if useUpper {
			fmtStr = "%X"
		}
		switch arg.Type() {
		case types.StringType, types.BytesType:
			if arg.Type() == types.BytesType {
				return fmt.Sprintf(fmtStr, arg.Value().([]byte)), nil
			}
			return fmt.Sprintf(fmtStr, arg.Value().(string)), nil
		case types.IntType:
			argInt, ok := arg.Value().(int64)
			if !ok {
				return "", fmt.Errorf("could not convert \"%s\" to int64", arg.Value())
			}
			return fmt.Sprintf(fmtStr, argInt), nil
		case types.UintType:
			argInt, ok := arg.Value().(uint64)
			if !ok {
				return "", fmt.Errorf("could not convert \"%s\" to uint64", arg.Value())
			}
			return fmt.Sprintf(fmtStr, argInt), nil
		default:
			return "", hexFormatError(runtimeID, arg.Type().TypeName())
		}
	}
}

// Octal implements formatStringInterpolator.Octal.
func (c *stringFormatter) Octal(arg ref.Val, locale string) (string, error) {
	switch arg.Type() {
	case types.IntType:
		argInt := arg.Value().(int64)
		return fmt.Sprintf("%o", argInt), nil
	case types.UintType:
		argInt := arg.Value().(uint64)
		return fmt.Sprintf("%o", argInt), nil
	default:
		return "", octalFormatError(runtimeID, arg.Type().TypeName())
	}
}

// stringFormatValidator implements the cel.ASTValidator interface allowing for static validation
// of string.format calls.
type stringFormatValidator struct{}

// Name returns the name of the validator.
func (stringFormatValidator) Name() string {
	return "cel.validator.string_format"
}

// Configure implements the ASTValidatorConfigurer interface and augments the list of functions to skip
// during homogeneous aggregate literal type-checks.
func (stringFormatValidator) Configure(config cel.MutableValidatorConfig) error {
	functions := config.GetOrDefault(cel.HomogeneousAggregateLiteralExemptFunctions, []string{}).([]string)
	functions = append(functions, "format")
	return config.Set(cel.HomogeneousAggregateLiteralExemptFunctions, functions)
}

// Validate parses all literal format strings and type checks the format clause against the argument
// at the corresponding ordinal within the list literal argument to the function, if one is specified.
func (stringFormatValidator) Validate(env *cel.Env, _ cel.ValidatorConfig, a *ast.AST, iss *cel.Issues) {
	root := ast.NavigateAST(a)
	formatCallExprs := ast.MatchDescendants(root, matchConstantFormatStringWithListLiteralArgs(a))
	for _, e := range formatCallExprs {
		call := e.AsCall()
		formatStr := call.Target().AsLiteral().Value().(string)
		args := call.Args()[0].AsList().Elements()
		formatCheck := &stringFormatChecker{
			args: args,
			ast:  a,
		}
		// use a placeholder locale, since locale doesn't affect syntax
		_, err := parseFormatString(formatStr, formatCheck, formatCheck, "en_US")
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

// getErrorExprID determines which list literal argument triggered a type-disagreement for the
// purposes of more accurate error message reports.
func getErrorExprID(id int64, err error) int64 {
	fmtErr, ok := err.(formatError)
	if ok {
		return fmtErr.id
	}
	wrapped := errors.Unwrap(err)
	if wrapped != nil {
		return getErrorExprID(id, wrapped)
	}
	return id
}

// matchConstantFormatStringWithListLiteralArgs matches all valid expression nodes for string
// format checking.
func matchConstantFormatStringWithListLiteralArgs(a *ast.AST) ast.ExprMatcher {
	return func(e ast.NavigableExpr) bool {
		if e.Kind() != ast.CallKind {
			return false
		}
		call := e.AsCall()
		if !call.IsMemberFunction() || call.FunctionName() != "format" {
			return false
		}
		overloadIDs := a.GetOverloadIDs(e.ID())
		if len(overloadIDs) != 0 {
			found := false
			for _, overload := range overloadIDs {
				if overload == overloads.ExtFormatString {
					found = true
					break
				}
			}
			if !found {
				return false
			}
		}
		formatString := call.Target()
		if formatString.Kind() != ast.LiteralKind || formatString.AsLiteral().Type() != cel.StringType {
			return false
		}
		args := call.Args()
		if len(args) != 1 {
			return false
		}
		formatArgs := args[0]
		return formatArgs.Kind() == ast.ListKind
	}
}

// stringFormatChecker implements the formatStringInterpolater interface
type stringFormatChecker struct {
	args          []ast.Expr
	argsRequested int
	currArgIndex  int64
	ast           *ast.AST
}

// String implements formatStringInterpolator.String.
func (c *stringFormatChecker) String(arg ref.Val, locale string) (string, error) {
	formatArg := c.args[c.currArgIndex]
	valid, badID := c.verifyString(formatArg)
	if !valid {
		return "", stringFormatError(badID, c.typeOf(badID).TypeName())
	}
	return "", nil
}

// Decimal implements formatStringInterpolator.Decimal.
func (c *stringFormatChecker) Decimal(arg ref.Val, locale string) (string, error) {
	id := c.args[c.currArgIndex].ID()
	valid := c.verifyTypeOneOf(id, types.IntType, types.UintType)
	if !valid {
		return "", decimalFormatError(id, c.typeOf(id).TypeName())
	}
	return "", nil
}

// Fixed implements formatStringInterpolator.Fixed.
func (c *stringFormatChecker) Fixed(precision *int) func(ref.Val, string) (string, error) {
	return func(arg ref.Val, locale string) (string, error) {
		id := c.args[c.currArgIndex].ID()
		// we allow StringType since "NaN", "Infinity", and "-Infinity" are also valid values
		valid := c.verifyTypeOneOf(id, types.DoubleType, types.StringType)
		if !valid {
			return "", fixedPointFormatError(id, c.typeOf(id).TypeName())
		}
		return "", nil
	}
}

// Scientific implements formatStringInterpolator.Scientific.
func (c *stringFormatChecker) Scientific(precision *int) func(ref.Val, string) (string, error) {
	return func(arg ref.Val, locale string) (string, error) {
		id := c.args[c.currArgIndex].ID()
		valid := c.verifyTypeOneOf(id, types.DoubleType, types.StringType)
		if !valid {
			return "", scientificFormatError(id, c.typeOf(id).TypeName())
		}
		return "", nil
	}
}

// Binary implements formatStringInterpolator.Binary.
func (c *stringFormatChecker) Binary(arg ref.Val, locale string) (string, error) {
	id := c.args[c.currArgIndex].ID()
	valid := c.verifyTypeOneOf(id, types.IntType, types.UintType, types.BoolType)
	if !valid {
		return "", binaryFormatError(id, c.typeOf(id).TypeName())
	}
	return "", nil
}

// Hex implements formatStringInterpolator.Hex.
func (c *stringFormatChecker) Hex(useUpper bool) func(ref.Val, string) (string, error) {
	return func(arg ref.Val, locale string) (string, error) {
		id := c.args[c.currArgIndex].ID()
		valid := c.verifyTypeOneOf(id, types.IntType, types.UintType, types.StringType, types.BytesType)
		if !valid {
			return "", hexFormatError(id, c.typeOf(id).TypeName())
		}
		return "", nil
	}
}

// Octal implements formatStringInterpolator.Octal.
func (c *stringFormatChecker) Octal(arg ref.Val, locale string) (string, error) {
	id := c.args[c.currArgIndex].ID()
	valid := c.verifyTypeOneOf(id, types.IntType, types.UintType)
	if !valid {
		return "", octalFormatError(id, c.typeOf(id).TypeName())
	}
	return "", nil
}

// Arg implements formatListArgs.Arg.
func (c *stringFormatChecker) Arg(index int64) (ref.Val, error) {
	c.argsRequested++
	c.currArgIndex = index
	// return a dummy value - this is immediately passed to back to us
	// through one of the FormatCallback functions, so anything will do
	return types.Int(0), nil
}

// Size implements formatListArgs.Size.
func (c *stringFormatChecker) Size() int64 {
	return int64(len(c.args))
}

func (c *stringFormatChecker) typeOf(id int64) *cel.Type {
	return c.ast.GetType(id)
}

func (c *stringFormatChecker) verifyTypeOneOf(id int64, validTypes ...*cel.Type) bool {
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

func (c *stringFormatChecker) verifyString(sub ast.Expr) (bool, int64) {
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

func binaryFormatError(id int64, badType string) error {
	return newFormatError(id, "only integers and bools can be formatted as binary, was given %s", badType)
}

func decimalFormatError(id int64, badType string) error {
	return newFormatError(id, "decimal clause can only be used on integers, was given %s", badType)
}

func fixedPointFormatError(id int64, badType string) error {
	return newFormatError(id, "fixed-point clause can only be used on doubles, was given %s", badType)
}

func hexFormatError(id int64, badType string) error {
	return newFormatError(id, "only integers, byte buffers, and strings can be formatted as hex, was given %s", badType)
}

func octalFormatError(id int64, badType string) error {
	return newFormatError(id, "octal clause can only be used on integers, was given %s", badType)
}

func scientificFormatError(id int64, badType string) error {
	return newFormatError(id, "scientific clause can only be used on doubles, was given %s", badType)
}

func stringFormatError(id int64, badType string) error {
	return newFormatError(id, "string clause can only be used on strings, bools, bytes, ints, doubles, maps, lists, types, durations, and timestamps, was given %s", badType)
}

type formatError struct {
	id  int64
	msg string
}

func newFormatError(id int64, msg string, args ...any) error {
	return formatError{
		id:  id,
		msg: fmt.Sprintf(msg, args...),
	}
}

// Error implements error.
func (e formatError) Error() string {
	return e.msg
}

// Is implements errors.Is.
func (e formatError) Is(target error) bool {
	return e.msg == target.Error()
}

// stringArgList implements the formatListArgs interface.
type stringArgList struct {
	args traits.Lister
}

// Arg implements formatListArgs.Arg.
func (c *stringArgList) Arg(index int64) (ref.Val, error) {
	if index >= c.args.Size().Value().(int64) {
		return nil, fmt.Errorf("index %d out of range", index)
	}
	return c.args.Get(types.Int(index)), nil
}

// Size implements formatListArgs.Size.
func (c *stringArgList) Size() int64 {
	return c.args.Size().Value().(int64)
}

// formatStringInterpolator is an interface that allows user-defined behavior
// for formatting clause implementations, as well as argument retrieval.
// Each function is expected to support the appropriate types as laid out in
// the string.format documentation, and to return an error if given an inappropriate type.
type formatStringInterpolator interface {
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

// formatListArgs is an interface that allows user-defined list-like datatypes to be used
// for formatting clause implementations.
type formatListArgs interface {
	// Arg returns the ref.Val at the given index, or an error if one occurred.
	Arg(int64) (ref.Val, error)

	// Size returns the length of the argument list.
	Size() int64
}

// parseFormatString formats a string according to the string.format syntax, taking the clause implementations
// from the provided FormatCallback and the args from the given FormatList.
func parseFormatString(formatStr string, callback formatStringInterpolator, list formatListArgs, locale string) (string, error) {
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
func parseAndFormatClause(formatStr string, val ref.Val, callback formatStringInterpolator, list formatListArgs, locale string) (int, string, error) {
	i := 1
	read, formatter, err := parseFormattingClause(formatStr[i:], callback)
	i += read
	if err != nil {
		return -1, "", newParseFormatError("could not parse formatting clause", err)
	}

	valStr, err := formatter(val, locale)
	if err != nil {
		return -1, "", newParseFormatError("error during formatting", err)
	}
	return i, valStr, nil
}

func parseFormattingClause(formatStr string, callback formatStringInterpolator) (int, clauseImpl, error) {
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

type parseFormatError struct {
	msg     string
	wrapped error
}

func newParseFormatError(msg string, wrapped error) error {
	return parseFormatError{msg: msg, wrapped: wrapped}
}

// Error implements error.
func (e parseFormatError) Error() string {
	return fmt.Sprintf("%s: %s", e.msg, e.wrapped.Error())
}

// Is implements errors.Is.
func (e parseFormatError) Is(target error) bool {
	return e.Error() == target.Error()
}

// Is implements errors.Unwrap.
func (e parseFormatError) Unwrap() error {
	return e.wrapped
}

const (
	runtimeID = int64(-1)
)
