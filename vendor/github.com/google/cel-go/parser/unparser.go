// Copyright 2019 Google LLC
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

package parser

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/types"
)

// Unparse takes an input expression and source position information and generates a human-readable
// expression.
//
// Note, unparsing an AST will often generate the same expression as was originally parsed, but some
// formatting may be lost in translation, notably:
//
// - All quoted literals are doubled quoted.
// - Byte literals are represented as octal escapes (same as Google SQL).
// - Floating point values are converted to the small number of digits needed to represent the value.
// - Spacing around punctuation marks may be lost.
// - Parentheses will only be applied when they affect operator precedence.
//
// This function optionally takes in one or more UnparserOption to alter the unparsing behavior, such as
// performing word wrapping on expressions.
func Unparse(expr ast.Expr, info *ast.SourceInfo, opts ...UnparserOption) (string, error) {
	unparserOpts := &unparserOption{
		wrapOnColumn:         defaultWrapOnColumn,
		wrapAfterColumnLimit: defaultWrapAfterColumnLimit,
		operatorsToWrapOn:    defaultOperatorsToWrapOn,
	}

	var err error
	for _, opt := range opts {
		unparserOpts, err = opt(unparserOpts)
		if err != nil {
			return "", err
		}
	}

	un := &unparser{
		info:    info,
		options: unparserOpts,
	}
	err = un.visit(expr)
	if err != nil {
		return "", err
	}
	return un.str.String(), nil
}

// unparser visits an expression to reconstruct a human-readable string from an AST.
type unparser struct {
	str              strings.Builder
	info             *ast.SourceInfo
	options          *unparserOption
	lastWrappedIndex int
}

func (un *unparser) visit(expr ast.Expr) error {
	if expr == nil {
		return errors.New("unsupported expression")
	}
	visited, err := un.visitMaybeMacroCall(expr)
	if visited || err != nil {
		return err
	}
	switch expr.Kind() {
	case ast.CallKind:
		return un.visitCall(expr)
	case ast.LiteralKind:
		return un.visitConst(expr)
	case ast.IdentKind:
		return un.visitIdent(expr)
	case ast.ListKind:
		return un.visitList(expr)
	case ast.MapKind:
		return un.visitStructMap(expr)
	case ast.SelectKind:
		return un.visitSelect(expr)
	case ast.StructKind:
		return un.visitStructMsg(expr)
	default:
		return fmt.Errorf("unsupported expression: %v", expr)
	}
}

func (un *unparser) visitCall(expr ast.Expr) error {
	c := expr.AsCall()
	fun := c.FunctionName()
	switch fun {
	// ternary operator
	case operators.Conditional:
		return un.visitCallConditional(expr)
	// optional select operator
	case operators.OptSelect:
		return un.visitOptSelect(expr)
	// index operator
	case operators.Index:
		return un.visitCallIndex(expr)
	// optional index operator
	case operators.OptIndex:
		return un.visitCallOptIndex(expr)
	// unary operators
	case operators.LogicalNot, operators.Negate:
		return un.visitCallUnary(expr)
	// binary operators
	case operators.Add,
		operators.Divide,
		operators.Equals,
		operators.Greater,
		operators.GreaterEquals,
		operators.In,
		operators.Less,
		operators.LessEquals,
		operators.LogicalAnd,
		operators.LogicalOr,
		operators.Modulo,
		operators.Multiply,
		operators.NotEquals,
		operators.OldIn,
		operators.Subtract:
		return un.visitCallBinary(expr)
	// standard function calls.
	default:
		return un.visitCallFunc(expr)
	}
}

func (un *unparser) visitCallBinary(expr ast.Expr) error {
	c := expr.AsCall()
	fun := c.FunctionName()
	args := c.Args()
	lhs := args[0]
	// add parens if the current operator is lower precedence than the lhs expr operator.
	lhsParen := isComplexOperatorWithRespectTo(fun, lhs)
	rhs := args[1]
	// add parens if the current operator is lower precedence than the rhs expr operator,
	// or the same precedence and the operator is left recursive.
	rhsParen := isComplexOperatorWithRespectTo(fun, rhs)
	if !rhsParen && isLeftRecursive(fun) {
		rhsParen = isSamePrecedence(fun, rhs)
	}
	err := un.visitMaybeNested(lhs, lhsParen)
	if err != nil {
		return err
	}
	unmangled, found := operators.FindReverseBinaryOperator(fun)
	if !found {
		return fmt.Errorf("cannot unmangle operator: %s", fun)
	}

	un.writeOperatorWithWrapping(fun, unmangled)
	return un.visitMaybeNested(rhs, rhsParen)
}

func (un *unparser) visitCallConditional(expr ast.Expr) error {
	c := expr.AsCall()
	args := c.Args()
	// add parens if operand is a conditional itself.
	nested := isSamePrecedence(operators.Conditional, args[0]) ||
		isComplexOperator(args[0])
	err := un.visitMaybeNested(args[0], nested)
	if err != nil {
		return err
	}
	un.writeOperatorWithWrapping(operators.Conditional, "?")

	// add parens if operand is a conditional itself.
	nested = isSamePrecedence(operators.Conditional, args[1]) ||
		isComplexOperator(args[1])
	err = un.visitMaybeNested(args[1], nested)
	if err != nil {
		return err
	}

	un.str.WriteString(" : ")
	// add parens if operand is a conditional itself.
	nested = isSamePrecedence(operators.Conditional, args[2]) ||
		isComplexOperator(args[2])

	return un.visitMaybeNested(args[2], nested)
}

func (un *unparser) visitCallFunc(expr ast.Expr) error {
	c := expr.AsCall()
	fun := c.FunctionName()
	args := c.Args()
	if c.IsMemberFunction() {
		nested := isBinaryOrTernaryOperator(c.Target())
		err := un.visitMaybeNested(c.Target(), nested)
		if err != nil {
			return err
		}
		un.str.WriteString(".")
	}
	un.str.WriteString(fun)
	un.str.WriteString("(")
	for i, arg := range args {
		err := un.visit(arg)
		if err != nil {
			return err
		}
		if i < len(args)-1 {
			un.str.WriteString(", ")
		}
	}
	un.str.WriteString(")")
	return nil
}

func (un *unparser) visitCallIndex(expr ast.Expr) error {
	return un.visitCallIndexInternal(expr, "[")
}

func (un *unparser) visitCallOptIndex(expr ast.Expr) error {
	return un.visitCallIndexInternal(expr, "[?")
}

func (un *unparser) visitCallIndexInternal(expr ast.Expr, op string) error {
	c := expr.AsCall()
	args := c.Args()
	nested := isBinaryOrTernaryOperator(args[0])
	err := un.visitMaybeNested(args[0], nested)
	if err != nil {
		return err
	}
	un.str.WriteString(op)
	err = un.visit(args[1])
	if err != nil {
		return err
	}
	un.str.WriteString("]")
	return nil
}

func (un *unparser) visitCallUnary(expr ast.Expr) error {
	c := expr.AsCall()
	fun := c.FunctionName()
	args := c.Args()
	unmangled, found := operators.FindReverse(fun)
	if !found {
		return fmt.Errorf("cannot unmangle operator: %s", fun)
	}
	un.str.WriteString(unmangled)
	nested := isComplexOperator(args[0])
	return un.visitMaybeNested(args[0], nested)
}

func (un *unparser) visitConst(expr ast.Expr) error {
	val := expr.AsLiteral()
	switch val := val.(type) {
	case types.Bool:
		un.str.WriteString(strconv.FormatBool(bool(val)))
	case types.Bytes:
		// bytes constants are surrounded with b"<bytes>"
		un.str.WriteString(`b"`)
		un.str.WriteString(bytesToOctets([]byte(val)))
		un.str.WriteString(`"`)
	case types.Double:
		// represent the float using the minimum required digits
		d := strconv.FormatFloat(float64(val), 'g', -1, 64)
		un.str.WriteString(d)
		if !strings.Contains(d, ".") {
			un.str.WriteString(".0")
		}
	case types.Int:
		i := strconv.FormatInt(int64(val), 10)
		un.str.WriteString(i)
	case types.Null:
		un.str.WriteString("null")
	case types.String:
		// strings will be double quoted with quotes escaped.
		un.str.WriteString(strconv.Quote(string(val)))
	case types.Uint:
		// uint literals have a 'u' suffix.
		ui := strconv.FormatUint(uint64(val), 10)
		un.str.WriteString(ui)
		un.str.WriteString("u")
	default:
		return fmt.Errorf("unsupported constant: %v", expr)
	}
	return nil
}

func (un *unparser) visitIdent(expr ast.Expr) error {
	un.str.WriteString(expr.AsIdent())
	return nil
}

func (un *unparser) visitList(expr ast.Expr) error {
	l := expr.AsList()
	elems := l.Elements()
	optIndices := make(map[int]bool, len(elems))
	for _, idx := range l.OptionalIndices() {
		optIndices[int(idx)] = true
	}
	un.str.WriteString("[")
	for i, elem := range elems {
		if optIndices[i] {
			un.str.WriteString("?")
		}
		err := un.visit(elem)
		if err != nil {
			return err
		}
		if i < len(elems)-1 {
			un.str.WriteString(", ")
		}
	}
	un.str.WriteString("]")
	return nil
}

func (un *unparser) visitOptSelect(expr ast.Expr) error {
	c := expr.AsCall()
	args := c.Args()
	operand := args[0]
	field := args[1].AsLiteral().(types.String)
	return un.visitSelectInternal(operand, false, ".?", string(field))
}

func (un *unparser) visitSelect(expr ast.Expr) error {
	sel := expr.AsSelect()
	return un.visitSelectInternal(sel.Operand(), sel.IsTestOnly(), ".", sel.FieldName())
}

func (un *unparser) visitSelectInternal(operand ast.Expr, testOnly bool, op string, field string) error {
	// handle the case when the select expression was generated by the has() macro.
	if testOnly {
		un.str.WriteString("has(")
	}
	nested := !testOnly && isBinaryOrTernaryOperator(operand)
	err := un.visitMaybeNested(operand, nested)
	if err != nil {
		return err
	}
	un.str.WriteString(op)
	un.str.WriteString(field)
	if testOnly {
		un.str.WriteString(")")
	}
	return nil
}

func (un *unparser) visitStructMsg(expr ast.Expr) error {
	m := expr.AsStruct()
	fields := m.Fields()
	un.str.WriteString(m.TypeName())
	un.str.WriteString("{")
	for i, f := range fields {
		field := f.AsStructField()
		f := field.Name()
		if field.IsOptional() {
			un.str.WriteString("?")
		}
		un.str.WriteString(f)
		un.str.WriteString(": ")
		v := field.Value()
		err := un.visit(v)
		if err != nil {
			return err
		}
		if i < len(fields)-1 {
			un.str.WriteString(", ")
		}
	}
	un.str.WriteString("}")
	return nil
}

func (un *unparser) visitStructMap(expr ast.Expr) error {
	m := expr.AsMap()
	entries := m.Entries()
	un.str.WriteString("{")
	for i, e := range entries {
		entry := e.AsMapEntry()
		k := entry.Key()
		if entry.IsOptional() {
			un.str.WriteString("?")
		}
		err := un.visit(k)
		if err != nil {
			return err
		}
		un.str.WriteString(": ")
		v := entry.Value()
		err = un.visit(v)
		if err != nil {
			return err
		}
		if i < len(entries)-1 {
			un.str.WriteString(", ")
		}
	}
	un.str.WriteString("}")
	return nil
}

func (un *unparser) visitMaybeMacroCall(expr ast.Expr) (bool, error) {
	call, found := un.info.GetMacroCall(expr.ID())
	if !found {
		return false, nil
	}
	return true, un.visit(call)
}

func (un *unparser) visitMaybeNested(expr ast.Expr, nested bool) error {
	if nested {
		un.str.WriteString("(")
	}
	err := un.visit(expr)
	if err != nil {
		return err
	}
	if nested {
		un.str.WriteString(")")
	}
	return nil
}

// isLeftRecursive indicates whether the parser resolves the call in a left-recursive manner as
// this can have an effect of how parentheses affect the order of operations in the AST.
func isLeftRecursive(op string) bool {
	return op != operators.LogicalAnd && op != operators.LogicalOr
}

// isSamePrecedence indicates whether the precedence of the input operator is the same as the
// precedence of the (possible) operation represented in the input Expr.
//
// If the expr is not a Call, the result is false.
func isSamePrecedence(op string, expr ast.Expr) bool {
	if expr.Kind() != ast.CallKind {
		return false
	}
	c := expr.AsCall()
	other := c.FunctionName()
	return operators.Precedence(op) == operators.Precedence(other)
}

// isLowerPrecedence indicates whether the precedence of the input operator is lower precedence
// than the (possible) operation represented in the input Expr.
//
// If the expr is not a Call, the result is false.
func isLowerPrecedence(op string, expr ast.Expr) bool {
	c := expr.AsCall()
	other := c.FunctionName()
	return operators.Precedence(op) < operators.Precedence(other)
}

// Indicates whether the expr is a complex operator, i.e., a call expression
// with 2 or more arguments.
func isComplexOperator(expr ast.Expr) bool {
	if expr.Kind() == ast.CallKind && len(expr.AsCall().Args()) >= 2 {
		return true
	}
	return false
}

// Indicates whether it is a complex operation compared to another.
// expr is *not* considered complex if it is not a call expression or has
// less than two arguments, or if it has a higher precedence than op.
func isComplexOperatorWithRespectTo(op string, expr ast.Expr) bool {
	if expr.Kind() != ast.CallKind || len(expr.AsCall().Args()) < 2 {
		return false
	}
	return isLowerPrecedence(op, expr)
}

// Indicate whether this is a binary or ternary operator.
func isBinaryOrTernaryOperator(expr ast.Expr) bool {
	if expr.Kind() != ast.CallKind || len(expr.AsCall().Args()) < 2 {
		return false
	}
	_, isBinaryOp := operators.FindReverseBinaryOperator(expr.AsCall().FunctionName())
	return isBinaryOp || isSamePrecedence(operators.Conditional, expr)
}

// bytesToOctets converts byte sequences to a string using a three digit octal encoded value
// per byte.
func bytesToOctets(byteVal []byte) string {
	var b strings.Builder
	for _, c := range byteVal {
		fmt.Fprintf(&b, "\\%03o", c)
	}
	return b.String()
}

// writeOperatorWithWrapping outputs the operator and inserts a newline for operators configured
// in the unparser options.
func (un *unparser) writeOperatorWithWrapping(fun string, unmangled string) bool {
	_, wrapOperatorExists := un.options.operatorsToWrapOn[fun]
	lineLength := un.str.Len() - un.lastWrappedIndex + len(fun)

	if wrapOperatorExists && lineLength >= un.options.wrapOnColumn {
		un.lastWrappedIndex = un.str.Len()
		// wrapAfterColumnLimit flag dictates whether the newline is placed
		// before or after the operator
		if un.options.wrapAfterColumnLimit {
			// Input: a && b
			// Output: a &&\nb
			un.str.WriteString(" ")
			un.str.WriteString(unmangled)
			un.str.WriteString("\n")
		} else {
			// Input: a && b
			// Output: a\n&& b
			un.str.WriteString("\n")
			un.str.WriteString(unmangled)
			un.str.WriteString(" ")
		}
		return true
	}
	un.str.WriteString(" ")
	un.str.WriteString(unmangled)
	un.str.WriteString(" ")
	return false
}

// Defined defaults for the unparser options
var (
	defaultWrapOnColumn         = 80
	defaultWrapAfterColumnLimit = true
	defaultOperatorsToWrapOn    = map[string]bool{
		operators.LogicalAnd: true,
		operators.LogicalOr:  true,
	}
)

// UnparserOption is a functional option for configuring the output formatting
// of the Unparse function.
type UnparserOption func(*unparserOption) (*unparserOption, error)

// Internal representation of the UnparserOption type
type unparserOption struct {
	wrapOnColumn         int
	operatorsToWrapOn    map[string]bool
	wrapAfterColumnLimit bool
}

// WrapOnColumn wraps the output expression when its string length exceeds a specified limit
// for operators set by WrapOnOperators function or by default, "&&" and "||" will be wrapped.
//
// Example usage:
//
//	Unparse(expr, sourceInfo, WrapOnColumn(40), WrapOnOperators(Operators.LogicalAnd))
//
// This will insert a newline immediately after the logical AND operator for the below example input:
//
// Input:
// 'my-principal-group' in request.auth.claims && request.auth.claims.iat > now - duration('5m')
//
// Output:
// 'my-principal-group' in request.auth.claims &&
// request.auth.claims.iat > now - duration('5m')
func WrapOnColumn(col int) UnparserOption {
	return func(opt *unparserOption) (*unparserOption, error) {
		if col < 1 {
			return nil, fmt.Errorf("Invalid unparser option. Wrap column value must be greater than or equal to 1. Got %v instead", col)
		}
		opt.wrapOnColumn = col
		return opt, nil
	}
}

// WrapOnOperators specifies which operators to perform word wrapping on an output expression when its string length
// exceeds the column limit set by WrapOnColumn function.
//
// Word wrapping is supported on non-unary symbolic operators. Refer to operators.go for the full list
//
// This will replace any previously supplied operators instead of merging them.
func WrapOnOperators(symbols ...string) UnparserOption {
	return func(opt *unparserOption) (*unparserOption, error) {
		opt.operatorsToWrapOn = make(map[string]bool)
		for _, symbol := range symbols {
			_, found := operators.FindReverse(symbol)
			if !found {
				return nil, fmt.Errorf("Invalid unparser option. Unsupported operator: %s", symbol)
			}
			arity := operators.Arity(symbol)
			if arity < 2 {
				return nil, fmt.Errorf("Invalid unparser option. Unary operators are unsupported: %s", symbol)
			}

			opt.operatorsToWrapOn[symbol] = true
		}

		return opt, nil
	}
}

// WrapAfterColumnLimit dictates whether to insert a newline before or after the specified operator
// when word wrapping is performed.
//
// Example usage:
//
//	Unparse(expr, sourceInfo, WrapOnColumn(40), WrapOnOperators(Operators.LogicalAnd), WrapAfterColumnLimit(false))
//
// This will insert a newline immediately before the logical AND operator for the below example input, ensuring
// that the length of a line never exceeds the specified column limit:
//
// Input:
// 'my-principal-group' in request.auth.claims && request.auth.claims.iat > now - duration('5m')
//
// Output:
// 'my-principal-group' in request.auth.claims
// && request.auth.claims.iat > now - duration('5m')
func WrapAfterColumnLimit(wrapAfter bool) UnparserOption {
	return func(opt *unparserOption) (*unparserOption, error) {
		opt.wrapAfterColumnLimit = wrapAfter
		return opt, nil
	}
}
