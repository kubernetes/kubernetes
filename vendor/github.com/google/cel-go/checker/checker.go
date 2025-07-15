// Copyright 2018 Google LLC
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

// Package checker defines functions to type-checked a parsed expression
// against a set of identifier and function declarations.
package checker

import (
	"fmt"
	"reflect"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/decls"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

type checker struct {
	*ast.AST
	ast.ExprFactory
	env                *Env
	errors             *typeErrors
	mappings           *mapping
	freeTypeVarCounter int
}

// Check performs type checking, giving a typed AST.
//
// The input is a parsed AST and an env which encapsulates type binding of variables,
// declarations of built-in functions, descriptions of protocol buffers, and a registry for
// errors.
//
// Returns a type-checked AST, which might not be usable if there are errors in the error
// registry.
func Check(parsed *ast.AST, source common.Source, env *Env) (*ast.AST, *common.Errors) {
	errs := common.NewErrors(source)
	typeMap := make(map[int64]*types.Type)
	refMap := make(map[int64]*ast.ReferenceInfo)
	c := checker{
		AST:                ast.NewCheckedAST(parsed, typeMap, refMap),
		ExprFactory:        ast.NewExprFactory(),
		env:                env,
		errors:             &typeErrors{errs: errs},
		mappings:           newMapping(),
		freeTypeVarCounter: 0,
	}
	c.check(c.Expr())

	// Walk over the final type map substituting any type parameters either by their bound value
	// or by DYN.
	for id, t := range c.TypeMap() {
		c.SetType(id, substitute(c.mappings, t, true))
	}
	return c.AST, errs
}

func (c *checker) check(e ast.Expr) {
	if e == nil {
		return
	}
	switch e.Kind() {
	case ast.LiteralKind:
		literal := ref.Val(e.AsLiteral())
		switch literal.Type() {
		case types.BoolType, types.BytesType, types.DoubleType, types.IntType,
			types.NullType, types.StringType, types.UintType:
			c.setType(e, literal.Type().(*types.Type))
		default:
			c.errors.unexpectedASTType(e.ID(), c.location(e), "literal", literal.Type().TypeName())
		}
	case ast.IdentKind:
		c.checkIdent(e)
	case ast.SelectKind:
		c.checkSelect(e)
	case ast.CallKind:
		c.checkCall(e)
	case ast.ListKind:
		c.checkCreateList(e)
	case ast.MapKind:
		c.checkCreateMap(e)
	case ast.StructKind:
		c.checkCreateStruct(e)
	case ast.ComprehensionKind:
		c.checkComprehension(e)
	default:
		c.errors.unexpectedASTType(e.ID(), c.location(e), "unspecified", reflect.TypeOf(e).Name())
	}
}

func (c *checker) checkIdent(e ast.Expr) {
	identName := e.AsIdent()
	// Check to see if the identifier is declared.
	if ident := c.env.LookupIdent(identName); ident != nil {
		c.setType(e, ident.Type())
		c.setReference(e, ast.NewIdentReference(ident.Name(), ident.Value()))
		// Overwrite the identifier with its fully qualified name.
		e.SetKindCase(c.NewIdent(e.ID(), ident.Name()))
		return
	}

	c.setType(e, types.ErrorType)
	c.errors.undeclaredReference(e.ID(), c.location(e), c.env.container.Name(), identName)
}

func (c *checker) checkSelect(e ast.Expr) {
	sel := e.AsSelect()
	// Before traversing down the tree, try to interpret as qualified name.
	qname, found := containers.ToQualifiedName(e)
	if found {
		ident := c.env.LookupIdent(qname)
		if ident != nil {
			// We don't check for a TestOnly expression here since the `found` result is
			// always going to be false for TestOnly expressions.

			// Rewrite the node to be a variable reference to the resolved fully-qualified
			// variable name.
			c.setType(e, ident.Type())
			c.setReference(e, ast.NewIdentReference(ident.Name(), ident.Value()))
			e.SetKindCase(c.NewIdent(e.ID(), ident.Name()))
			return
		}
	}

	resultType := c.checkSelectField(e, sel.Operand(), sel.FieldName(), false)
	if sel.IsTestOnly() {
		resultType = types.BoolType
	}
	c.setType(e, substitute(c.mappings, resultType, false))
}

func (c *checker) checkOptSelect(e ast.Expr) {
	// Collect metadata related to the opt select call packaged by the parser.
	call := e.AsCall()
	if len(call.Args()) != 2 || call.IsMemberFunction() {
		t := ""
		if call.IsMemberFunction() {
			t = " member call with"
		}
		c.errors.notAnOptionalFieldSelectionCall(e.ID(), c.location(e),
			fmt.Sprintf(
				"incorrect signature.%s argument count: %d%s", t, len(call.Args())))
		return
	}

	operand := call.Args()[0]
	field := call.Args()[1]
	fieldName, isString := maybeUnwrapString(field)
	if !isString {
		c.errors.notAnOptionalFieldSelection(field.ID(), c.location(field), field)
		return
	}

	// Perform type-checking using the field selection logic.
	resultType := c.checkSelectField(e, operand, fieldName, true)
	c.setType(e, substitute(c.mappings, resultType, false))
	c.setReference(e, ast.NewFunctionReference("select_optional_field"))
}

func (c *checker) checkSelectField(e, operand ast.Expr, field string, optional bool) *types.Type {
	// Interpret as field selection, first traversing down the operand.
	c.check(operand)
	operandType := substitute(c.mappings, c.getType(operand), false)

	// If the target type is 'optional', unwrap it for the sake of this check.
	targetType, isOpt := maybeUnwrapOptional(operandType)

	// Assume error type by default as most types do not support field selection.
	resultType := types.ErrorType
	switch targetType.Kind() {
	case types.MapKind:
		// Maps yield their value type as the selection result type.
		resultType = targetType.Parameters()[1]
	case types.StructKind:
		// Objects yield their field type declaration as the selection result type, but only if
		// the field is defined.
		messageType := targetType
		if fieldType, found := c.lookupFieldType(e.ID(), messageType.TypeName(), field); found {
			resultType = fieldType
		}
	case types.TypeParamKind:
		// Set the operand type to DYN to prevent assignment to a potentially incorrect type
		// at a later point in type-checking. The isAssignable call will update the type
		// substitutions for the type param under the covers.
		c.isAssignable(types.DynType, targetType)
		// Also, set the result type to DYN.
		resultType = types.DynType
	default:
		// Dynamic / error values are treated as DYN type. Errors are handled this way as well
		// in order to allow forward progress on the check.
		if !isDynOrError(targetType) {
			c.errors.typeDoesNotSupportFieldSelection(e.ID(), c.location(e), targetType)
		}
		resultType = types.DynType
	}

	// If the target type was optional coming in, then the result must be optional going out.
	if isOpt || optional {
		return types.NewOptionalType(resultType)
	}
	return resultType
}

func (c *checker) checkCall(e ast.Expr) {
	// Note: similar logic exists within the `interpreter/planner.go`. If making changes here
	// please consider the impact on planner.go and consolidate implementations or mirror code
	// as appropriate.
	call := e.AsCall()
	fnName := call.FunctionName()
	if fnName == operators.OptSelect {
		c.checkOptSelect(e)
		return
	}

	args := call.Args()
	// Traverse arguments.
	for _, arg := range args {
		c.check(arg)
	}

	// Regular static call with simple name.
	if !call.IsMemberFunction() {
		// Check for the existence of the function.
		fn := c.env.LookupFunction(fnName)
		if fn == nil {
			c.errors.undeclaredReference(e.ID(), c.location(e), c.env.container.Name(), fnName)
			c.setType(e, types.ErrorType)
			return
		}
		// Overwrite the function name with its fully qualified resolved name.
		e.SetKindCase(c.NewCall(e.ID(), fn.Name(), args...))
		// Check to see whether the overload resolves.
		c.resolveOverloadOrError(e, fn, nil, args)
		return
	}

	// If a receiver 'target' is present, it may either be a receiver function, or a namespaced
	// function, but not both. Given a.b.c() either a.b.c is a function or c is a function with
	// target a.b.
	//
	// Check whether the target is a namespaced function name.
	target := call.Target()
	qualifiedPrefix, maybeQualified := containers.ToQualifiedName(target)
	if maybeQualified {
		maybeQualifiedName := qualifiedPrefix + "." + fnName
		fn := c.env.LookupFunction(maybeQualifiedName)
		if fn != nil {
			// The function name is namespaced and so preserving the target operand would
			// be an inaccurate representation of the desired evaluation behavior.
			// Overwrite with fully-qualified resolved function name sans receiver target.
			e.SetKindCase(c.NewCall(e.ID(), fn.Name(), args...))
			c.resolveOverloadOrError(e, fn, nil, args)
			return
		}
	}

	// Regular instance call.
	c.check(target)
	fn := c.env.LookupFunction(fnName)
	// Function found, attempt overload resolution.
	if fn != nil {
		c.resolveOverloadOrError(e, fn, target, args)
		return
	}
	// Function name not declared, record error.
	c.setType(e, types.ErrorType)
	c.errors.undeclaredReference(e.ID(), c.location(e), c.env.container.Name(), fnName)
}

func (c *checker) resolveOverloadOrError(
	e ast.Expr, fn *decls.FunctionDecl, target ast.Expr, args []ast.Expr) {
	// Attempt to resolve the overload.
	resolution := c.resolveOverload(e, fn, target, args)
	// No such overload, error noted in the resolveOverload call, type recorded here.
	if resolution == nil {
		c.setType(e, types.ErrorType)
		return
	}
	// Overload found.
	c.setType(e, resolution.Type)
	c.setReference(e, resolution.Reference)
}

func (c *checker) resolveOverload(
	call ast.Expr, fn *decls.FunctionDecl, target ast.Expr, args []ast.Expr) *overloadResolution {

	var argTypes []*types.Type
	if target != nil {
		argTypes = append(argTypes, c.getType(target))
	}
	for _, arg := range args {
		argTypes = append(argTypes, c.getType(arg))
	}

	var resultType *types.Type
	var checkedRef *ast.ReferenceInfo
	for _, overload := range fn.OverloadDecls() {
		// Determine whether the overload is currently considered.
		if c.env.isOverloadDisabled(overload.ID()) {
			continue
		}

		// Ensure the call style for the overload matches.
		if (target == nil && overload.IsMemberFunction()) ||
			(target != nil && !overload.IsMemberFunction()) {
			// not a compatible call style.
			continue
		}

		// Alternative type-checking behavior when the logical operators are compacted into
		// variadic AST representations.
		if fn.Name() == operators.LogicalAnd || fn.Name() == operators.LogicalOr {
			checkedRef = ast.NewFunctionReference(overload.ID())
			for i, argType := range argTypes {
				if !c.isAssignable(argType, types.BoolType) {
					c.errors.typeMismatch(
						args[i].ID(),
						c.locationByID(args[i].ID()),
						types.BoolType,
						argType)
					resultType = types.ErrorType
				}
			}
			if isError(resultType) {
				return nil
			}
			return newResolution(checkedRef, types.BoolType)
		}

		overloadType := newFunctionType(overload.ResultType(), overload.ArgTypes()...)
		typeParams := overload.TypeParams()
		if len(typeParams) != 0 {
			// Instantiate overload's type with fresh type variables.
			substitutions := newMapping()
			for _, typePar := range typeParams {
				substitutions.add(types.NewTypeParamType(typePar), c.newTypeVar())
			}
			overloadType = substitute(substitutions, overloadType, false)
		}

		candidateArgTypes := overloadType.Parameters()[1:]
		if c.isAssignableList(argTypes, candidateArgTypes) {
			if checkedRef == nil {
				checkedRef = ast.NewFunctionReference(overload.ID())
			} else {
				checkedRef.AddOverload(overload.ID())
			}

			// First matching overload, determines result type.
			fnResultType := substitute(c.mappings, overloadType.Parameters()[0], false)
			if resultType == nil {
				resultType = fnResultType
			} else if !isDyn(resultType) && !fnResultType.IsExactType(resultType) {
				resultType = types.DynType
			}
		}
	}

	if resultType == nil {
		for i, argType := range argTypes {
			argTypes[i] = substitute(c.mappings, argType, true)
		}
		c.errors.noMatchingOverload(call.ID(), c.location(call), fn.Name(), argTypes, target != nil)
		return nil
	}

	return newResolution(checkedRef, resultType)
}

func (c *checker) checkCreateList(e ast.Expr) {
	create := e.AsList()
	var elemsType *types.Type
	optionalIndices := create.OptionalIndices()
	optionals := make(map[int32]bool, len(optionalIndices))
	for _, optInd := range optionalIndices {
		optionals[optInd] = true
	}
	for i, e := range create.Elements() {
		c.check(e)
		elemType := c.getType(e)
		if optionals[int32(i)] {
			var isOptional bool
			elemType, isOptional = maybeUnwrapOptional(elemType)
			if !isOptional && !isDyn(elemType) {
				c.errors.typeMismatch(e.ID(), c.location(e), types.NewOptionalType(elemType), elemType)
			}
		}
		elemsType = c.joinTypes(e, elemsType, elemType)
	}
	if elemsType == nil {
		// If the list is empty, assign free type var to elem type.
		elemsType = c.newTypeVar()
	}
	c.setType(e, types.NewListType(elemsType))
}

func (c *checker) checkCreateMap(e ast.Expr) {
	mapVal := e.AsMap()
	var mapKeyType *types.Type
	var mapValueType *types.Type
	for _, e := range mapVal.Entries() {
		entry := e.AsMapEntry()
		key := entry.Key()
		c.check(key)
		mapKeyType = c.joinTypes(key, mapKeyType, c.getType(key))

		val := entry.Value()
		c.check(val)
		valType := c.getType(val)
		if entry.IsOptional() {
			var isOptional bool
			valType, isOptional = maybeUnwrapOptional(valType)
			if !isOptional && !isDyn(valType) {
				c.errors.typeMismatch(val.ID(), c.location(val), types.NewOptionalType(valType), valType)
			}
		}
		mapValueType = c.joinTypes(val, mapValueType, valType)
	}
	if mapKeyType == nil {
		// If the map is empty, assign free type variables to typeKey and value type.
		mapKeyType = c.newTypeVar()
		mapValueType = c.newTypeVar()
	}
	c.setType(e, types.NewMapType(mapKeyType, mapValueType))
}

func (c *checker) checkCreateStruct(e ast.Expr) {
	msgVal := e.AsStruct()
	// Determine the type of the message.
	resultType := types.ErrorType
	ident := c.env.LookupIdent(msgVal.TypeName())
	if ident == nil {
		c.errors.undeclaredReference(
			e.ID(), c.location(e), c.env.container.Name(), msgVal.TypeName())
		c.setType(e, types.ErrorType)
		return
	}
	// Ensure the type name is fully qualified in the AST.
	typeName := ident.Name()
	if msgVal.TypeName() != typeName {
		e.SetKindCase(c.NewStruct(e.ID(), typeName, msgVal.Fields()))
		msgVal = e.AsStruct()
	}
	c.setReference(e, ast.NewIdentReference(typeName, nil))
	identKind := ident.Type().Kind()
	if identKind != types.ErrorKind {
		if identKind != types.TypeKind {
			c.errors.notAType(e.ID(), c.location(e), ident.Type().DeclaredTypeName())
		} else {
			resultType = ident.Type().Parameters()[0]
			// Backwards compatibility test between well-known types and message types
			// In this context, the type is being instantiated by its protobuf name which
			// is not ideal or recommended, but some users expect this to work.
			if isWellKnownType(resultType) {
				typeName = getWellKnownTypeName(resultType)
			} else if resultType.Kind() == types.StructKind {
				typeName = resultType.DeclaredTypeName()
			} else {
				c.errors.notAMessageType(e.ID(), c.location(e), resultType.DeclaredTypeName())
				resultType = types.ErrorType
			}
		}
	}
	c.setType(e, resultType)

	// Check the field initializers.
	for _, f := range msgVal.Fields() {
		field := f.AsStructField()
		fieldName := field.Name()
		value := field.Value()
		c.check(value)

		fieldType := types.ErrorType
		ft, found := c.lookupFieldType(f.ID(), typeName, fieldName)
		if found {
			fieldType = ft
		}

		valType := c.getType(value)
		if field.IsOptional() {
			var isOptional bool
			valType, isOptional = maybeUnwrapOptional(valType)
			if !isOptional && !isDyn(valType) {
				c.errors.typeMismatch(value.ID(), c.location(value), types.NewOptionalType(valType), valType)
			}
		}
		if !c.isAssignable(fieldType, valType) {
			c.errors.fieldTypeMismatch(f.ID(), c.locationByID(f.ID()), fieldName, fieldType, valType)
		}
	}
}

func (c *checker) checkComprehension(e ast.Expr) {
	comp := e.AsComprehension()
	c.check(comp.IterRange())
	c.check(comp.AccuInit())
	rangeType := substitute(c.mappings, c.getType(comp.IterRange()), false)

	// Create a scope for the comprehension since it has a local accumulation variable.
	// This scope will contain the accumulation variable used to compute the result.
	accuType := c.getType(comp.AccuInit())
	c.env = c.env.enterScope()
	c.env.AddIdents(decls.NewVariable(comp.AccuVar(), accuType))

	var varType, var2Type *types.Type
	switch rangeType.Kind() {
	case types.ListKind:
		// varType represents the list element type for one-variable comprehensions.
		varType = rangeType.Parameters()[0]
		if comp.HasIterVar2() {
			// varType represents the list index (int) for two-variable comprehensions,
			// and var2Type represents the list element type.
			var2Type = varType
			varType = types.IntType
		}
	case types.MapKind:
		// varType represents the map entry key for all comprehension types.
		varType = rangeType.Parameters()[0]
		if comp.HasIterVar2() {
			// var2Type represents the map entry value for two-variable comprehensions.
			var2Type = rangeType.Parameters()[1]
		}
	case types.DynKind, types.ErrorKind, types.TypeParamKind:
		// Set the range type to DYN to prevent assignment to a potentially incorrect type
		// at a later point in type-checking. The isAssignable call will update the type
		// substitutions for the type param under the covers.
		c.isAssignable(types.DynType, rangeType)
		// Set the range iteration variable to type DYN as well.
		varType = types.DynType
		if comp.HasIterVar2() {
			var2Type = types.DynType
		}
	default:
		c.errors.notAComprehensionRange(comp.IterRange().ID(), c.location(comp.IterRange()), rangeType)
		varType = types.ErrorType
		if comp.HasIterVar2() {
			var2Type = types.ErrorType
		}
	}

	// Create a block scope for the loop.
	c.env = c.env.enterScope()
	c.env.AddIdents(decls.NewVariable(comp.IterVar(), varType))
	if comp.HasIterVar2() {
		c.env.AddIdents(decls.NewVariable(comp.IterVar2(), var2Type))
	}
	// Check the variable references in the condition and step.
	c.check(comp.LoopCondition())
	c.assertType(comp.LoopCondition(), types.BoolType)
	c.check(comp.LoopStep())
	c.assertType(comp.LoopStep(), accuType)
	// Exit the loop's block scope before checking the result.
	c.env = c.env.exitScope()
	c.check(comp.Result())
	// Exit the comprehension scope.
	c.env = c.env.exitScope()
	c.setType(e, substitute(c.mappings, c.getType(comp.Result()), false))
}

// Checks compatibility of joined types, and returns the most general common type.
func (c *checker) joinTypes(e ast.Expr, previous, current *types.Type) *types.Type {
	if previous == nil {
		return current
	}
	if c.isAssignable(previous, current) {
		return mostGeneral(previous, current)
	}
	if c.dynAggregateLiteralElementTypesEnabled() {
		return types.DynType
	}
	c.errors.typeMismatch(e.ID(), c.location(e), previous, current)
	return types.ErrorType
}

func (c *checker) dynAggregateLiteralElementTypesEnabled() bool {
	return c.env.aggLitElemType == dynElementType
}

func (c *checker) newTypeVar() *types.Type {
	id := c.freeTypeVarCounter
	c.freeTypeVarCounter++
	return types.NewTypeParamType(fmt.Sprintf("_var%d", id))
}

func (c *checker) isAssignable(t1, t2 *types.Type) bool {
	subs := isAssignable(c.mappings, t1, t2)
	if subs != nil {
		c.mappings = subs
		return true
	}

	return false
}

func (c *checker) isAssignableList(l1, l2 []*types.Type) bool {
	subs := isAssignableList(c.mappings, l1, l2)
	if subs != nil {
		c.mappings = subs
		return true
	}

	return false
}

func maybeUnwrapString(e ast.Expr) (string, bool) {
	switch e.Kind() {
	case ast.LiteralKind:
		literal := e.AsLiteral()
		switch v := literal.(type) {
		case types.String:
			return string(v), true
		}
	}
	return "", false
}

func (c *checker) setType(e ast.Expr, t *types.Type) {
	if old, found := c.TypeMap()[e.ID()]; found && !old.IsExactType(t) {
		c.errors.incompatibleType(e.ID(), c.location(e), e, old, t)
		return
	}
	c.SetType(e.ID(), t)
}

func (c *checker) getType(e ast.Expr) *types.Type {
	return c.TypeMap()[e.ID()]
}

func (c *checker) setReference(e ast.Expr, r *ast.ReferenceInfo) {
	if old, found := c.ReferenceMap()[e.ID()]; found && !old.Equals(r) {
		c.errors.referenceRedefinition(e.ID(), c.location(e), e, old, r)
		return
	}
	c.SetReference(e.ID(), r)
}

func (c *checker) assertType(e ast.Expr, t *types.Type) {
	if !c.isAssignable(t, c.getType(e)) {
		c.errors.typeMismatch(e.ID(), c.location(e), t, c.getType(e))
	}
}

type overloadResolution struct {
	Type      *types.Type
	Reference *ast.ReferenceInfo
}

func newResolution(r *ast.ReferenceInfo, t *types.Type) *overloadResolution {
	return &overloadResolution{
		Reference: r,
		Type:      t,
	}
}

func (c *checker) location(e ast.Expr) common.Location {
	return c.locationByID(e.ID())
}

func (c *checker) locationByID(id int64) common.Location {
	return c.SourceInfo().GetStartLocation(id)
}

func (c *checker) lookupFieldType(exprID int64, structType, fieldName string) (*types.Type, bool) {
	if _, found := c.env.provider.FindStructType(structType); !found {
		// This should not happen, anyway, report an error.
		c.errors.unexpectedFailedResolution(exprID, c.locationByID(exprID), structType)
		return nil, false
	}

	if ft, found := c.env.provider.FindStructFieldType(structType, fieldName); found {
		return ft.Type, found
	}

	c.errors.undefinedField(exprID, c.locationByID(exprID), fieldName)
	return nil, false
}

func isWellKnownType(t *types.Type) bool {
	switch t.Kind() {
	case types.AnyKind, types.TimestampKind, types.DurationKind, types.DynKind, types.NullTypeKind:
		return true
	case types.BoolKind, types.BytesKind, types.DoubleKind, types.IntKind, types.StringKind, types.UintKind:
		return t.IsAssignableType(types.NullType)
	case types.ListKind:
		return t.Parameters()[0] == types.DynType
	case types.MapKind:
		return t.Parameters()[0] == types.StringType && t.Parameters()[1] == types.DynType
	}
	return false
}

func getWellKnownTypeName(t *types.Type) string {
	if name, found := wellKnownTypes[t.Kind()]; found {
		return name
	}
	return ""
}

var (
	wellKnownTypes = map[types.Kind]string{
		types.AnyKind:       "google.protobuf.Any",
		types.BoolKind:      "google.protobuf.BoolValue",
		types.BytesKind:     "google.protobuf.BytesValue",
		types.DoubleKind:    "google.protobuf.DoubleValue",
		types.DurationKind:  "google.protobuf.Duration",
		types.DynKind:       "google.protobuf.Value",
		types.IntKind:       "google.protobuf.Int64Value",
		types.ListKind:      "google.protobuf.ListValue",
		types.NullTypeKind:  "google.protobuf.NullValue",
		types.MapKind:       "google.protobuf.Struct",
		types.StringKind:    "google.protobuf.StringValue",
		types.TimestampKind: "google.protobuf.Timestamp",
		types.UintKind:      "google.protobuf.UInt64Value",
	}
)
