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

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/decls"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/types"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

type checker struct {
	env                *Env
	errors             *typeErrors
	mappings           *mapping
	freeTypeVarCounter int
	sourceInfo         *exprpb.SourceInfo
	types              map[int64]*types.Type
	references         map[int64]*ast.ReferenceInfo
}

// Check performs type checking, giving a typed AST.
// The input is a ParsedExpr proto and an env which encapsulates
// type binding of variables, declarations of built-in functions,
// descriptions of protocol buffers, and a registry for errors.
// Returns a CheckedExpr proto, which might not be usable if
// there are errors in the error registry.
func Check(parsedExpr *exprpb.ParsedExpr, source common.Source, env *Env) (*ast.CheckedAST, *common.Errors) {
	errs := common.NewErrors(source)
	c := checker{
		env:                env,
		errors:             &typeErrors{errs: errs},
		mappings:           newMapping(),
		freeTypeVarCounter: 0,
		sourceInfo:         parsedExpr.GetSourceInfo(),
		types:              make(map[int64]*types.Type),
		references:         make(map[int64]*ast.ReferenceInfo),
	}
	c.check(parsedExpr.GetExpr())

	// Walk over the final type map substituting any type parameters either by their bound value or
	// by DYN.
	m := make(map[int64]*types.Type)
	for id, t := range c.types {
		m[id] = substitute(c.mappings, t, true)
	}

	return &ast.CheckedAST{
		Expr:         parsedExpr.GetExpr(),
		SourceInfo:   parsedExpr.GetSourceInfo(),
		TypeMap:      m,
		ReferenceMap: c.references,
	}, errs
}

func (c *checker) check(e *exprpb.Expr) {
	if e == nil {
		return
	}
	switch e.GetExprKind().(type) {
	case *exprpb.Expr_ConstExpr:
		literal := e.GetConstExpr()
		switch literal.GetConstantKind().(type) {
		case *exprpb.Constant_BoolValue:
			c.checkBoolLiteral(e)
		case *exprpb.Constant_BytesValue:
			c.checkBytesLiteral(e)
		case *exprpb.Constant_DoubleValue:
			c.checkDoubleLiteral(e)
		case *exprpb.Constant_Int64Value:
			c.checkInt64Literal(e)
		case *exprpb.Constant_NullValue:
			c.checkNullLiteral(e)
		case *exprpb.Constant_StringValue:
			c.checkStringLiteral(e)
		case *exprpb.Constant_Uint64Value:
			c.checkUint64Literal(e)
		}
	case *exprpb.Expr_IdentExpr:
		c.checkIdent(e)
	case *exprpb.Expr_SelectExpr:
		c.checkSelect(e)
	case *exprpb.Expr_CallExpr:
		c.checkCall(e)
	case *exprpb.Expr_ListExpr:
		c.checkCreateList(e)
	case *exprpb.Expr_StructExpr:
		c.checkCreateStruct(e)
	case *exprpb.Expr_ComprehensionExpr:
		c.checkComprehension(e)
	default:
		c.errors.unexpectedASTType(e.GetId(), c.location(e), e)
	}
}

func (c *checker) checkInt64Literal(e *exprpb.Expr) {
	c.setType(e, types.IntType)
}

func (c *checker) checkUint64Literal(e *exprpb.Expr) {
	c.setType(e, types.UintType)
}

func (c *checker) checkStringLiteral(e *exprpb.Expr) {
	c.setType(e, types.StringType)
}

func (c *checker) checkBytesLiteral(e *exprpb.Expr) {
	c.setType(e, types.BytesType)
}

func (c *checker) checkDoubleLiteral(e *exprpb.Expr) {
	c.setType(e, types.DoubleType)
}

func (c *checker) checkBoolLiteral(e *exprpb.Expr) {
	c.setType(e, types.BoolType)
}

func (c *checker) checkNullLiteral(e *exprpb.Expr) {
	c.setType(e, types.NullType)
}

func (c *checker) checkIdent(e *exprpb.Expr) {
	identExpr := e.GetIdentExpr()
	// Check to see if the identifier is declared.
	if ident := c.env.LookupIdent(identExpr.GetName()); ident != nil {
		c.setType(e, ident.Type())
		c.setReference(e, ast.NewIdentReference(ident.Name(), ident.Value()))
		// Overwrite the identifier with its fully qualified name.
		identExpr.Name = ident.Name()
		return
	}

	c.setType(e, types.ErrorType)
	c.errors.undeclaredReference(e.GetId(), c.location(e), c.env.container.Name(), identExpr.GetName())
}

func (c *checker) checkSelect(e *exprpb.Expr) {
	sel := e.GetSelectExpr()
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
			identName := ident.Name()
			e.ExprKind = &exprpb.Expr_IdentExpr{
				IdentExpr: &exprpb.Expr_Ident{
					Name: identName,
				},
			}
			return
		}
	}

	resultType := c.checkSelectField(e, sel.GetOperand(), sel.GetField(), false)
	if sel.TestOnly {
		resultType = types.BoolType
	}
	c.setType(e, substitute(c.mappings, resultType, false))
}

func (c *checker) checkOptSelect(e *exprpb.Expr) {
	// Collect metadata related to the opt select call packaged by the parser.
	call := e.GetCallExpr()
	operand := call.GetArgs()[0]
	field := call.GetArgs()[1]
	fieldName, isString := maybeUnwrapString(field)
	if !isString {
		c.errors.notAnOptionalFieldSelection(field.GetId(), c.location(field), field)
		return
	}

	// Perform type-checking using the field selection logic.
	resultType := c.checkSelectField(e, operand, fieldName, true)
	c.setType(e, substitute(c.mappings, resultType, false))
	c.setReference(e, ast.NewFunctionReference("select_optional_field"))
}

func (c *checker) checkSelectField(e, operand *exprpb.Expr, field string, optional bool) *types.Type {
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
		if fieldType, found := c.lookupFieldType(e.GetId(), messageType.TypeName(), field); found {
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
			c.errors.typeDoesNotSupportFieldSelection(e.GetId(), c.location(e), targetType)
		}
		resultType = types.DynType
	}

	// If the target type was optional coming in, then the result must be optional going out.
	if isOpt || optional {
		return types.NewOptionalType(resultType)
	}
	return resultType
}

func (c *checker) checkCall(e *exprpb.Expr) {
	// Note: similar logic exists within the `interpreter/planner.go`. If making changes here
	// please consider the impact on planner.go and consolidate implementations or mirror code
	// as appropriate.
	call := e.GetCallExpr()
	fnName := call.GetFunction()
	if fnName == operators.OptSelect {
		c.checkOptSelect(e)
		return
	}

	args := call.GetArgs()
	// Traverse arguments.
	for _, arg := range args {
		c.check(arg)
	}

	target := call.GetTarget()
	// Regular static call with simple name.
	if target == nil {
		// Check for the existence of the function.
		fn := c.env.LookupFunction(fnName)
		if fn == nil {
			c.errors.undeclaredReference(e.GetId(), c.location(e), c.env.container.Name(), fnName)
			c.setType(e, types.ErrorType)
			return
		}
		// Overwrite the function name with its fully qualified resolved name.
		call.Function = fn.Name()
		// Check to see whether the overload resolves.
		c.resolveOverloadOrError(e, fn, nil, args)
		return
	}

	// If a receiver 'target' is present, it may either be a receiver function, or a namespaced
	// function, but not both. Given a.b.c() either a.b.c is a function or c is a function with
	// target a.b.
	//
	// Check whether the target is a namespaced function name.
	qualifiedPrefix, maybeQualified := containers.ToQualifiedName(target)
	if maybeQualified {
		maybeQualifiedName := qualifiedPrefix + "." + fnName
		fn := c.env.LookupFunction(maybeQualifiedName)
		if fn != nil {
			// The function name is namespaced and so preserving the target operand would
			// be an inaccurate representation of the desired evaluation behavior.
			// Overwrite with fully-qualified resolved function name sans receiver target.
			call.Target = nil
			call.Function = fn.Name()
			c.resolveOverloadOrError(e, fn, nil, args)
			return
		}
	}

	// Regular instance call.
	c.check(call.Target)
	fn := c.env.LookupFunction(fnName)
	// Function found, attempt overload resolution.
	if fn != nil {
		c.resolveOverloadOrError(e, fn, target, args)
		return
	}
	// Function name not declared, record error.
	c.setType(e, types.ErrorType)
	c.errors.undeclaredReference(e.GetId(), c.location(e), c.env.container.Name(), fnName)
}

func (c *checker) resolveOverloadOrError(
	e *exprpb.Expr, fn *decls.FunctionDecl, target *exprpb.Expr, args []*exprpb.Expr) {
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
	call *exprpb.Expr, fn *decls.FunctionDecl, target *exprpb.Expr, args []*exprpb.Expr) *overloadResolution {

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
						args[i].GetId(),
						c.locationByID(args[i].GetId()),
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
		c.errors.noMatchingOverload(call.GetId(), c.location(call), fn.Name(), argTypes, target != nil)
		return nil
	}

	return newResolution(checkedRef, resultType)
}

func (c *checker) checkCreateList(e *exprpb.Expr) {
	create := e.GetListExpr()
	var elemsType *types.Type
	optionalIndices := create.GetOptionalIndices()
	optionals := make(map[int32]bool, len(optionalIndices))
	for _, optInd := range optionalIndices {
		optionals[optInd] = true
	}
	for i, e := range create.GetElements() {
		c.check(e)
		elemType := c.getType(e)
		if optionals[int32(i)] {
			var isOptional bool
			elemType, isOptional = maybeUnwrapOptional(elemType)
			if !isOptional && !isDyn(elemType) {
				c.errors.typeMismatch(e.GetId(), c.location(e), types.NewOptionalType(elemType), elemType)
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

func (c *checker) checkCreateStruct(e *exprpb.Expr) {
	str := e.GetStructExpr()
	if str.GetMessageName() != "" {
		c.checkCreateMessage(e)
	} else {
		c.checkCreateMap(e)
	}
}

func (c *checker) checkCreateMap(e *exprpb.Expr) {
	mapVal := e.GetStructExpr()
	var mapKeyType *types.Type
	var mapValueType *types.Type
	for _, ent := range mapVal.GetEntries() {
		key := ent.GetMapKey()
		c.check(key)
		mapKeyType = c.joinTypes(key, mapKeyType, c.getType(key))

		val := ent.GetValue()
		c.check(val)
		valType := c.getType(val)
		if ent.GetOptionalEntry() {
			var isOptional bool
			valType, isOptional = maybeUnwrapOptional(valType)
			if !isOptional && !isDyn(valType) {
				c.errors.typeMismatch(val.GetId(), c.location(val), types.NewOptionalType(valType), valType)
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

func (c *checker) checkCreateMessage(e *exprpb.Expr) {
	msgVal := e.GetStructExpr()
	// Determine the type of the message.
	resultType := types.ErrorType
	ident := c.env.LookupIdent(msgVal.GetMessageName())
	if ident == nil {
		c.errors.undeclaredReference(
			e.GetId(), c.location(e), c.env.container.Name(), msgVal.GetMessageName())
		c.setType(e, types.ErrorType)
		return
	}
	// Ensure the type name is fully qualified in the AST.
	typeName := ident.Name()
	msgVal.MessageName = typeName
	c.setReference(e, ast.NewIdentReference(ident.Name(), nil))
	identKind := ident.Type().Kind()
	if identKind != types.ErrorKind {
		if identKind != types.TypeKind {
			c.errors.notAType(e.GetId(), c.location(e), ident.Type().DeclaredTypeName())
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
				c.errors.notAMessageType(e.GetId(), c.location(e), resultType.DeclaredTypeName())
				resultType = types.ErrorType
			}
		}
	}
	c.setType(e, resultType)

	// Check the field initializers.
	for _, ent := range msgVal.GetEntries() {
		field := ent.GetFieldKey()
		value := ent.GetValue()
		c.check(value)

		fieldType := types.ErrorType
		ft, found := c.lookupFieldType(ent.GetId(), typeName, field)
		if found {
			fieldType = ft
		}

		valType := c.getType(value)
		if ent.GetOptionalEntry() {
			var isOptional bool
			valType, isOptional = maybeUnwrapOptional(valType)
			if !isOptional && !isDyn(valType) {
				c.errors.typeMismatch(value.GetId(), c.location(value), types.NewOptionalType(valType), valType)
			}
		}
		if !c.isAssignable(fieldType, valType) {
			c.errors.fieldTypeMismatch(ent.GetId(), c.locationByID(ent.GetId()), field, fieldType, valType)
		}
	}
}

func (c *checker) checkComprehension(e *exprpb.Expr) {
	comp := e.GetComprehensionExpr()
	c.check(comp.GetIterRange())
	c.check(comp.GetAccuInit())
	accuType := c.getType(comp.GetAccuInit())
	rangeType := substitute(c.mappings, c.getType(comp.GetIterRange()), false)
	var varType *types.Type

	switch rangeType.Kind() {
	case types.ListKind:
		varType = rangeType.Parameters()[0]
	case types.MapKind:
		// Ranges over the keys.
		varType = rangeType.Parameters()[0]
	case types.DynKind, types.ErrorKind, types.TypeParamKind:
		// Set the range type to DYN to prevent assignment to a potentially incorrect type
		// at a later point in type-checking. The isAssignable call will update the type
		// substitutions for the type param under the covers.
		c.isAssignable(types.DynType, rangeType)
		// Set the range iteration variable to type DYN as well.
		varType = types.DynType
	default:
		c.errors.notAComprehensionRange(comp.GetIterRange().GetId(), c.location(comp.GetIterRange()), rangeType)
		varType = types.ErrorType
	}

	// Create a scope for the comprehension since it has a local accumulation variable.
	// This scope will contain the accumulation variable used to compute the result.
	c.env = c.env.enterScope()
	c.env.AddIdents(decls.NewVariable(comp.GetAccuVar(), accuType))
	// Create a block scope for the loop.
	c.env = c.env.enterScope()
	c.env.AddIdents(decls.NewVariable(comp.GetIterVar(), varType))
	// Check the variable references in the condition and step.
	c.check(comp.GetLoopCondition())
	c.assertType(comp.GetLoopCondition(), types.BoolType)
	c.check(comp.GetLoopStep())
	c.assertType(comp.GetLoopStep(), accuType)
	// Exit the loop's block scope before checking the result.
	c.env = c.env.exitScope()
	c.check(comp.GetResult())
	// Exit the comprehension scope.
	c.env = c.env.exitScope()
	c.setType(e, substitute(c.mappings, c.getType(comp.GetResult()), false))
}

// Checks compatibility of joined types, and returns the most general common type.
func (c *checker) joinTypes(e *exprpb.Expr, previous, current *types.Type) *types.Type {
	if previous == nil {
		return current
	}
	if c.isAssignable(previous, current) {
		return mostGeneral(previous, current)
	}
	if c.dynAggregateLiteralElementTypesEnabled() {
		return types.DynType
	}
	c.errors.typeMismatch(e.GetId(), c.location(e), previous, current)
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

func maybeUnwrapString(e *exprpb.Expr) (string, bool) {
	switch e.GetExprKind().(type) {
	case *exprpb.Expr_ConstExpr:
		literal := e.GetConstExpr()
		switch literal.GetConstantKind().(type) {
		case *exprpb.Constant_StringValue:
			return literal.GetStringValue(), true
		}
	}
	return "", false
}

func (c *checker) setType(e *exprpb.Expr, t *types.Type) {
	if old, found := c.types[e.GetId()]; found && !old.IsExactType(t) {
		c.errors.incompatibleType(e.GetId(), c.location(e), e, old, t)
		return
	}
	c.types[e.GetId()] = t
}

func (c *checker) getType(e *exprpb.Expr) *types.Type {
	return c.types[e.GetId()]
}

func (c *checker) setReference(e *exprpb.Expr, r *ast.ReferenceInfo) {
	if old, found := c.references[e.GetId()]; found && !old.Equals(r) {
		c.errors.referenceRedefinition(e.GetId(), c.location(e), e, old, r)
		return
	}
	c.references[e.GetId()] = r
}

func (c *checker) assertType(e *exprpb.Expr, t *types.Type) {
	if !c.isAssignable(t, c.getType(e)) {
		c.errors.typeMismatch(e.GetId(), c.location(e), t, c.getType(e))
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

func (c *checker) location(e *exprpb.Expr) common.Location {
	return c.locationByID(e.GetId())
}

func (c *checker) locationByID(id int64) common.Location {
	positions := c.sourceInfo.GetPositions()
	var line = 1
	if offset, found := positions[id]; found {
		col := int(offset)
		for _, lineOffset := range c.sourceInfo.GetLineOffsets() {
			if lineOffset < offset {
				line++
				col = int(offset - lineOffset)
			} else {
				break
			}
		}
		return common.NewLocation(line, col)
	}
	return common.NoLocation
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
