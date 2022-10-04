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

	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/types/ref"

	"google.golang.org/protobuf/proto"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

type checker struct {
	env                *Env
	errors             *typeErrors
	mappings           *mapping
	freeTypeVarCounter int
	sourceInfo         *exprpb.SourceInfo
	types              map[int64]*exprpb.Type
	references         map[int64]*exprpb.Reference
}

// Check performs type checking, giving a typed AST.
// The input is a ParsedExpr proto and an env which encapsulates
// type binding of variables, declarations of built-in functions,
// descriptions of protocol buffers, and a registry for errors.
// Returns a CheckedExpr proto, which might not be usable if
// there are errors in the error registry.
func Check(parsedExpr *exprpb.ParsedExpr,
	source common.Source,
	env *Env) (*exprpb.CheckedExpr, *common.Errors) {
	c := checker{
		env:                env,
		errors:             &typeErrors{common.NewErrors(source)},
		mappings:           newMapping(),
		freeTypeVarCounter: 0,
		sourceInfo:         parsedExpr.GetSourceInfo(),
		types:              make(map[int64]*exprpb.Type),
		references:         make(map[int64]*exprpb.Reference),
	}
	c.check(parsedExpr.GetExpr())

	// Walk over the final type map substituting any type parameters either by their bound value or
	// by DYN.
	m := make(map[int64]*exprpb.Type)
	for k, v := range c.types {
		m[k] = substitute(c.mappings, v, true)
	}

	return &exprpb.CheckedExpr{
		Expr:         parsedExpr.GetExpr(),
		SourceInfo:   parsedExpr.GetSourceInfo(),
		TypeMap:      m,
		ReferenceMap: c.references,
	}, c.errors.Errors
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
		c.errors.ReportError(
			c.location(e), "Unrecognized ast type: %v", reflect.TypeOf(e))
	}
}

func (c *checker) checkInt64Literal(e *exprpb.Expr) {
	c.setType(e, decls.Int)
}

func (c *checker) checkUint64Literal(e *exprpb.Expr) {
	c.setType(e, decls.Uint)
}

func (c *checker) checkStringLiteral(e *exprpb.Expr) {
	c.setType(e, decls.String)
}

func (c *checker) checkBytesLiteral(e *exprpb.Expr) {
	c.setType(e, decls.Bytes)
}

func (c *checker) checkDoubleLiteral(e *exprpb.Expr) {
	c.setType(e, decls.Double)
}

func (c *checker) checkBoolLiteral(e *exprpb.Expr) {
	c.setType(e, decls.Bool)
}

func (c *checker) checkNullLiteral(e *exprpb.Expr) {
	c.setType(e, decls.Null)
}

func (c *checker) checkIdent(e *exprpb.Expr) {
	identExpr := e.GetIdentExpr()
	// Check to see if the identifier is declared.
	if ident := c.env.LookupIdent(identExpr.GetName()); ident != nil {
		c.setType(e, ident.GetIdent().GetType())
		c.setReference(e, newIdentReference(ident.GetName(), ident.GetIdent().GetValue()))
		// Overwrite the identifier with its fully qualified name.
		identExpr.Name = ident.GetName()
		return
	}

	c.setType(e, decls.Error)
	c.errors.undeclaredReference(
		c.location(e), c.env.container.Name(), identExpr.GetName())
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
			c.setType(e, ident.GetIdent().Type)
			c.setReference(e, newIdentReference(ident.GetName(), ident.GetIdent().Value))
			identName := ident.GetName()
			e.ExprKind = &exprpb.Expr_IdentExpr{
				IdentExpr: &exprpb.Expr_Ident{
					Name: identName,
				},
			}
			return
		}
	}

	// Interpret as field selection, first traversing down the operand.
	c.check(sel.GetOperand())
	targetType := substitute(c.mappings, c.getType(sel.GetOperand()), false)
	// Assume error type by default as most types do not support field selection.
	resultType := decls.Error
	switch kindOf(targetType) {
	case kindMap:
		// Maps yield their value type as the selection result type.
		mapType := targetType.GetMapType()
		resultType = mapType.GetValueType()
	case kindObject:
		// Objects yield their field type declaration as the selection result type, but only if
		// the field is defined.
		messageType := targetType
		if fieldType, found := c.lookupFieldType(c.location(e), messageType.GetMessageType(), sel.GetField()); found {
			resultType = fieldType.Type
		}
	case kindTypeParam:
		// Set the operand type to DYN to prevent assignment to a potentially incorrect type
		// at a later point in type-checking. The isAssignable call will update the type
		// substitutions for the type param under the covers.
		c.isAssignable(decls.Dyn, targetType)
		// Also, set the result type to DYN.
		resultType = decls.Dyn
	default:
		// Dynamic / error values are treated as DYN type. Errors are handled this way as well
		// in order to allow forward progress on the check.
		if isDynOrError(targetType) {
			resultType = decls.Dyn
		} else {
			c.errors.typeDoesNotSupportFieldSelection(c.location(e), targetType)
		}
	}
	if sel.TestOnly {
		resultType = decls.Bool
	}
	c.setType(e, substitute(c.mappings, resultType, false))
}

func (c *checker) checkCall(e *exprpb.Expr) {
	// Note: similar logic exists within the `interpreter/planner.go`. If making changes here
	// please consider the impact on planner.go and consolidate implementations or mirror code
	// as appropriate.
	call := e.GetCallExpr()
	target := call.GetTarget()
	args := call.GetArgs()
	fnName := call.GetFunction()

	// Traverse arguments.
	for _, arg := range args {
		c.check(arg)
	}

	// Regular static call with simple name.
	if target == nil {
		// Check for the existence of the function.
		fn := c.env.LookupFunction(fnName)
		if fn == nil {
			c.errors.undeclaredReference(
				c.location(e), c.env.container.Name(), fnName)
			c.setType(e, decls.Error)
			return
		}
		// Overwrite the function name with its fully qualified resolved name.
		call.Function = fn.GetName()
		// Check to see whether the overload resolves.
		c.resolveOverloadOrError(c.location(e), e, fn, nil, args)
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
			call.Function = fn.GetName()
			c.resolveOverloadOrError(c.location(e), e, fn, nil, args)
			return
		}
	}

	// Regular instance call.
	c.check(call.Target)
	fn := c.env.LookupFunction(fnName)
	// Function found, attempt overload resolution.
	if fn != nil {
		c.resolveOverloadOrError(c.location(e), e, fn, target, args)
		return
	}
	// Function name not declared, record error.
	c.errors.undeclaredReference(c.location(e), c.env.container.Name(), fnName)
}

func (c *checker) resolveOverloadOrError(
	loc common.Location,
	e *exprpb.Expr,
	fn *exprpb.Decl, target *exprpb.Expr, args []*exprpb.Expr) {
	// Attempt to resolve the overload.
	resolution := c.resolveOverload(loc, fn, target, args)
	// No such overload, error noted in the resolveOverload call, type recorded here.
	if resolution == nil {
		c.setType(e, decls.Error)
		return
	}
	// Overload found.
	c.setType(e, resolution.Type)
	c.setReference(e, resolution.Reference)
}

func (c *checker) resolveOverload(
	loc common.Location,
	fn *exprpb.Decl, target *exprpb.Expr, args []*exprpb.Expr) *overloadResolution {

	var argTypes []*exprpb.Type
	if target != nil {
		argTypes = append(argTypes, c.getType(target))
	}
	for _, arg := range args {
		argTypes = append(argTypes, c.getType(arg))
	}

	var resultType *exprpb.Type
	var checkedRef *exprpb.Reference
	for _, overload := range fn.GetFunction().GetOverloads() {
		// Determine whether the overload is currently considered.
		if c.env.isOverloadDisabled(overload.GetOverloadId()) {
			continue
		}

		// Ensure the call style for the overload matches.
		if (target == nil && overload.GetIsInstanceFunction()) ||
			(target != nil && !overload.GetIsInstanceFunction()) {
			// not a compatible call style.
			continue
		}

		overloadType := decls.NewFunctionType(overload.ResultType, overload.Params...)
		if len(overload.GetTypeParams()) > 0 {
			// Instantiate overload's type with fresh type variables.
			substitutions := newMapping()
			for _, typePar := range overload.GetTypeParams() {
				substitutions.add(decls.NewTypeParamType(typePar), c.newTypeVar())
			}
			overloadType = substitute(substitutions, overloadType, false)
		}

		candidateArgTypes := overloadType.GetFunction().GetArgTypes()
		if c.isAssignableList(argTypes, candidateArgTypes) {
			if checkedRef == nil {
				checkedRef = newFunctionReference(overload.GetOverloadId())
			} else {
				checkedRef.OverloadId = append(checkedRef.GetOverloadId(), overload.GetOverloadId())
			}

			// First matching overload, determines result type.
			fnResultType := substitute(c.mappings, overloadType.GetFunction().GetResultType(), false)
			if resultType == nil {
				resultType = fnResultType
			} else if !isDyn(resultType) && !proto.Equal(fnResultType, resultType) {
				resultType = decls.Dyn
			}
		}
	}

	if resultType == nil {
		c.errors.noMatchingOverload(loc, fn.GetName(), argTypes, target != nil)
		resultType = decls.Error
		return nil
	}

	return newResolution(checkedRef, resultType)
}

func (c *checker) checkCreateList(e *exprpb.Expr) {
	create := e.GetListExpr()
	var elemType *exprpb.Type
	for _, e := range create.GetElements() {
		c.check(e)
		elemType = c.joinTypes(c.location(e), elemType, c.getType(e))
	}
	if elemType == nil {
		// If the list is empty, assign free type var to elem type.
		elemType = c.newTypeVar()
	}
	c.setType(e, decls.NewListType(elemType))
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
	var keyType *exprpb.Type
	var valueType *exprpb.Type
	for _, ent := range mapVal.GetEntries() {
		key := ent.GetMapKey()
		c.check(key)
		keyType = c.joinTypes(c.location(key), keyType, c.getType(key))

		c.check(ent.Value)
		valueType = c.joinTypes(c.location(ent.Value), valueType, c.getType(ent.Value))
	}
	if keyType == nil {
		// If the map is empty, assign free type variables to typeKey and value type.
		keyType = c.newTypeVar()
		valueType = c.newTypeVar()
	}
	c.setType(e, decls.NewMapType(keyType, valueType))
}

func (c *checker) checkCreateMessage(e *exprpb.Expr) {
	msgVal := e.GetStructExpr()
	// Determine the type of the message.
	messageType := decls.Error
	decl := c.env.LookupIdent(msgVal.GetMessageName())
	if decl == nil {
		c.errors.undeclaredReference(
			c.location(e), c.env.container.Name(), msgVal.GetMessageName())
		return
	}
	// Ensure the type name is fully qualified in the AST.
	msgVal.MessageName = decl.GetName()
	c.setReference(e, newIdentReference(decl.GetName(), nil))
	ident := decl.GetIdent()
	identKind := kindOf(ident.GetType())
	if identKind != kindError {
		if identKind != kindType {
			c.errors.notAType(c.location(e), ident.GetType())
		} else {
			messageType = ident.GetType().GetType()
			if kindOf(messageType) != kindObject {
				c.errors.notAMessageType(c.location(e), messageType)
				messageType = decls.Error
			}
		}
	}
	if isObjectWellKnownType(messageType) {
		c.setType(e, getObjectWellKnownType(messageType))
	} else {
		c.setType(e, messageType)
	}

	// Check the field initializers.
	for _, ent := range msgVal.GetEntries() {
		field := ent.GetFieldKey()
		value := ent.GetValue()
		c.check(value)

		fieldType := decls.Error
		if t, found := c.lookupFieldType(
			c.locationByID(ent.GetId()),
			messageType.GetMessageType(),
			field); found {
			fieldType = t.Type
		}
		if !c.isAssignable(fieldType, c.getType(value)) {
			c.errors.fieldTypeMismatch(
				c.locationByID(ent.Id), field, fieldType, c.getType(value))
		}
	}
}

func (c *checker) checkComprehension(e *exprpb.Expr) {
	comp := e.GetComprehensionExpr()
	c.check(comp.GetIterRange())
	c.check(comp.GetAccuInit())
	accuType := c.getType(comp.GetAccuInit())
	rangeType := substitute(c.mappings, c.getType(comp.GetIterRange()), false)
	var varType *exprpb.Type

	switch kindOf(rangeType) {
	case kindList:
		varType = rangeType.GetListType().GetElemType()
	case kindMap:
		// Ranges over the keys.
		varType = rangeType.GetMapType().GetKeyType()
	case kindDyn, kindError, kindTypeParam:
		// Set the range type to DYN to prevent assignment to a potentially incorrect type
		// at a later point in type-checking. The isAssignable call will update the type
		// substitutions for the type param under the covers.
		c.isAssignable(decls.Dyn, rangeType)
		// Set the range iteration variable to type DYN as well.
		varType = decls.Dyn
	default:
		c.errors.notAComprehensionRange(c.location(comp.GetIterRange()), rangeType)
		varType = decls.Error
	}

	// Create a scope for the comprehension since it has a local accumulation variable.
	// This scope will contain the accumulation variable used to compute the result.
	c.env = c.env.enterScope()
	c.env.Add(decls.NewVar(comp.GetAccuVar(), accuType))
	// Create a block scope for the loop.
	c.env = c.env.enterScope()
	c.env.Add(decls.NewVar(comp.GetIterVar(), varType))
	// Check the variable references in the condition and step.
	c.check(comp.GetLoopCondition())
	c.assertType(comp.GetLoopCondition(), decls.Bool)
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
func (c *checker) joinTypes(loc common.Location,
	previous *exprpb.Type,
	current *exprpb.Type) *exprpb.Type {
	if previous == nil {
		return current
	}
	if c.isAssignable(previous, current) {
		return mostGeneral(previous, current)
	}
	if c.dynAggregateLiteralElementTypesEnabled() {
		return decls.Dyn
	}
	c.errors.typeMismatch(loc, previous, current)
	return decls.Error
}

func (c *checker) dynAggregateLiteralElementTypesEnabled() bool {
	return c.env.aggLitElemType == dynElementType
}

func (c *checker) newTypeVar() *exprpb.Type {
	id := c.freeTypeVarCounter
	c.freeTypeVarCounter++
	return decls.NewTypeParamType(fmt.Sprintf("_var%d", id))
}

func (c *checker) isAssignable(t1 *exprpb.Type, t2 *exprpb.Type) bool {
	subs := isAssignable(c.mappings, t1, t2)
	if subs != nil {
		c.mappings = subs
		return true
	}

	return false
}

func (c *checker) isAssignableList(l1 []*exprpb.Type, l2 []*exprpb.Type) bool {
	subs := isAssignableList(c.mappings, l1, l2)
	if subs != nil {
		c.mappings = subs
		return true
	}

	return false
}

func (c *checker) lookupFieldType(l common.Location, messageType string, fieldName string) (*ref.FieldType, bool) {
	if _, found := c.env.provider.FindType(messageType); !found {
		// This should not happen, anyway, report an error.
		c.errors.unexpectedFailedResolution(l, messageType)
		return nil, false
	}

	if ft, found := c.env.provider.FindFieldType(messageType, fieldName); found {
		return ft, found
	}

	c.errors.undefinedField(l, fieldName)
	return nil, false
}

func (c *checker) setType(e *exprpb.Expr, t *exprpb.Type) {
	if old, found := c.types[e.GetId()]; found && !proto.Equal(old, t) {
		c.errors.ReportError(c.location(e),
			"(Incompatible) Type already exists for expression: %v(%d) old:%v, new:%v", e, e.GetId(), old, t)
		return
	}
	c.types[e.GetId()] = t
}

func (c *checker) getType(e *exprpb.Expr) *exprpb.Type {
	return c.types[e.GetId()]
}

func (c *checker) setReference(e *exprpb.Expr, r *exprpb.Reference) {
	if old, found := c.references[e.GetId()]; found && !proto.Equal(old, r) {
		c.errors.ReportError(c.location(e),
			"Reference already exists for expression: %v(%d) old:%v, new:%v", e, e.GetId(), old, r)
		return
	}
	c.references[e.GetId()] = r
}

func (c *checker) assertType(e *exprpb.Expr, t *exprpb.Type) {
	if !c.isAssignable(t, c.getType(e)) {
		c.errors.typeMismatch(c.location(e), t, c.getType(e))
	}
}

type overloadResolution struct {
	Reference *exprpb.Reference
	Type      *exprpb.Type
}

func newResolution(checkedRef *exprpb.Reference, t *exprpb.Type) *overloadResolution {
	return &overloadResolution{
		Reference: checkedRef,
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

func newIdentReference(name string, value *exprpb.Constant) *exprpb.Reference {
	return &exprpb.Reference{Name: name, Value: value}
}

func newFunctionReference(overloads ...string) *exprpb.Reference {
	return &exprpb.Reference{OverloadId: overloads}
}
