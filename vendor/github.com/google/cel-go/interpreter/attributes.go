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

package interpreter

import (
	"fmt"
	"strings"

	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

// AttributeFactory provides methods creating Attribute and Qualifier values.
type AttributeFactory interface {
	// AbsoluteAttribute creates an attribute that refers to a top-level variable name.
	//
	// Checked expressions generate absolute attribute with a single name.
	// Parse-only expressions may have more than one possible absolute identifier when the
	// expression is created within a container, e.g. package or namespace.
	//
	// When there is more than one name supplied to the AbsoluteAttribute call, the names
	// must be in CEL's namespace resolution order. The name arguments provided here are
	// returned in the same order as they were provided by the NamespacedAttribute
	// CandidateVariableNames method.
	AbsoluteAttribute(id int64, names ...string) NamespacedAttribute

	// ConditionalAttribute creates an attribute with two Attribute branches, where the Attribute
	// that is resolved depends on the boolean evaluation of the input 'expr'.
	ConditionalAttribute(id int64, expr Interpretable, t, f Attribute) Attribute

	// MaybeAttribute creates an attribute that refers to either a field selection or a namespaced
	// variable name.
	//
	// Only expressions which have not been type-checked may generate oneof attributes.
	MaybeAttribute(id int64, name string) Attribute

	// RelativeAttribute creates an attribute whose value is a qualification of a dynamic
	// computation rather than a static variable reference.
	RelativeAttribute(id int64, operand Interpretable) Attribute

	// NewQualifier creates a qualifier on the target object with a given value.
	//
	// The 'val' may be an Attribute or any proto-supported map key type: bool, int, string, uint.
	//
	// The qualifier may consider the object type being qualified, if present. If absent, the
	// qualification should be considered dynamic and the qualification should still work, though
	// it may be sub-optimal.
	NewQualifier(objType *types.Type, qualID int64, val any, opt bool) (Qualifier, error)
}

// Qualifier marker interface for designating different qualifier values and where they appear
// within field selections and index call expressions (`_[_]`).
type Qualifier interface {
	// ID where the qualifier appears within an expression.
	ID() int64

	// IsOptional specifies whether the qualifier is optional.
	// Instead of a direct qualification, an optional qualifier will be resolved via QualifyIfPresent
	// rather than Qualify. A non-optional qualifier may also be resolved through QualifyIfPresent if
	// the object to qualify is itself optional.
	IsOptional() bool

	// Qualify performs a qualification, e.g. field selection, on the input object and returns
	// the value of the access and whether the value was set. A non-nil value with a false presence
	// test result indicates that the value being returned is the default value.
	Qualify(vars Activation, obj any) (any, error)

	// QualifyIfPresent qualifies the object if the qualifier is declared or defined on the object.
	// The 'presenceOnly' flag indicates that the value is not necessary, just a boolean status as
	// to whether the qualifier is present.
	QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error)
}

// ConstantQualifier interface embeds the Qualifier interface and provides an option to inspect the
// qualifier's constant value.
//
// Non-constant qualifiers are of Attribute type.
type ConstantQualifier interface {
	Qualifier

	// Value returns the constant value associated with the qualifier.
	Value() ref.Val
}

// Attribute values are a variable or value with an optional set of qualifiers, such as field, key,
// or index accesses.
type Attribute interface {
	Qualifier

	// AddQualifier adds a qualifier on the Attribute or error if the qualification is not a valid qualifier type.
	AddQualifier(Qualifier) (Attribute, error)

	// Resolve returns the value of the Attribute and whether it was present given an Activation.
	// For objects which support safe traversal, the value may be non-nil and the presence flag be false.
	//
	// If an error is encountered during attribute resolution, it will be returned immediately.
	// If the attribute cannot be resolved within the Activation, the result must be: `nil`, `error`
	// with the error indicating which variable was missing.
	Resolve(Activation) (any, error)
}

// NamespacedAttribute values are a variable within a namespace, and an optional set of qualifiers
// such as field, key, or index accesses.
type NamespacedAttribute interface {
	Attribute

	// CandidateVariableNames returns the possible namespaced variable names for this Attribute in
	// the CEL namespace resolution order.
	CandidateVariableNames() []string

	// Qualifiers returns the list of qualifiers associated with the Attribute.
	Qualifiers() []Qualifier
}

// NewAttributeFactory returns a default AttributeFactory which is produces Attribute values
// capable of resolving types by simple names and qualify the values using the supported qualifier
// types: bool, int, string, and uint.
func NewAttributeFactory(cont *containers.Container, a types.Adapter, p types.Provider) AttributeFactory {
	return &attrFactory{
		container: cont,
		adapter:   a,
		provider:  p,
	}
}

type attrFactory struct {
	container *containers.Container
	adapter   types.Adapter
	provider  types.Provider
}

// AbsoluteAttribute refers to a variable value and an optional qualifier path.
//
// The namespaceNames represent the names the variable could have based on namespace
// resolution rules.
func (r *attrFactory) AbsoluteAttribute(id int64, names ...string) NamespacedAttribute {
	return &absoluteAttribute{
		id:             id,
		namespaceNames: names,
		qualifiers:     []Qualifier{},
		adapter:        r.adapter,
		provider:       r.provider,
		fac:            r,
	}
}

// ConditionalAttribute supports the case where an attribute selection may occur on a conditional
// expression, e.g. (cond ? a : b).c
func (r *attrFactory) ConditionalAttribute(id int64, expr Interpretable, t, f Attribute) Attribute {
	return &conditionalAttribute{
		id:      id,
		expr:    expr,
		truthy:  t,
		falsy:   f,
		adapter: r.adapter,
		fac:     r,
	}
}

// MaybeAttribute collects variants of unchecked AbsoluteAttribute values which could either be
// direct variable accesses or some combination of variable access with qualification.
func (r *attrFactory) MaybeAttribute(id int64, name string) Attribute {
	return &maybeAttribute{
		id: id,
		attrs: []NamespacedAttribute{
			r.AbsoluteAttribute(id, r.container.ResolveCandidateNames(name)...),
		},
		adapter:  r.adapter,
		provider: r.provider,
		fac:      r,
	}
}

// RelativeAttribute refers to an expression and an optional qualifier path.
func (r *attrFactory) RelativeAttribute(id int64, operand Interpretable) Attribute {
	return &relativeAttribute{
		id:         id,
		operand:    operand,
		qualifiers: []Qualifier{},
		adapter:    r.adapter,
		fac:        r,
	}
}

// NewQualifier is an implementation of the AttributeFactory interface.
func (r *attrFactory) NewQualifier(objType *types.Type, qualID int64, val any, opt bool) (Qualifier, error) {
	// Before creating a new qualifier check to see if this is a protobuf message field access.
	// If so, use the precomputed GetFrom qualification method rather than the standard
	// stringQualifier.
	str, isStr := val.(string)
	if isStr && objType != nil && objType.Kind() == types.StructKind {
		ft, found := r.provider.FindStructFieldType(objType.TypeName(), str)
		if found && ft.IsSet != nil && ft.GetFrom != nil {
			return &fieldQualifier{
				id:        qualID,
				Name:      str,
				FieldType: ft,
				adapter:   r.adapter,
				optional:  opt,
			}, nil
		}
	}
	return newQualifier(r.adapter, qualID, val, opt)
}

type absoluteAttribute struct {
	id int64
	// namespaceNames represent the names the variable could have based on declared container
	// (package) of the expression.
	namespaceNames []string
	qualifiers     []Qualifier
	adapter        types.Adapter
	provider       types.Provider
	fac            AttributeFactory
}

// ID implements the Attribute interface method.
func (a *absoluteAttribute) ID() int64 {
	qualCount := len(a.qualifiers)
	if qualCount == 0 {
		return a.id
	}
	return a.qualifiers[qualCount-1].ID()
}

// IsOptional returns trivially false for an attribute as the attribute represents a fully
// qualified variable name. If the attribute is used in an optional manner, then an attrQualifier
// is created and marks the attribute as optional.
func (a *absoluteAttribute) IsOptional() bool {
	return false
}

// AddQualifier implements the Attribute interface method.
func (a *absoluteAttribute) AddQualifier(qual Qualifier) (Attribute, error) {
	a.qualifiers = append(a.qualifiers, qual)
	return a, nil
}

// CandidateVariableNames implements the NamespaceAttribute interface method.
func (a *absoluteAttribute) CandidateVariableNames() []string {
	return a.namespaceNames
}

// Qualifiers returns the list of Qualifier instances associated with the namespaced attribute.
func (a *absoluteAttribute) Qualifiers() []Qualifier {
	return a.qualifiers
}

// Qualify is an implementation of the Qualifier interface method.
func (a *absoluteAttribute) Qualify(vars Activation, obj any) (any, error) {
	return attrQualify(a.fac, vars, obj, a)
}

// QualifyIfPresent is an implementation of the Qualifier interface method.
func (a *absoluteAttribute) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	return attrQualifyIfPresent(a.fac, vars, obj, a, presenceOnly)
}

// String implements the Stringer interface method.
func (a *absoluteAttribute) String() string {
	return fmt.Sprintf("id: %v, names: %v", a.id, a.namespaceNames)
}

// Resolve returns the resolved Attribute value given the Activation, or error if the Attribute
// variable is not found, or if its Qualifiers cannot be applied successfully.
//
// If the variable name cannot be found as an Activation variable or in the TypeProvider as
// a type, then the result is `nil`, `error` with the error indicating the name of the first
// variable searched as missing.
func (a *absoluteAttribute) Resolve(vars Activation) (any, error) {
	for _, nm := range a.namespaceNames {
		// If the variable is found, process it. Otherwise, wait until the checks to
		// determine whether the type is unknown before returning.
		obj, found := vars.ResolveName(nm)
		if found {
			if celErr, ok := obj.(*types.Err); ok {
				return nil, celErr.Unwrap()
			}
			obj, isOpt, err := applyQualifiers(vars, obj, a.qualifiers)
			if err != nil {
				return nil, err
			}
			if isOpt {
				val := a.adapter.NativeToValue(obj)
				if types.IsUnknown(val) {
					return val, nil
				}
				return types.OptionalOf(val), nil
			}
			return obj, nil
		}
		// Attempt to resolve the qualified type name if the name is not a variable identifier.
		typ, found := a.provider.FindIdent(nm)
		if found {
			if len(a.qualifiers) == 0 {
				return typ, nil
			}
		}
	}
	var attrNames strings.Builder
	for i, nm := range a.namespaceNames {
		if i != 0 {
			attrNames.WriteString(", ")
		}
		attrNames.WriteString(nm)
	}
	return nil, missingAttribute(attrNames.String())
}

type conditionalAttribute struct {
	id      int64
	expr    Interpretable
	truthy  Attribute
	falsy   Attribute
	adapter types.Adapter
	fac     AttributeFactory
}

// ID is an implementation of the Attribute interface method.
func (a *conditionalAttribute) ID() int64 {
	// There's a field access after the conditional.
	if a.truthy.ID() == a.falsy.ID() {
		return a.truthy.ID()
	}
	// Otherwise return the conditional id as the consistent id being tracked.
	return a.id
}

// IsOptional returns trivially false for an attribute as the attribute represents a fully
// qualified variable name. If the attribute is used in an optional manner, then an attrQualifier
// is created and marks the attribute as optional.
func (a *conditionalAttribute) IsOptional() bool {
	return false
}

// AddQualifier appends the same qualifier to both sides of the conditional, in effect managing
// the qualification of alternate attributes.
func (a *conditionalAttribute) AddQualifier(qual Qualifier) (Attribute, error) {
	_, err := a.truthy.AddQualifier(qual)
	if err != nil {
		return nil, err
	}
	_, err = a.falsy.AddQualifier(qual)
	if err != nil {
		return nil, err
	}
	return a, nil
}

// Qualify is an implementation of the Qualifier interface method.
func (a *conditionalAttribute) Qualify(vars Activation, obj any) (any, error) {
	return attrQualify(a.fac, vars, obj, a)
}

// QualifyIfPresent is an implementation of the Qualifier interface method.
func (a *conditionalAttribute) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	return attrQualifyIfPresent(a.fac, vars, obj, a, presenceOnly)
}

// Resolve evaluates the condition, and then resolves the truthy or falsy branch accordingly.
func (a *conditionalAttribute) Resolve(vars Activation) (any, error) {
	val := a.expr.Eval(vars)
	if val == types.True {
		return a.truthy.Resolve(vars)
	}
	if val == types.False {
		return a.falsy.Resolve(vars)
	}
	if types.IsUnknown(val) {
		return val, nil
	}
	return nil, types.MaybeNoSuchOverloadErr(val).(*types.Err)
}

// String is an implementation of the Stringer interface method.
func (a *conditionalAttribute) String() string {
	return fmt.Sprintf("id: %v, truthy attribute: %v, falsy attribute: %v", a.id, a.truthy, a.falsy)
}

type maybeAttribute struct {
	id       int64
	attrs    []NamespacedAttribute
	adapter  types.Adapter
	provider types.Provider
	fac      AttributeFactory
}

// ID is an implementation of the Attribute interface method.
func (a *maybeAttribute) ID() int64 {
	return a.attrs[0].ID()
}

// IsOptional returns trivially false for an attribute as the attribute represents a fully
// qualified variable name. If the attribute is used in an optional manner, then an attrQualifier
// is created and marks the attribute as optional.
func (a *maybeAttribute) IsOptional() bool {
	return false
}

// AddQualifier adds a qualifier to each possible attribute variant, and also creates
// a new namespaced variable from the qualified value.
//
// The algorithm for building the maybe attribute is as follows:
//
// 1. Create a maybe attribute from a simple identifier when it occurs in a parsed-only expression
//
//	mb = MaybeAttribute(<id>, "a")
//
// Initializing the maybe attribute creates an absolute attribute internally which includes the
// possible namespaced names of the attribute. In this example, let's assume we are in namespace
// 'ns', then the maybe is either one of the following variable names:
//
//	possible variables names -- ns.a, a
//
// 2. Adding a qualifier to the maybe means that the variable name could be a longer qualified
// name, or a field selection on one of the possible variable names produced earlier:
//
//	mb.AddQualifier("b")
//
//	possible variables names -- ns.a.b, a.b
//	possible field selection -- ns.a['b'], a['b']
//
// If none of the attributes within the maybe resolves a value, the result is an error.
func (a *maybeAttribute) AddQualifier(qual Qualifier) (Attribute, error) {
	str := ""
	isStr := false
	cq, isConst := qual.(ConstantQualifier)
	if isConst {
		str, isStr = cq.Value().Value().(string)
	}
	var augmentedNames []string
	// First add the qualifier to all existing attributes in the oneof.
	for _, attr := range a.attrs {
		if isStr && len(attr.Qualifiers()) == 0 {
			candidateVars := attr.CandidateVariableNames()
			augmentedNames = make([]string, len(candidateVars))
			for i, name := range candidateVars {
				augmentedNames[i] = fmt.Sprintf("%s.%s", name, str)
			}
		}
		_, err := attr.AddQualifier(qual)
		if err != nil {
			return nil, err
		}
	}
	// Next, ensure the most specific variable / type reference is searched first.
	if len(augmentedNames) != 0 {
		a.attrs = append([]NamespacedAttribute{a.fac.AbsoluteAttribute(qual.ID(), augmentedNames...)}, a.attrs...)
	}
	return a, nil
}

// Qualify is an implementation of the Qualifier interface method.
func (a *maybeAttribute) Qualify(vars Activation, obj any) (any, error) {
	return attrQualify(a.fac, vars, obj, a)
}

// QualifyIfPresent is an implementation of the Qualifier interface method.
func (a *maybeAttribute) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	return attrQualifyIfPresent(a.fac, vars, obj, a, presenceOnly)
}

// Resolve follows the variable resolution rules to determine whether the attribute is a variable
// or a field selection.
func (a *maybeAttribute) Resolve(vars Activation) (any, error) {
	var maybeErr error
	for _, attr := range a.attrs {
		obj, err := attr.Resolve(vars)
		// Return an error if one is encountered.
		if err != nil {
			resErr, ok := err.(*resolutionError)
			if !ok {
				return nil, err
			}
			// If this was not a missing variable error, return it.
			if !resErr.isMissingAttribute() {
				return nil, err
			}
			// When the variable is missing in a maybe attribute we defer erroring.
			if maybeErr == nil {
				maybeErr = resErr
			}
			// Continue attempting to resolve possible variables.
			continue
		}
		return obj, nil
	}
	// Else, produce a no such attribute error.
	return nil, maybeErr
}

// String is an implementation of the Stringer interface method.
func (a *maybeAttribute) String() string {
	return fmt.Sprintf("id: %v, attributes: %v", a.id, a.attrs)
}

type relativeAttribute struct {
	id         int64
	operand    Interpretable
	qualifiers []Qualifier
	adapter    types.Adapter
	fac        AttributeFactory
}

// ID is an implementation of the Attribute interface method.
func (a *relativeAttribute) ID() int64 {
	qualCount := len(a.qualifiers)
	if qualCount == 0 {
		return a.id
	}
	return a.qualifiers[qualCount-1].ID()
}

// IsOptional returns trivially false for an attribute as the attribute represents a fully
// qualified variable name. If the attribute is used in an optional manner, then an attrQualifier
// is created and marks the attribute as optional.
func (a *relativeAttribute) IsOptional() bool {
	return false
}

// AddQualifier implements the Attribute interface method.
func (a *relativeAttribute) AddQualifier(qual Qualifier) (Attribute, error) {
	a.qualifiers = append(a.qualifiers, qual)
	return a, nil
}

// Qualify is an implementation of the Qualifier interface method.
func (a *relativeAttribute) Qualify(vars Activation, obj any) (any, error) {
	return attrQualify(a.fac, vars, obj, a)
}

// QualifyIfPresent is an implementation of the Qualifier interface method.
func (a *relativeAttribute) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	return attrQualifyIfPresent(a.fac, vars, obj, a, presenceOnly)
}

// Resolve expression value and qualifier relative to the expression result.
func (a *relativeAttribute) Resolve(vars Activation) (any, error) {
	// First, evaluate the operand.
	v := a.operand.Eval(vars)
	if types.IsError(v) {
		return nil, v.(*types.Err)
	}
	if types.IsUnknown(v) {
		return v, nil
	}
	obj, isOpt, err := applyQualifiers(vars, v, a.qualifiers)
	if err != nil {
		return nil, err
	}
	if isOpt {
		val := a.adapter.NativeToValue(obj)
		if types.IsUnknown(val) {
			return val, nil
		}
		return types.OptionalOf(val), nil
	}
	return obj, nil
}

// String is an implementation of the Stringer interface method.
func (a *relativeAttribute) String() string {
	return fmt.Sprintf("id: %v, operand: %v", a.id, a.operand)
}

func newQualifier(adapter types.Adapter, id int64, v any, opt bool) (Qualifier, error) {
	var qual Qualifier
	switch val := v.(type) {
	case Attribute:
		// Note, attributes are initially identified as non-optional since they represent a top-level
		// field access; however, when used as a relative qualifier, e.g. a[?b.c], then an attrQualifier
		// is created which intercepts the IsOptional check for the attribute in order to return the
		// correct result.
		return &attrQualifier{
			id:        id,
			Attribute: val,
			optional:  opt,
		}, nil
	case string:
		qual = &stringQualifier{
			id:       id,
			value:    val,
			celValue: types.String(val),
			adapter:  adapter,
			optional: opt,
		}
	case int:
		qual = &intQualifier{
			id: id, value: int64(val), celValue: types.Int(val), adapter: adapter, optional: opt,
		}
	case int32:
		qual = &intQualifier{
			id: id, value: int64(val), celValue: types.Int(val), adapter: adapter, optional: opt,
		}
	case int64:
		qual = &intQualifier{
			id: id, value: val, celValue: types.Int(val), adapter: adapter, optional: opt,
		}
	case uint:
		qual = &uintQualifier{
			id: id, value: uint64(val), celValue: types.Uint(val), adapter: adapter, optional: opt,
		}
	case uint32:
		qual = &uintQualifier{
			id: id, value: uint64(val), celValue: types.Uint(val), adapter: adapter, optional: opt,
		}
	case uint64:
		qual = &uintQualifier{
			id: id, value: val, celValue: types.Uint(val), adapter: adapter, optional: opt,
		}
	case bool:
		qual = &boolQualifier{
			id: id, value: val, celValue: types.Bool(val), adapter: adapter, optional: opt,
		}
	case float32:
		qual = &doubleQualifier{
			id:       id,
			value:    float64(val),
			celValue: types.Double(val),
			adapter:  adapter,
			optional: opt,
		}
	case float64:
		qual = &doubleQualifier{
			id: id, value: val, celValue: types.Double(val), adapter: adapter, optional: opt,
		}
	case types.String:
		qual = &stringQualifier{
			id: id, value: string(val), celValue: val, adapter: adapter, optional: opt,
		}
	case types.Int:
		qual = &intQualifier{
			id: id, value: int64(val), celValue: val, adapter: adapter, optional: opt,
		}
	case types.Uint:
		qual = &uintQualifier{
			id: id, value: uint64(val), celValue: val, adapter: adapter, optional: opt,
		}
	case types.Bool:
		qual = &boolQualifier{
			id: id, value: bool(val), celValue: val, adapter: adapter, optional: opt,
		}
	case types.Double:
		qual = &doubleQualifier{
			id: id, value: float64(val), celValue: val, adapter: adapter, optional: opt,
		}
	case *types.Unknown:
		qual = &unknownQualifier{id: id, value: val}
	default:
		if q, ok := v.(Qualifier); ok {
			return q, nil
		}
		return nil, fmt.Errorf("invalid qualifier type: %T", v)
	}
	return qual, nil
}

type attrQualifier struct {
	id int64
	Attribute
	optional bool
}

// ID implements the Qualifier interface method and returns the qualification instruction id
// rather than the attribute id.
func (q *attrQualifier) ID() int64 {
	return q.id
}

// IsOptional implements the Qualifier interface method.
func (q *attrQualifier) IsOptional() bool {
	return q.optional
}

type stringQualifier struct {
	id       int64
	value    string
	celValue ref.Val
	adapter  types.Adapter
	optional bool
}

// ID is an implementation of the Qualifier interface method.
func (q *stringQualifier) ID() int64 {
	return q.id
}

// IsOptional implements the Qualifier interface method.
func (q *stringQualifier) IsOptional() bool {
	return q.optional
}

// Qualify implements the Qualifier interface method.
func (q *stringQualifier) Qualify(vars Activation, obj any) (any, error) {
	val, _, err := q.qualifyInternal(vars, obj, false, false)
	return val, err
}

// QualifyIfPresent is an implementation of the Qualifier interface method.
func (q *stringQualifier) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	return q.qualifyInternal(vars, obj, true, presenceOnly)
}

func (q *stringQualifier) qualifyInternal(vars Activation, obj any, presenceTest, presenceOnly bool) (any, bool, error) {
	s := q.value
	switch o := obj.(type) {
	case map[string]any:
		obj, isKey := o[s]
		if isKey {
			return obj, true, nil
		}
	case map[string]string:
		obj, isKey := o[s]
		if isKey {
			return obj, true, nil
		}
	case map[string]int:
		obj, isKey := o[s]
		if isKey {
			return obj, true, nil
		}
	case map[string]int32:
		obj, isKey := o[s]
		if isKey {
			return obj, true, nil
		}
	case map[string]int64:
		obj, isKey := o[s]
		if isKey {
			return obj, true, nil
		}
	case map[string]uint:
		obj, isKey := o[s]
		if isKey {
			return obj, true, nil
		}
	case map[string]uint32:
		obj, isKey := o[s]
		if isKey {
			return obj, true, nil
		}
	case map[string]uint64:
		obj, isKey := o[s]
		if isKey {
			return obj, true, nil
		}
	case map[string]float32:
		obj, isKey := o[s]
		if isKey {
			return obj, true, nil
		}
	case map[string]float64:
		obj, isKey := o[s]
		if isKey {
			return obj, true, nil
		}
	case map[string]bool:
		obj, isKey := o[s]
		if isKey {
			return obj, true, nil
		}
	default:
		return refQualify(q.adapter, obj, q.celValue, presenceTest, presenceOnly)
	}
	if presenceTest {
		return nil, false, nil
	}
	return nil, false, missingKey(q.celValue)
}

// Value implements the ConstantQualifier interface
func (q *stringQualifier) Value() ref.Val {
	return q.celValue
}

type intQualifier struct {
	id       int64
	value    int64
	celValue ref.Val
	adapter  types.Adapter
	optional bool
}

// ID is an implementation of the Qualifier interface method.
func (q *intQualifier) ID() int64 {
	return q.id
}

// IsOptional implements the Qualifier interface method.
func (q *intQualifier) IsOptional() bool {
	return q.optional
}

// Qualify implements the Qualifier interface method.
func (q *intQualifier) Qualify(vars Activation, obj any) (any, error) {
	val, _, err := q.qualifyInternal(vars, obj, false, false)
	return val, err
}

// QualifyIfPresent is an implementation of the Qualifier interface method.
func (q *intQualifier) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	return q.qualifyInternal(vars, obj, true, presenceOnly)
}

func (q *intQualifier) qualifyInternal(vars Activation, obj any, presenceTest, presenceOnly bool) (any, bool, error) {
	i := q.value
	var isMap bool
	switch o := obj.(type) {
	// The specialized map types supported by an int qualifier are considerably fewer than the set
	// of specialized map types supported by string qualifiers since they are less frequently used
	// than string-based map keys. Additional specializations may be added in the future if
	// desired.
	case map[int]any:
		isMap = true
		obj, isKey := o[int(i)]
		if isKey {
			return obj, true, nil
		}
	case map[int32]any:
		isMap = true
		obj, isKey := o[int32(i)]
		if isKey {
			return obj, true, nil
		}
	case map[int64]any:
		isMap = true
		obj, isKey := o[i]
		if isKey {
			return obj, true, nil
		}
	case []any:
		isIndex := i >= 0 && i < int64(len(o))
		if isIndex {
			return o[i], true, nil
		}
	case []string:
		isIndex := i >= 0 && i < int64(len(o))
		if isIndex {
			return o[i], true, nil
		}
	case []int:
		isIndex := i >= 0 && i < int64(len(o))
		if isIndex {
			return o[i], true, nil
		}
	case []int32:
		isIndex := i >= 0 && i < int64(len(o))
		if isIndex {
			return o[i], true, nil
		}
	case []int64:
		isIndex := i >= 0 && i < int64(len(o))
		if isIndex {
			return o[i], true, nil
		}
	case []uint:
		isIndex := i >= 0 && i < int64(len(o))
		if isIndex {
			return o[i], true, nil
		}
	case []uint32:
		isIndex := i >= 0 && i < int64(len(o))
		if isIndex {
			return o[i], true, nil
		}
	case []uint64:
		isIndex := i >= 0 && i < int64(len(o))
		if isIndex {
			return o[i], true, nil
		}
	case []float32:
		isIndex := i >= 0 && i < int64(len(o))
		if isIndex {
			return o[i], true, nil
		}
	case []float64:
		isIndex := i >= 0 && i < int64(len(o))
		if isIndex {
			return o[i], true, nil
		}
	case []bool:
		isIndex := i >= 0 && i < int64(len(o))
		if isIndex {
			return o[i], true, nil
		}
	default:
		return refQualify(q.adapter, obj, q.celValue, presenceTest, presenceOnly)
	}
	if presenceTest {
		return nil, false, nil
	}
	if isMap {
		return nil, false, missingKey(q.celValue)
	}
	return nil, false, missingIndex(q.celValue)
}

// Value implements the ConstantQualifier interface
func (q *intQualifier) Value() ref.Val {
	return q.celValue
}

type uintQualifier struct {
	id       int64
	value    uint64
	celValue ref.Val
	adapter  types.Adapter
	optional bool
}

// ID is an implementation of the Qualifier interface method.
func (q *uintQualifier) ID() int64 {
	return q.id
}

// IsOptional implements the Qualifier interface method.
func (q *uintQualifier) IsOptional() bool {
	return q.optional
}

// Qualify implements the Qualifier interface method.
func (q *uintQualifier) Qualify(vars Activation, obj any) (any, error) {
	val, _, err := q.qualifyInternal(vars, obj, false, false)
	return val, err
}

// QualifyIfPresent is an implementation of the Qualifier interface method.
func (q *uintQualifier) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	return q.qualifyInternal(vars, obj, true, presenceOnly)
}

func (q *uintQualifier) qualifyInternal(vars Activation, obj any, presenceTest, presenceOnly bool) (any, bool, error) {
	u := q.value
	switch o := obj.(type) {
	// The specialized map types supported by a uint qualifier are considerably fewer than the set
	// of specialized map types supported by string qualifiers since they are less frequently used
	// than string-based map keys. Additional specializations may be added in the future if
	// desired.
	case map[uint]any:
		obj, isKey := o[uint(u)]
		if isKey {
			return obj, true, nil
		}
	case map[uint32]any:
		obj, isKey := o[uint32(u)]
		if isKey {
			return obj, true, nil
		}
	case map[uint64]any:
		obj, isKey := o[u]
		if isKey {
			return obj, true, nil
		}
	default:
		return refQualify(q.adapter, obj, q.celValue, presenceTest, presenceOnly)
	}
	if presenceTest {
		return nil, false, nil
	}
	return nil, false, missingKey(q.celValue)
}

// Value implements the ConstantQualifier interface
func (q *uintQualifier) Value() ref.Val {
	return q.celValue
}

type boolQualifier struct {
	id       int64
	value    bool
	celValue ref.Val
	adapter  types.Adapter
	optional bool
}

// ID is an implementation of the Qualifier interface method.
func (q *boolQualifier) ID() int64 {
	return q.id
}

// IsOptional implements the Qualifier interface method.
func (q *boolQualifier) IsOptional() bool {
	return q.optional
}

// Qualify implements the Qualifier interface method.
func (q *boolQualifier) Qualify(vars Activation, obj any) (any, error) {
	val, _, err := q.qualifyInternal(vars, obj, false, false)
	return val, err
}

// QualifyIfPresent is an implementation of the Qualifier interface method.
func (q *boolQualifier) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	return q.qualifyInternal(vars, obj, true, presenceOnly)
}

func (q *boolQualifier) qualifyInternal(vars Activation, obj any, presenceTest, presenceOnly bool) (any, bool, error) {
	b := q.value
	switch o := obj.(type) {
	case map[bool]any:
		obj, isKey := o[b]
		if isKey {
			return obj, true, nil
		}
	default:
		return refQualify(q.adapter, obj, q.celValue, presenceTest, presenceOnly)
	}
	if presenceTest {
		return nil, false, nil
	}
	return nil, false, missingKey(q.celValue)
}

// Value implements the ConstantQualifier interface
func (q *boolQualifier) Value() ref.Val {
	return q.celValue
}

// fieldQualifier indicates that the qualification is a well-defined field with a known
// field type. When the field type is known this can be used to improve the speed and
// efficiency of field resolution.
type fieldQualifier struct {
	id        int64
	Name      string
	FieldType *types.FieldType
	adapter   types.Adapter
	optional  bool
}

// ID is an implementation of the Qualifier interface method.
func (q *fieldQualifier) ID() int64 {
	return q.id
}

// IsOptional implements the Qualifier interface method.
func (q *fieldQualifier) IsOptional() bool {
	return q.optional
}

// Qualify implements the Qualifier interface method.
func (q *fieldQualifier) Qualify(vars Activation, obj any) (any, error) {
	if rv, ok := obj.(ref.Val); ok {
		obj = rv.Value()
	}
	val, err := q.FieldType.GetFrom(obj)
	if err != nil {
		return nil, err
	}
	return val, nil
}

// QualifyIfPresent is an implementation of the Qualifier interface method.
func (q *fieldQualifier) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	if rv, ok := obj.(ref.Val); ok {
		obj = rv.Value()
	}
	if !q.FieldType.IsSet(obj) {
		return nil, false, nil
	}
	if presenceOnly {
		return nil, true, nil
	}
	val, err := q.FieldType.GetFrom(obj)
	if err != nil {
		return nil, false, err
	}
	return val, true, nil
}

// Value implements the ConstantQualifier interface
func (q *fieldQualifier) Value() ref.Val {
	return types.String(q.Name)
}

// doubleQualifier qualifies a CEL object, map, or list using a double value.
//
// This qualifier is used for working with dynamic data like JSON or protobuf.Any where the value
// type may not be known ahead of time and may not conform to the standard types supported as valid
// protobuf map key types.
type doubleQualifier struct {
	id       int64
	value    float64
	celValue ref.Val
	adapter  types.Adapter
	optional bool
}

// ID is an implementation of the Qualifier interface method.
func (q *doubleQualifier) ID() int64 {
	return q.id
}

// IsOptional implements the Qualifier interface method.
func (q *doubleQualifier) IsOptional() bool {
	return q.optional
}

// Qualify implements the Qualifier interface method.
func (q *doubleQualifier) Qualify(vars Activation, obj any) (any, error) {
	val, _, err := q.qualifyInternal(vars, obj, false, false)
	return val, err
}

func (q *doubleQualifier) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	return q.qualifyInternal(vars, obj, true, presenceOnly)
}

func (q *doubleQualifier) qualifyInternal(vars Activation, obj any, presenceTest, presenceOnly bool) (any, bool, error) {
	return refQualify(q.adapter, obj, q.celValue, presenceTest, presenceOnly)
}

// Value implements the ConstantQualifier interface
func (q *doubleQualifier) Value() ref.Val {
	return q.celValue
}

// unknownQualifier is a simple qualifier which always returns a preconfigured set of unknown values
// for any value subject to qualification. This is consistent with CEL's unknown handling elsewhere.
type unknownQualifier struct {
	id    int64
	value *types.Unknown
}

// ID is an implementation of the Qualifier interface method.
func (q *unknownQualifier) ID() int64 {
	return q.id
}

// IsOptional returns trivially false as an the unknown value is always returned.
func (q *unknownQualifier) IsOptional() bool {
	return false
}

// Qualify returns the unknown value associated with this qualifier.
func (q *unknownQualifier) Qualify(vars Activation, obj any) (any, error) {
	return q.value, nil
}

// QualifyIfPresent is an implementation of the Qualifier interface method.
func (q *unknownQualifier) QualifyIfPresent(vars Activation, obj any, presenceOnly bool) (any, bool, error) {
	return q.value, true, nil
}

// Value implements the ConstantQualifier interface
func (q *unknownQualifier) Value() ref.Val {
	return q.value
}

func applyQualifiers(vars Activation, obj any, qualifiers []Qualifier) (any, bool, error) {
	optObj, isOpt := obj.(*types.Optional)
	if isOpt {
		if !optObj.HasValue() {
			return optObj, false, nil
		}
		obj = optObj.GetValue().Value()
	}

	var err error
	for _, qual := range qualifiers {
		var qualObj any
		isOpt = isOpt || qual.IsOptional()
		if isOpt {
			var present bool
			qualObj, present, err = qual.QualifyIfPresent(vars, obj, false)
			if err != nil {
				return nil, false, err
			}
			if !present {
				// We return optional none here with a presence of 'false' as the layers
				// above will attempt to call types.OptionalOf() on a present value if any
				// of the qualifiers is optional.
				return types.OptionalNone, false, nil
			}
		} else {
			qualObj, err = qual.Qualify(vars, obj)
			if err != nil {
				return nil, false, err
			}
		}
		obj = qualObj
	}
	return obj, isOpt, nil
}

// attrQualify performs a qualification using the result of an attribute evaluation.
func attrQualify(fac AttributeFactory, vars Activation, obj any, qualAttr Attribute) (any, error) {
	val, err := qualAttr.Resolve(vars)
	if err != nil {
		return nil, err
	}
	qual, err := fac.NewQualifier(nil, qualAttr.ID(), val, qualAttr.IsOptional())
	if err != nil {
		return nil, err
	}
	return qual.Qualify(vars, obj)
}

// attrQualifyIfPresent conditionally performs the qualification of the result of attribute is present
// on the target object.
func attrQualifyIfPresent(fac AttributeFactory, vars Activation, obj any, qualAttr Attribute,
	presenceOnly bool) (any, bool, error) {
	val, err := qualAttr.Resolve(vars)
	if err != nil {
		return nil, false, err
	}
	qual, err := fac.NewQualifier(nil, qualAttr.ID(), val, qualAttr.IsOptional())
	if err != nil {
		return nil, false, err
	}
	return qual.QualifyIfPresent(vars, obj, presenceOnly)
}

// refQualify attempts to convert the value to a CEL value and then uses reflection methods to try and
// apply the qualifier with the option to presence test field accesses before retrieving field values.
func refQualify(adapter types.Adapter, obj any, idx ref.Val, presenceTest, presenceOnly bool) (ref.Val, bool, error) {
	celVal := adapter.NativeToValue(obj)
	switch v := celVal.(type) {
	case *types.Unknown:
		return v, true, nil
	case *types.Err:
		return nil, false, v
	case traits.Mapper:
		val, found := v.Find(idx)
		// If the index is of the wrong type for the map, then it is possible
		// for the Find call to produce an error.
		if types.IsError(val) {
			return nil, false, val.(*types.Err)
		}
		if found {
			return val, true, nil
		}
		if presenceTest {
			return nil, false, nil
		}
		return nil, false, missingKey(idx)
	case traits.Lister:
		// If the index argument is not a valid numeric type, then it is possible
		// for the index operation to produce an error.
		i, err := types.IndexOrError(idx)
		if err != nil {
			return nil, false, err
		}
		celIndex := types.Int(i)
		if i >= 0 && celIndex < v.Size().(types.Int) {
			return v.Get(idx), true, nil
		}
		if presenceTest {
			return nil, false, nil
		}
		return nil, false, missingIndex(idx)
	case traits.Indexer:
		if presenceTest {
			ft, ok := v.(traits.FieldTester)
			if ok {
				presence := ft.IsSet(idx)
				if types.IsError(presence) {
					return nil, false, presence.(*types.Err)
				}
				// If not found or presence only test, then return.
				// Otherwise, if found, obtain the value later on.
				if presenceOnly || presence == types.False {
					return nil, presence == types.True, nil
				}
			}
		}
		val := v.Get(idx)
		if types.IsError(val) {
			return nil, false, val.(*types.Err)
		}
		return val, true, nil
	default:
		if presenceTest {
			return nil, false, nil
		}
		return nil, false, missingKey(idx)
	}
}

// resolutionError is a custom error type which encodes the different error states which may
// occur during attribute resolution.
type resolutionError struct {
	missingAttribute string
	missingIndex     ref.Val
	missingKey       ref.Val
}

func (e *resolutionError) isMissingAttribute() bool {
	return e.missingAttribute != ""
}

func missingIndex(missing ref.Val) *resolutionError {
	return &resolutionError{
		missingIndex: missing,
	}
}

func missingKey(missing ref.Val) *resolutionError {
	return &resolutionError{
		missingKey: missing,
	}
}

func missingAttribute(attr string) *resolutionError {
	return &resolutionError{
		missingAttribute: attr,
	}
}

// Error implements the error interface method.
func (e *resolutionError) Error() string {
	if e.missingKey != nil {
		return fmt.Sprintf("no such key: %v", e.missingKey)
	}
	if e.missingIndex != nil {
		return fmt.Sprintf("index out of bounds: %v", e.missingIndex)
	}
	if e.missingAttribute != "" {
		return fmt.Sprintf("no such attribute(s): %s", e.missingAttribute)
	}
	return "invalid attribute"
}

// Is implements the errors.Is() method used by more recent versions of Go.
func (e *resolutionError) Is(err error) bool {
	return err.Error() == e.Error()
}
