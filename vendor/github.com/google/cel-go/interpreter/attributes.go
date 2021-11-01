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
	"errors"
	"fmt"
	"math"

	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
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
	NewQualifier(objType *exprpb.Type, qualID int64, val interface{}) (Qualifier, error)
}

// Qualifier marker interface for designating different qualifier values and where they appear
// within field selections and index call expressions (`_[_]`).
type Qualifier interface {
	// ID where the qualifier appears within an expression.
	ID() int64

	// Qualify performs a qualification, e.g. field selection, on the input object and returns
	// the value or error that results.
	Qualify(vars Activation, obj interface{}) (interface{}, error)
}

// ConstantQualifier interface embeds the Qualifier interface and provides an option to inspect the
// qualifier's constant value.
//
// Non-constant qualifiers are of Attribute type.
type ConstantQualifier interface {
	Qualifier

	Value() ref.Val
}

// Attribute values are a variable or value with an optional set of qualifiers, such as field, key,
// or index accesses.
type Attribute interface {
	Qualifier

	// AddQualifier adds a qualifier on the Attribute or error if the qualification is not a valid
	// qualifier type.
	AddQualifier(Qualifier) (Attribute, error)

	// Resolve returns the value of the Attribute given the current Activation.
	Resolve(Activation) (interface{}, error)
}

// NamespacedAttribute values are a variable within a namespace, and an optional set of qualifiers
// such as field, key, or index accesses.
type NamespacedAttribute interface {
	Attribute

	// CandidateVariableNames returns the possible namespaced variable names for this Attribute in
	// the CEL namespace resolution order.
	CandidateVariableNames() []string

	// Qualifiers returns the list of qualifiers associated with the Attribute.s
	Qualifiers() []Qualifier

	// TryResolve attempts to return the value of the attribute given the current Activation.
	// If an error is encountered during attribute resolution, it will be returned immediately.
	// If the attribute cannot be resolved within the Activation, the result must be: `nil`,
	// `false`, `nil`.
	TryResolve(Activation) (interface{}, bool, error)
}

// NewAttributeFactory returns a default AttributeFactory which is produces Attribute values
// capable of resolving types by simple names and qualify the values using the supported qualifier
// types: bool, int, string, and uint.
func NewAttributeFactory(cont *containers.Container,
	a ref.TypeAdapter,
	p ref.TypeProvider) AttributeFactory {
	return &attrFactory{
		container: cont,
		adapter:   a,
		provider:  p,
	}
}

type attrFactory struct {
	container *containers.Container
	adapter   ref.TypeAdapter
	provider  ref.TypeProvider
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
func (r *attrFactory) NewQualifier(objType *exprpb.Type,
	qualID int64,
	val interface{}) (Qualifier, error) {
	// Before creating a new qualifier check to see if this is a protobuf message field access.
	// If so, use the precomputed GetFrom qualification method rather than the standard
	// stringQualifier.
	str, isStr := val.(string)
	if isStr && objType != nil && objType.GetMessageType() != "" {
		ft, found := r.provider.FindFieldType(objType.GetMessageType(), str)
		if found && ft.IsSet != nil && ft.GetFrom != nil {
			return &fieldQualifier{
				id:        qualID,
				Name:      str,
				FieldType: ft,
				adapter:   r.adapter,
			}, nil
		}
	}
	return newQualifier(r.adapter, qualID, val)
}

type absoluteAttribute struct {
	id int64
	// namespaceNames represent the names the variable could have based on declared container
	// (package) of the expression.
	namespaceNames []string
	qualifiers     []Qualifier
	adapter        ref.TypeAdapter
	provider       ref.TypeProvider
	fac            AttributeFactory
}

// ID implements the Attribute interface method.
func (a *absoluteAttribute) ID() int64 {
	return a.id
}

// Cost implements the Coster interface method.
func (a *absoluteAttribute) Cost() (min, max int64) {
	for _, q := range a.qualifiers {
		minQ, maxQ := estimateCost(q)
		min += minQ
		max += maxQ
	}
	min++ // For object retrieval.
	max++
	return
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
func (a *absoluteAttribute) Qualify(vars Activation, obj interface{}) (interface{}, error) {
	val, err := a.Resolve(vars)
	if err != nil {
		return nil, err
	}
	unk, isUnk := val.(types.Unknown)
	if isUnk {
		return unk, nil
	}
	qual, err := a.fac.NewQualifier(nil, a.id, val)
	if err != nil {
		return nil, err
	}
	return qual.Qualify(vars, obj)
}

// Resolve returns the resolved Attribute value given the Activation, or error if the Attribute
// variable is not found, or if its Qualifiers cannot be applied successfully.
func (a *absoluteAttribute) Resolve(vars Activation) (interface{}, error) {
	obj, found, err := a.TryResolve(vars)
	if err != nil {
		return nil, err
	}
	if found {
		return obj, nil
	}
	return nil, fmt.Errorf("no such attribute: %v", a)
}

// String implements the Stringer interface method.
func (a *absoluteAttribute) String() string {
	return fmt.Sprintf("id: %v, names: %v", a.id, a.namespaceNames)
}

// TryResolve iterates through the namespaced variable names until one is found within the
// Activation or TypeProvider.
//
// If the variable name cannot be found as an Activation variable or in the TypeProvider as
// a type, then the result is `nil`, `false`, `nil` per the interface requirement.
func (a *absoluteAttribute) TryResolve(vars Activation) (interface{}, bool, error) {
	for _, nm := range a.namespaceNames {
		// If the variable is found, process it. Otherwise, wait until the checks to
		// determine whether the type is unknown before returning.
		op, found := vars.ResolveName(nm)
		if found {
			var err error
			for _, qual := range a.qualifiers {
				op, err = qual.Qualify(vars, op)
				if err != nil {
					return nil, true, err
				}
			}
			return op, true, nil
		}
		// Attempt to resolve the qualified type name if the name is not a variable identifier.
		typ, found := a.provider.FindIdent(nm)
		if found {
			if len(a.qualifiers) == 0 {
				return typ, true, nil
			}
			return nil, true, fmt.Errorf("no such attribute: %v", typ)
		}
	}
	return nil, false, nil
}

type conditionalAttribute struct {
	id      int64
	expr    Interpretable
	truthy  Attribute
	falsy   Attribute
	adapter ref.TypeAdapter
	fac     AttributeFactory
}

// ID is an implementation of the Attribute interface method.
func (a *conditionalAttribute) ID() int64 {
	return a.id
}

// Cost provides the heuristic cost of a ternary operation <expr> ? <t> : <f>.
// The cost is computed as cost(expr) plus the min/max costs of evaluating either
// `t` or `f`.
func (a *conditionalAttribute) Cost() (min, max int64) {
	tMin, tMax := estimateCost(a.truthy)
	fMin, fMax := estimateCost(a.falsy)
	eMin, eMax := estimateCost(a.expr)
	return eMin + findMin(tMin, fMin), eMax + findMax(tMax, fMax)
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
func (a *conditionalAttribute) Qualify(vars Activation, obj interface{}) (interface{}, error) {
	val, err := a.Resolve(vars)
	if err != nil {
		return nil, err
	}
	unk, isUnk := val.(types.Unknown)
	if isUnk {
		return unk, nil
	}
	qual, err := a.fac.NewQualifier(nil, a.id, val)
	if err != nil {
		return nil, err
	}
	return qual.Qualify(vars, obj)
}

// Resolve evaluates the condition, and then resolves the truthy or falsy branch accordingly.
func (a *conditionalAttribute) Resolve(vars Activation) (interface{}, error) {
	val := a.expr.Eval(vars)
	if types.IsError(val) {
		return nil, val.(*types.Err)
	}
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
	adapter  ref.TypeAdapter
	provider ref.TypeProvider
	fac      AttributeFactory
}

// ID is an implementation of the Attribute interface method.
func (a *maybeAttribute) ID() int64 {
	return a.id
}

// Cost implements the Coster interface method. The min cost is computed as the minimal cost among
// all the possible attributes, the max cost ditto.
func (a *maybeAttribute) Cost() (min, max int64) {
	min, max = math.MaxInt64, 0
	for _, a := range a.attrs {
		minA, maxA := estimateCost(a)
		min = findMin(min, minA)
		max = findMax(max, maxA)
	}
	return
}

func findMin(x, y int64) int64 {
	if x < y {
		return x
	}
	return y
}

func findMax(x, y int64) int64 {
	if x > y {
		return x
	}
	return y
}

// AddQualifier adds a qualifier to each possible attribute variant, and also creates
// a new namespaced variable from the qualified value.
//
// The algorithm for building the maybe attribute is as follows:
//
// 1. Create a maybe attribute from a simple identifier when it occurs in a parsed-only expression
//
//    mb = MaybeAttribute(<id>, "a")
//
//    Initializing the maybe attribute creates an absolute attribute internally which includes the
//    possible namespaced names of the attribute. In this example, let's assume we are in namespace
//    'ns', then the maybe is either one of the following variable names:
//
//    possible variables names -- ns.a, a
//
// 2. Adding a qualifier to the maybe means that the variable name could be a longer qualified
//    name, or a field selection on one of the possible variable names produced earlier:
//
//    mb.AddQualifier("b")
//
//    possible variables names -- ns.a.b, a.b
//    possible field selection -- ns.a['b'], a['b']
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
	a.attrs = append([]NamespacedAttribute{
		a.fac.AbsoluteAttribute(qual.ID(), augmentedNames...),
	}, a.attrs...)
	return a, nil
}

// Qualify is an implementation of the Qualifier interface method.
func (a *maybeAttribute) Qualify(vars Activation, obj interface{}) (interface{}, error) {
	val, err := a.Resolve(vars)
	if err != nil {
		return nil, err
	}
	unk, isUnk := val.(types.Unknown)
	if isUnk {
		return unk, nil
	}
	qual, err := a.fac.NewQualifier(nil, a.id, val)
	if err != nil {
		return nil, err
	}
	return qual.Qualify(vars, obj)
}

// Resolve follows the variable resolution rules to determine whether the attribute is a variable
// or a field selection.
func (a *maybeAttribute) Resolve(vars Activation) (interface{}, error) {
	for _, attr := range a.attrs {
		obj, found, err := attr.TryResolve(vars)
		// Return an error if one is encountered.
		if err != nil {
			return nil, err
		}
		// If the object was found, return it.
		if found {
			return obj, nil
		}
	}
	// Else, produce a no such attribute error.
	return nil, fmt.Errorf("no such attribute: %v", a)
}

// String is an implementation of the Stringer interface method.
func (a *maybeAttribute) String() string {
	return fmt.Sprintf("id: %v, attributes: %v", a.id, a.attrs)
}

type relativeAttribute struct {
	id         int64
	operand    Interpretable
	qualifiers []Qualifier
	adapter    ref.TypeAdapter
	fac        AttributeFactory
}

// ID is an implementation of the Attribute interface method.
func (a *relativeAttribute) ID() int64 {
	return a.id
}

// Cost implements the Coster interface method.
func (a *relativeAttribute) Cost() (min, max int64) {
	min, max = estimateCost(a.operand)
	for _, qual := range a.qualifiers {
		minQ, maxQ := estimateCost(qual)
		min += minQ
		max += maxQ
	}
	return
}

// AddQualifier implements the Attribute interface method.
func (a *relativeAttribute) AddQualifier(qual Qualifier) (Attribute, error) {
	a.qualifiers = append(a.qualifiers, qual)
	return a, nil
}

// Qualify is an implementation of the Qualifier interface method.
func (a *relativeAttribute) Qualify(vars Activation, obj interface{}) (interface{}, error) {
	val, err := a.Resolve(vars)
	if err != nil {
		return nil, err
	}
	unk, isUnk := val.(types.Unknown)
	if isUnk {
		return unk, nil
	}
	qual, err := a.fac.NewQualifier(nil, a.id, val)
	if err != nil {
		return nil, err
	}
	return qual.Qualify(vars, obj)
}

// Resolve expression value and qualifier relative to the expression result.
func (a *relativeAttribute) Resolve(vars Activation) (interface{}, error) {
	// First, evaluate the operand.
	v := a.operand.Eval(vars)
	if types.IsError(v) {
		return nil, v.(*types.Err)
	}
	if types.IsUnknown(v) {
		return v, nil
	}
	// Next, qualify it. Qualification handles unkonwns as well, so there's no need to recheck.
	var err error
	var obj interface{} = v
	for _, qual := range a.qualifiers {
		obj, err = qual.Qualify(vars, obj)
		if err != nil {
			return nil, err
		}
	}
	return obj, nil
}

// String is an implementation of the Stringer interface method.
func (a *relativeAttribute) String() string {
	return fmt.Sprintf("id: %v, operand: %v", a.id, a.operand)
}

func newQualifier(adapter ref.TypeAdapter, id int64, v interface{}) (Qualifier, error) {
	var qual Qualifier
	switch val := v.(type) {
	case Attribute:
		return &attrQualifier{id: id, Attribute: val}, nil
	case string:
		qual = &stringQualifier{id: id, value: val, celValue: types.String(val), adapter: adapter}
	case int:
		qual = &intQualifier{id: id, value: int64(val), celValue: types.Int(val), adapter: adapter}
	case int32:
		qual = &intQualifier{id: id, value: int64(val), celValue: types.Int(val), adapter: adapter}
	case int64:
		qual = &intQualifier{id: id, value: val, celValue: types.Int(val), adapter: adapter}
	case uint:
		qual = &uintQualifier{id: id, value: uint64(val), celValue: types.Uint(val), adapter: adapter}
	case uint32:
		qual = &uintQualifier{id: id, value: uint64(val), celValue: types.Uint(val), adapter: adapter}
	case uint64:
		qual = &uintQualifier{id: id, value: val, celValue: types.Uint(val), adapter: adapter}
	case bool:
		qual = &boolQualifier{id: id, value: val, celValue: types.Bool(val), adapter: adapter}
	case types.String:
		qual = &stringQualifier{id: id, value: string(val), celValue: val, adapter: adapter}
	case types.Int:
		qual = &intQualifier{id: id, value: int64(val), celValue: val, adapter: adapter}
	case types.Uint:
		qual = &uintQualifier{id: id, value: uint64(val), celValue: val, adapter: adapter}
	case types.Bool:
		qual = &boolQualifier{id: id, value: bool(val), celValue: val, adapter: adapter}
	default:
		return nil, fmt.Errorf("invalid qualifier type: %T", v)
	}
	return qual, nil
}

type attrQualifier struct {
	id int64
	Attribute
}

func (q *attrQualifier) ID() int64 {
	return q.id
}

// Cost returns zero for constant field qualifiers
func (q *attrQualifier) Cost() (min, max int64) {
	return estimateCost(q.Attribute)
}

type stringQualifier struct {
	id       int64
	value    string
	celValue ref.Val
	adapter  ref.TypeAdapter
}

// ID is an implementation of the Qualifier interface method.
func (q *stringQualifier) ID() int64 {
	return q.id
}

// Qualify implements the Qualifier interface method.
func (q *stringQualifier) Qualify(vars Activation, obj interface{}) (interface{}, error) {
	s := q.value
	isMap := false
	isKey := false
	switch o := obj.(type) {
	case map[string]interface{}:
		isMap = true
		obj, isKey = o[s]
	case map[string]string:
		isMap = true
		obj, isKey = o[s]
	case map[string]int:
		isMap = true
		obj, isKey = o[s]
	case map[string]int32:
		isMap = true
		obj, isKey = o[s]
	case map[string]int64:
		isMap = true
		obj, isKey = o[s]
	case map[string]uint:
		isMap = true
		obj, isKey = o[s]
	case map[string]uint32:
		isMap = true
		obj, isKey = o[s]
	case map[string]uint64:
		isMap = true
		obj, isKey = o[s]
	case map[string]float32:
		isMap = true
		obj, isKey = o[s]
	case map[string]float64:
		isMap = true
		obj, isKey = o[s]
	case map[string]bool:
		isMap = true
		obj, isKey = o[s]
	case types.Unknown:
		return o, nil
	default:
		elem, err := refResolve(q.adapter, q.celValue, obj)
		if err != nil {
			return nil, err
		}
		if types.IsUnknown(elem) {
			return elem, nil
		}
		return elem, nil
	}
	if isMap && !isKey {
		return nil, fmt.Errorf("no such key: %v", s)
	}
	return obj, nil
}

// Value implements the ConstantQualifier interface
func (q *stringQualifier) Value() ref.Val {
	return q.celValue
}

// Cost returns zero for constant field qualifiers
func (q *stringQualifier) Cost() (min, max int64) {
	return 0, 0
}

type intQualifier struct {
	id       int64
	value    int64
	celValue ref.Val
	adapter  ref.TypeAdapter
}

// ID is an implementation of the Qualifier interface method.
func (q *intQualifier) ID() int64 {
	return q.id
}

// Qualify implements the Qualifier interface method.
func (q *intQualifier) Qualify(vars Activation, obj interface{}) (interface{}, error) {
	i := q.value
	isMap := false
	isKey := false
	isIndex := false
	switch o := obj.(type) {
	// The specialized map types supported by an int qualifier are considerably fewer than the set
	// of specialized map types supported by string qualifiers since they are less frequently used
	// than string-based map keys. Additional specializations may be added in the future if
	// desired.
	case map[int]interface{}:
		isMap = true
		obj, isKey = o[int(i)]
	case map[int32]interface{}:
		isMap = true
		obj, isKey = o[int32(i)]
	case map[int64]interface{}:
		isMap = true
		obj, isKey = o[i]
	case []interface{}:
		isIndex = i >= 0 && i < int64(len(o))
		if isIndex {
			obj = o[i]
		}
	case []string:
		isIndex = i >= 0 && i < int64(len(o))
		if isIndex {
			obj = o[i]
		}
	case []int:
		isIndex = i >= 0 && i < int64(len(o))
		if isIndex {
			obj = o[i]
		}
	case []int32:
		isIndex = i >= 0 && i < int64(len(o))
		if isIndex {
			obj = o[i]
		}
	case []int64:
		isIndex = i >= 0 && i < int64(len(o))
		if isIndex {
			obj = o[i]
		}
	case []uint:
		isIndex = i >= 0 && i < int64(len(o))
		if isIndex {
			obj = o[i]
		}
	case []uint32:
		isIndex = i >= 0 && i < int64(len(o))
		if isIndex {
			obj = o[i]
		}
	case []uint64:
		isIndex = i >= 0 && i < int64(len(o))
		if isIndex {
			obj = o[i]
		}
	case []float32:
		isIndex = i >= 0 && i < int64(len(o))
		if isIndex {
			obj = o[i]
		}
	case []float64:
		isIndex = i >= 0 && i < int64(len(o))
		if isIndex {
			obj = o[i]
		}
	case []bool:
		isIndex = i >= 0 && i < int64(len(o))
		if isIndex {
			obj = o[i]
		}
	case types.Unknown:
		return o, nil
	default:
		elem, err := refResolve(q.adapter, q.celValue, obj)
		if err != nil {
			return nil, err
		}
		if types.IsUnknown(elem) {
			return elem, nil
		}
		return elem, nil
	}
	if isMap && !isKey {
		return nil, fmt.Errorf("no such key: %v", i)
	}
	if !isMap && !isIndex {
		return nil, fmt.Errorf("index out of bounds: %v", i)
	}
	return obj, nil
}

// Value implements the ConstantQualifier interface
func (q *intQualifier) Value() ref.Val {
	return q.celValue
}

// Cost returns zero for constant field qualifiers
func (q *intQualifier) Cost() (min, max int64) {
	return 0, 0
}

type uintQualifier struct {
	id       int64
	value    uint64
	celValue ref.Val
	adapter  ref.TypeAdapter
}

// ID is an implementation of the Qualifier interface method.
func (q *uintQualifier) ID() int64 {
	return q.id
}

// Qualify implements the Qualifier interface method.
func (q *uintQualifier) Qualify(vars Activation, obj interface{}) (interface{}, error) {
	u := q.value
	isMap := false
	isKey := false
	switch o := obj.(type) {
	// The specialized map types supported by a uint qualifier are considerably fewer than the set
	// of specialized map types supported by string qualifiers since they are less frequently used
	// than string-based map keys. Additional specializations may be added in the future if
	// desired.
	case map[uint]interface{}:
		isMap = true
		obj, isKey = o[uint(u)]
	case map[uint32]interface{}:
		isMap = true
		obj, isKey = o[uint32(u)]
	case map[uint64]interface{}:
		isMap = true
		obj, isKey = o[u]
	case types.Unknown:
		return o, nil
	default:
		elem, err := refResolve(q.adapter, q.celValue, obj)
		if err != nil {
			return nil, err
		}
		if types.IsUnknown(elem) {
			return elem, nil
		}
		return elem, nil
	}
	if isMap && !isKey {
		return nil, fmt.Errorf("no such key: %v", u)
	}
	return obj, nil
}

// Value implements the ConstantQualifier interface
func (q *uintQualifier) Value() ref.Val {
	return q.celValue
}

// Cost returns zero for constant field qualifiers
func (q *uintQualifier) Cost() (min, max int64) {
	return 0, 0
}

type boolQualifier struct {
	id       int64
	value    bool
	celValue ref.Val
	adapter  ref.TypeAdapter
}

// ID is an implementation of the Qualifier interface method.
func (q *boolQualifier) ID() int64 {
	return q.id
}

// Qualify implements the Qualifier interface method.
func (q *boolQualifier) Qualify(vars Activation, obj interface{}) (interface{}, error) {
	b := q.value
	isKey := false
	switch o := obj.(type) {
	// The specialized map types supported by a bool qualifier are considerably fewer than the set
	// of specialized map types supported by string qualifiers since they are less frequently used
	// than string-based map keys. Additional specializations may be added in the future if
	// desired.
	case map[bool]interface{}:
		obj, isKey = o[b]
	case types.Unknown:
		return o, nil
	default:
		elem, err := refResolve(q.adapter, q.celValue, obj)
		if err != nil {
			return nil, err
		}
		if types.IsUnknown(elem) {
			return elem, nil
		}
		return elem, nil
	}
	if !isKey {
		return nil, fmt.Errorf("no such key: %v", b)
	}
	return obj, nil
}

// Value implements the ConstantQualifier interface
func (q *boolQualifier) Value() ref.Val {
	return q.celValue
}

// Cost returns zero for constant field qualifiers
func (q *boolQualifier) Cost() (min, max int64) {
	return 0, 0
}

// fieldQualifier indicates that the qualification is a well-defined field with a known
// field type. When the field type is known this can be used to improve the speed and
// efficiency of field resolution.
type fieldQualifier struct {
	id        int64
	Name      string
	FieldType *ref.FieldType
	adapter   ref.TypeAdapter
}

// ID is an implementation of the Qualifier interface method.
func (q *fieldQualifier) ID() int64 {
	return q.id
}

// Qualify implements the Qualifier interface method.
func (q *fieldQualifier) Qualify(vars Activation, obj interface{}) (interface{}, error) {
	if rv, ok := obj.(ref.Val); ok {
		obj = rv.Value()
	}
	return q.FieldType.GetFrom(obj)
}

// Value implements the ConstantQualifier interface
func (q *fieldQualifier) Value() ref.Val {
	return types.String(q.Name)
}

// Cost returns zero for constant field qualifiers
func (q *fieldQualifier) Cost() (min, max int64) {
	return 0, 0
}

// refResolve attempts to convert the value to a CEL value and then uses reflection methods
// to try and resolve the qualifier.
func refResolve(adapter ref.TypeAdapter, idx ref.Val, obj interface{}) (ref.Val, error) {
	celVal := adapter.NativeToValue(obj)
	mapper, isMapper := celVal.(traits.Mapper)
	if isMapper {
		elem, found := mapper.Find(idx)
		if !found {
			return nil, fmt.Errorf("no such key: %v", idx)
		}
		if types.IsError(elem) {
			return nil, elem.(*types.Err)
		}
		return elem, nil
	}
	indexer, isIndexer := celVal.(traits.Indexer)
	if isIndexer {
		elem := indexer.Get(idx)
		if types.IsError(elem) {
			return nil, elem.(*types.Err)
		}
		return elem, nil
	}
	if types.IsUnknown(celVal) {
		return celVal, nil
	}
	// TODO: If the types.Err value contains more than just an error message at some point in the
	// future, then it would be reasonable to return error values as ref.Val types rather than
	// simple go error types.
	if types.IsError(celVal) {
		return nil, celVal.(*types.Err)
	}
	return nil, errors.New("no such overload")
}
