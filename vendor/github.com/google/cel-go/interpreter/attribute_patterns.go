// Copyright 2020 Google LLC
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

	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// AttributePattern represents a top-level variable with an optional set of qualifier patterns.
//
// When using a CEL expression within a container, e.g. a package or namespace, the variable name
// in the pattern must match the qualified name produced during the variable namespace resolution.
// For example, if variable `c` appears in an expression whose container is `a.b`, the variable
// name supplied to the pattern must be `a.b.c`
//
// The qualifier patterns for attribute matching must be one of the following:
//
//   - valid map key type: string, int, uint, bool
//   - wildcard (*)
//
// Examples:
//
//   1. ns.myvar["complex-value"]
//   2. ns.myvar["complex-value"][0]
//   3. ns.myvar["complex-value"].*.name
//
// The first example is simple: match an attribute where the variable is 'ns.myvar' with a
// field access on 'complex-value'. The second example expands the match to indicate that only
// a specific index `0` should match. And lastly, the third example matches any indexed access
// that later selects the 'name' field.
type AttributePattern struct {
	variable          string
	qualifierPatterns []*AttributeQualifierPattern
}

// NewAttributePattern produces a new mutable AttributePattern based on a variable name.
func NewAttributePattern(variable string) *AttributePattern {
	return &AttributePattern{
		variable:          variable,
		qualifierPatterns: []*AttributeQualifierPattern{},
	}
}

// QualString adds a string qualifier pattern to the AttributePattern. The string may be a valid
// identifier, or string map key including empty string.
func (apat *AttributePattern) QualString(pattern string) *AttributePattern {
	apat.qualifierPatterns = append(apat.qualifierPatterns,
		&AttributeQualifierPattern{value: pattern})
	return apat
}

// QualInt adds an int qualifier pattern to the AttributePattern. The index may be either a map or
// list index.
func (apat *AttributePattern) QualInt(pattern int64) *AttributePattern {
	apat.qualifierPatterns = append(apat.qualifierPatterns,
		&AttributeQualifierPattern{value: pattern})
	return apat
}

// QualUint adds an uint qualifier pattern for a map index operation to the AttributePattern.
func (apat *AttributePattern) QualUint(pattern uint64) *AttributePattern {
	apat.qualifierPatterns = append(apat.qualifierPatterns,
		&AttributeQualifierPattern{value: pattern})
	return apat
}

// QualBool adds a bool qualifier pattern for a map index operation to the AttributePattern.
func (apat *AttributePattern) QualBool(pattern bool) *AttributePattern {
	apat.qualifierPatterns = append(apat.qualifierPatterns,
		&AttributeQualifierPattern{value: pattern})
	return apat
}

// Wildcard adds a special sentinel qualifier pattern that will match any single qualifier.
func (apat *AttributePattern) Wildcard() *AttributePattern {
	apat.qualifierPatterns = append(apat.qualifierPatterns,
		&AttributeQualifierPattern{wildcard: true})
	return apat
}

// VariableMatches returns true if the fully qualified variable matches the AttributePattern
// fully qualified variable name.
func (apat *AttributePattern) VariableMatches(variable string) bool {
	return apat.variable == variable
}

// QualifierPatterns returns the set of AttributeQualifierPattern values on the AttributePattern.
func (apat *AttributePattern) QualifierPatterns() []*AttributeQualifierPattern {
	return apat.qualifierPatterns
}

// AttributeQualifierPattern holds a wilcard or valued qualifier pattern.
type AttributeQualifierPattern struct {
	wildcard bool
	value    interface{}
}

// Matches returns true if the qualifier pattern is a wildcard, or the Qualifier implements the
// qualifierValueEquator interface and its IsValueEqualTo returns true for the qualifier pattern.
func (qpat *AttributeQualifierPattern) Matches(q Qualifier) bool {
	if qpat.wildcard {
		return true
	}
	qve, ok := q.(qualifierValueEquator)
	return ok && qve.QualifierValueEquals(qpat.value)
}

// qualifierValueEquator defines an interface for determining if an input value, of valid map key
// type, is equal to the value held in the Qualifier. This interface is used by the
// AttributeQualifierPattern to determine pattern matches for non-wildcard qualifier patterns.
//
// Note: Attribute values are also Qualifier values; however, Attriutes are resolved before
// qualification happens. This is an implementation detail, but one relevant to why the Attribute
// types do not surface in the list of implementations.
//
// See: partialAttributeFactory.matchesUnknownPatterns for more details on how this interface is
// used.
type qualifierValueEquator interface {
	// QualifierValueEquals returns true if the input value is equal to the value held in the
	// Qualifier.
	QualifierValueEquals(value interface{}) bool
}

// QualifierValueEquals implementation for boolean qualifiers.
func (q *boolQualifier) QualifierValueEquals(value interface{}) bool {
	bval, ok := value.(bool)
	return ok && q.value == bval
}

// QualifierValueEquals implementation for field qualifiers.
func (q *fieldQualifier) QualifierValueEquals(value interface{}) bool {
	sval, ok := value.(string)
	return ok && q.Name == sval
}

// QualifierValueEquals implementation for string qualifiers.
func (q *stringQualifier) QualifierValueEquals(value interface{}) bool {
	sval, ok := value.(string)
	return ok && q.value == sval
}

// QualifierValueEquals implementation for int qualifiers.
func (q *intQualifier) QualifierValueEquals(value interface{}) bool {
	ival, ok := value.(int64)
	return ok && q.value == ival
}

// QualifierValueEquals implementation for uint qualifiers.
func (q *uintQualifier) QualifierValueEquals(value interface{}) bool {
	uval, ok := value.(uint64)
	return ok && q.value == uval
}

// NewPartialAttributeFactory returns an AttributeFactory implementation capable of performing
// AttributePattern matches with PartialActivation inputs.
func NewPartialAttributeFactory(container *containers.Container,
	adapter ref.TypeAdapter,
	provider ref.TypeProvider) AttributeFactory {
	fac := NewAttributeFactory(container, adapter, provider)
	return &partialAttributeFactory{
		AttributeFactory: fac,
		container:        container,
		adapter:          adapter,
		provider:         provider,
	}
}

type partialAttributeFactory struct {
	AttributeFactory
	container *containers.Container
	adapter   ref.TypeAdapter
	provider  ref.TypeProvider
}

// AbsoluteAttribute implementation of the AttributeFactory interface which wraps the
// NamespacedAttribute resolution in an internal attributeMatcher object to dynamically match
// unknown patterns from PartialActivation inputs if given.
func (fac *partialAttributeFactory) AbsoluteAttribute(id int64, names ...string) NamespacedAttribute {
	attr := fac.AttributeFactory.AbsoluteAttribute(id, names...)
	return &attributeMatcher{fac: fac, NamespacedAttribute: attr}
}

// MaybeAttribute implementation of the AttributeFactory interface which ensure that the set of
// 'maybe' NamespacedAttribute values are produced using the PartialAttributeFactory rather than
// the base AttributeFactory implementation.
func (fac *partialAttributeFactory) MaybeAttribute(id int64, name string) Attribute {
	return &maybeAttribute{
		id: id,
		attrs: []NamespacedAttribute{
			fac.AbsoluteAttribute(id, fac.container.ResolveCandidateNames(name)...),
		},
		adapter:  fac.adapter,
		provider: fac.provider,
		fac:      fac,
	}
}

// matchesUnknownPatterns returns true if the variable names and qualifiers for a given
// Attribute value match any of the ActivationPattern objects in the set of unknown activation
// patterns on the given PartialActivation.
//
// For example, in the expression `a.b`, the Attribute is composed of variable `a`, with string
// qualifier `b`. When a PartialActivation is supplied, it indicates that some or all of the data
// provided in the input is unknown by specifying unknown AttributePatterns. An AttributePattern
// that refers to variable `a` with a string qualifier of `c` will not match `a.b`; however, any
// of the following patterns will match Attribute `a.b`:
//
// - `AttributePattern("a")`
// - `AttributePattern("a").Wildcard()`
// - `AttributePattern("a").QualString("b")`
// - `AttributePattern("a").QualString("b").QualInt(0)`
//
// Any AttributePattern which overlaps an Attribute or vice-versa will produce an Unknown result
// for the last pattern matched variable or qualifier in the Attribute. In the first matching
// example, the expression id representing variable `a` would be listed in the Unknown result,
// whereas in the other pattern examples, the qualifier `b` would be returned as the Unknown.
func (fac *partialAttributeFactory) matchesUnknownPatterns(
	vars PartialActivation,
	attrID int64,
	variableNames []string,
	qualifiers []Qualifier) (types.Unknown, error) {
	patterns := vars.UnknownAttributePatterns()
	candidateIndices := map[int]struct{}{}
	for _, variable := range variableNames {
		for i, pat := range patterns {
			if pat.VariableMatches(variable) {
				candidateIndices[i] = struct{}{}
			}
		}
	}
	// Determine whether to return early if there are no candidate unknown patterns.
	if len(candidateIndices) == 0 {
		return nil, nil
	}
	// Determine whether to return early if there are no qualifiers.
	if len(qualifiers) == 0 {
		return types.Unknown{attrID}, nil
	}
	// Resolve the attribute qualifiers into a static set. This prevents more dynamic
	// Attribute resolutions than necessary when there are multiple unknown patterns
	// that traverse the same Attribute-based qualifier field.
	newQuals := make([]Qualifier, len(qualifiers))
	for i, qual := range qualifiers {
		attr, isAttr := qual.(Attribute)
		if isAttr {
			val, err := attr.Resolve(vars)
			if err != nil {
				return nil, err
			}
			unk, isUnk := val.(types.Unknown)
			if isUnk {
				return unk, nil
			}
			// If this resolution behavior ever changes, new implementations of the
			// qualifierValueEquator may be required to handle proper resolution.
			qual, err = fac.NewQualifier(nil, qual.ID(), val)
			if err != nil {
				return nil, err
			}
		}
		newQuals[i] = qual
	}
	// Determine whether any of the unknown patterns match.
	for patIdx := range candidateIndices {
		pat := patterns[patIdx]
		isUnk := true
		matchExprID := attrID
		qualPats := pat.QualifierPatterns()
		for i, qual := range newQuals {
			if i >= len(qualPats) {
				break
			}
			matchExprID = qual.ID()
			qualPat := qualPats[i]
			// Note, the AttributeQualifierPattern relies on the input Qualifier not being an
			// Attribute, since there is no way to resolve the Attribute with the information
			// provided to the Matches call.
			if !qualPat.Matches(qual) {
				isUnk = false
				break
			}
		}
		if isUnk {
			return types.Unknown{matchExprID}, nil
		}
	}
	return nil, nil
}

// attributeMatcher embeds the NamespacedAttribute interface which allows it to participate in
// AttributePattern matching against Attribute values without having to modify the code paths that
// identify Attributes in expressions.
type attributeMatcher struct {
	NamespacedAttribute
	qualifiers []Qualifier
	fac        *partialAttributeFactory
}

// AddQualifier implements the Attribute interface method.
func (m *attributeMatcher) AddQualifier(qual Qualifier) (Attribute, error) {
	// Add the qualifier to the embedded NamespacedAttribute. If the input to the Resolve
	// method is not a PartialActivation, or does not match an unknown attribute pattern, the
	// Resolve method is directly invoked on the underlying NamespacedAttribute.
	_, err := m.NamespacedAttribute.AddQualifier(qual)
	if err != nil {
		return nil, err
	}
	// The attributeMatcher overloads TryResolve and will attempt to match unknown patterns against
	// the variable name and qualifier set contained within the Attribute. These values are not
	// directly inspectable on the top-level NamespacedAttribute interface and so are tracked within
	// the attributeMatcher.
	m.qualifiers = append(m.qualifiers, qual)
	return m, nil
}

// Resolve is an implementation of the Attribute interface method which uses the
// attributeMatcher TryResolve implementation rather than the embedded NamespacedAttribute
// Resolve implementation.
func (m *attributeMatcher) Resolve(vars Activation) (interface{}, error) {
	obj, found, err := m.TryResolve(vars)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("no such attribute: %v", m.NamespacedAttribute)
	}
	return obj, nil
}

// TryResolve is an implementation of the NamespacedAttribute interface method which tests
// for matching unknown attribute patterns and returns types.Unknown if present. Otherwise,
// the standard Resolve logic applies.
func (m *attributeMatcher) TryResolve(vars Activation) (interface{}, bool, error) {
	id := m.NamespacedAttribute.ID()
	partial, isPartial := vars.(PartialActivation)
	if isPartial {
		unk, err := m.fac.matchesUnknownPatterns(
			partial,
			id,
			m.CandidateVariableNames(),
			m.qualifiers)
		if err != nil {
			return nil, true, err
		}
		if unk != nil {
			return unk, true, nil
		}
	}
	return m.NamespacedAttribute.TryResolve(vars)
}

// Qualify is an implementation of the Qualifier interface method.
func (m *attributeMatcher) Qualify(vars Activation, obj interface{}) (interface{}, error) {
	val, err := m.Resolve(vars)
	if err != nil {
		return nil, err
	}
	unk, isUnk := val.(types.Unknown)
	if isUnk {
		return unk, nil
	}
	qual, err := m.fac.NewQualifier(nil, m.ID(), val)
	if err != nil {
		return nil, err
	}
	return qual.Qualify(vars, obj)
}
