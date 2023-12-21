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

package types

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
	"unicode"

	"github.com/google/cel-go/common/types/ref"
)

var (
	unspecifiedAttribute = &AttributeTrail{qualifierPath: []any{}}
)

// NewAttributeTrail creates a new simple attribute from a variable name.
func NewAttributeTrail(variable string) *AttributeTrail {
	if variable == "" {
		return unspecifiedAttribute
	}
	return &AttributeTrail{variable: variable}
}

// AttributeTrail specifies a variable with an optional qualifier path. An attribute value is expected to
// correspond to an AbsoluteAttribute, meaning a field selection which starts with a top-level variable.
//
// The qualifer path elements adhere to the AttributeQualifier type constraint.
type AttributeTrail struct {
	variable      string
	qualifierPath []any
}

// Equal returns whether two attribute values have the same variable name and qualifier paths.
func (a *AttributeTrail) Equal(other *AttributeTrail) bool {
	if a.Variable() != other.Variable() || len(a.QualifierPath()) != len(other.QualifierPath()) {
		return false
	}
	for i, q := range a.QualifierPath() {
		qual := other.QualifierPath()[i]
		if !qualifiersEqual(q, qual) {
			return false
		}
	}
	return true
}

func qualifiersEqual(a, b any) bool {
	if a == b {
		return true
	}
	switch numA := a.(type) {
	case int64:
		numB, ok := b.(uint64)
		if !ok {
			return false
		}
		return intUintEqual(numA, numB)
	case uint64:
		numB, ok := b.(int64)
		if !ok {
			return false
		}
		return intUintEqual(numB, numA)
	default:
		return false
	}
}

func intUintEqual(i int64, u uint64) bool {
	if i < 0 || u > math.MaxInt64 {
		return false
	}
	return i == int64(u)
}

// Variable returns the variable name associated with the attribute.
func (a *AttributeTrail) Variable() string {
	return a.variable
}

// QualifierPath returns the optional set of qualifying fields or indices applied to the variable.
func (a *AttributeTrail) QualifierPath() []any {
	return a.qualifierPath
}

// String returns the string representation of the Attribute.
func (a *AttributeTrail) String() string {
	if a.variable == "" {
		return "<unspecified>"
	}
	var str strings.Builder
	str.WriteString(a.variable)
	for _, q := range a.qualifierPath {
		switch q := q.(type) {
		case bool, int64:
			str.WriteString(fmt.Sprintf("[%v]", q))
		case uint64:
			str.WriteString(fmt.Sprintf("[%vu]", q))
		case string:
			if isIdentifierCharacter(q) {
				str.WriteString(fmt.Sprintf(".%v", q))
			} else {
				str.WriteString(fmt.Sprintf("[%q]", q))
			}
		}
	}
	return str.String()
}

func isIdentifierCharacter(str string) bool {
	for _, c := range str {
		if unicode.IsLetter(c) || unicode.IsDigit(c) || string(c) == "_" {
			continue
		}
		return false
	}
	return true
}

// AttributeQualifier constrains the possible types which may be used to qualify an attribute.
type AttributeQualifier interface {
	bool | int64 | uint64 | string
}

// QualifyAttribute qualifies an attribute using a valid AttributeQualifier type.
func QualifyAttribute[T AttributeQualifier](attr *AttributeTrail, qualifier T) *AttributeTrail {
	attr.qualifierPath = append(attr.qualifierPath, qualifier)
	return attr
}

// Unknown type which collects expression ids which caused the current value to become unknown.
type Unknown struct {
	attributeTrails map[int64][]*AttributeTrail
}

// NewUnknown creates a new unknown at a given expression id for an attribute.
//
// If the attribute is nil, the attribute value will be the `unspecifiedAttribute`.
func NewUnknown(id int64, attr *AttributeTrail) *Unknown {
	if attr == nil {
		attr = unspecifiedAttribute
	}
	return &Unknown{
		attributeTrails: map[int64][]*AttributeTrail{id: {attr}},
	}
}

// IDs returns the set of unknown expression ids contained by this value.
//
// Numeric identifiers are guaranteed to be in sorted order.
func (u *Unknown) IDs() []int64 {
	ids := make(int64Slice, len(u.attributeTrails))
	i := 0
	for id := range u.attributeTrails {
		ids[i] = id
		i++
	}
	ids.Sort()
	return ids
}

// GetAttributeTrails returns the attribute trails, if present, missing for a given expression id.
func (u *Unknown) GetAttributeTrails(id int64) ([]*AttributeTrail, bool) {
	trails, found := u.attributeTrails[id]
	return trails, found
}

// Contains returns true if the input unknown is a subset of the current unknown.
func (u *Unknown) Contains(other *Unknown) bool {
	for id, otherTrails := range other.attributeTrails {
		trails, found := u.attributeTrails[id]
		if !found || len(otherTrails) != len(trails) {
			return false
		}
		for _, ot := range otherTrails {
			found := false
			for _, t := range trails {
				if t.Equal(ot) {
					found = true
					break
				}
			}
			if !found {
				return false
			}
		}
	}
	return true
}

// ConvertToNative implements ref.Val.ConvertToNative.
func (u *Unknown) ConvertToNative(typeDesc reflect.Type) (any, error) {
	return u.Value(), nil
}

// ConvertToType is an identity function since unknown values cannot be modified.
func (u *Unknown) ConvertToType(typeVal ref.Type) ref.Val {
	return u
}

// Equal is an identity function since unknown values cannot be modified.
func (u *Unknown) Equal(other ref.Val) ref.Val {
	return u
}

// String implements the Stringer interface
func (u *Unknown) String() string {
	var str strings.Builder
	for id, attrs := range u.attributeTrails {
		if str.Len() != 0 {
			str.WriteString(", ")
		}
		if len(attrs) == 1 {
			str.WriteString(fmt.Sprintf("%v (%d)", attrs[0], id))
		} else {
			str.WriteString(fmt.Sprintf("%v (%d)", attrs, id))
		}
	}
	return str.String()
}

// Type implements ref.Val.Type.
func (u *Unknown) Type() ref.Type {
	return UnknownType
}

// Value implements ref.Val.Value.
func (u *Unknown) Value() any {
	return u
}

// IsUnknown returns whether the element ref.Val is in instance of *types.Unknown
func IsUnknown(val ref.Val) bool {
	switch val.(type) {
	case *Unknown:
		return true
	default:
		return false
	}
}

// MaybeMergeUnknowns determines whether an input value and another, possibly nil, unknown will produce
// an unknown result.
//
// If the input `val` is another Unknown, then the result will be the merge of the `val` and the input
// `unk`. If the `val` is not unknown, then the result will depend on whether the input `unk` is nil.
// If both values are non-nil and unknown, then the return value will be a merge of both unknowns.
func MaybeMergeUnknowns(val ref.Val, unk *Unknown) (*Unknown, bool) {
	src, isUnk := val.(*Unknown)
	if !isUnk {
		if unk != nil {
			return unk, true
		}
		return unk, false
	}
	return MergeUnknowns(src, unk), true
}

// MergeUnknowns combines two unknown values into a new unknown value.
func MergeUnknowns(unk1, unk2 *Unknown) *Unknown {
	if unk1 == nil {
		return unk2
	}
	if unk2 == nil {
		return unk1
	}
	out := &Unknown{
		attributeTrails: make(map[int64][]*AttributeTrail, len(unk1.attributeTrails)+len(unk2.attributeTrails)),
	}
	for id, ats := range unk1.attributeTrails {
		out.attributeTrails[id] = ats
	}
	for id, ats := range unk2.attributeTrails {
		existing, found := out.attributeTrails[id]
		if !found {
			out.attributeTrails[id] = ats
			continue
		}

		for _, at := range ats {
			found := false
			for _, et := range existing {
				if at.Equal(et) {
					found = true
					break
				}
			}
			if !found {
				existing = append(existing, at)
			}
		}
		out.attributeTrails[id] = existing
	}
	return out
}

// int64Slice is an implementation of the sort.Interface
type int64Slice []int64

// Len returns the number of elements in the slice.
func (x int64Slice) Len() int { return len(x) }

// Less indicates whether the value at index i is less than the value at index j.
func (x int64Slice) Less(i, j int) bool { return x[i] < x[j] }

// Swap swaps the values at indices i and j in place.
func (x int64Slice) Swap(i, j int) { x[i], x[j] = x[j], x[i] }

// Sort is a convenience method: x.Sort() calls Sort(x).
func (x int64Slice) Sort() { sort.Sort(x) }
