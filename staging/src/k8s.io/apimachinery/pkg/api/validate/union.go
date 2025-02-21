/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package validate

import (
	"fmt"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Union verifies that exactly one member of a union is specified.
//
// UnionMembership must define all the members of the union.
//
// For example:
//
//	var abcUnionMembership := schema.NewUnionMembership("a", "b", "c")
//	func ValidateABC(opCtx operation.Context, fldPath *field.Path, in *ABC) (errs fields.ErrorList) {
//		errs = append(errs, Union(opCtx, fldPath, in, abcUnionMembership, in.A, in.B, in.C)...)
//		return errs
//	}
func Union(opCtx operation.Context, fldPath *field.Path, _, _ any, union *UnionMembership, fieldValues ...any) field.ErrorList {
	if len(union.members) != len(fieldValues) {
		return field.ErrorList{
			field.InternalError(fldPath,
				fmt.Errorf("number of field values (%d) does not match number of union members (%d)",
					len(fieldValues), len(union.members))),
		}
	}
	var specifiedMember *string
	for i, fieldValue := range fieldValues {
		rv := reflect.ValueOf(fieldValue)
		if rv.IsValid() && !rv.IsZero() {
			m := union.members[i]
			if specifiedMember != nil && *specifiedMember != m.discriminatorValue {
				return field.ErrorList{
					field.Invalid(fldPath, fmt.Sprintf("{%s}", strings.Join(union.specifiedFields(fieldValues), ", ")),
						fmt.Sprintf("must specify exactly one of: %s", strings.Join(union.allFields(), ", "))),
				}
			}
			name := m.discriminatorValue
			specifiedMember = &name
		}
	}
	if specifiedMember == nil {
		return field.ErrorList{field.Invalid(fldPath, "",
			fmt.Sprintf("must specify exactly one of: %s",
				strings.Join(union.allFields(), ", ")))}
	}
	return nil
}

// DiscriminatedUnion verifies specified union member matches the discriminator.
//
// UnionMembership must define all the members of the union and the discriminator.
//
// For example:
//
//	var abcUnionMembership := schema.NewDiscriminatedUnionMembership("type", "a", "b", "c")
//	func ValidateABC(opCtx operation.Context, fldPath, *field.Path, in *ABC) (errs fields.ErrorList) {
//		errs = append(errs, DiscriminatedUnion(opCtx, fldPath, in, abcUnionMembership, in.Type, in.A, in.B, in.C)...)
//		return errs
//	}
//
// It is not an error for the discriminatorValue to be unknown.  That must be
// validated on its own.
func DiscriminatedUnion[T ~string](opCtx operation.Context, fldPath *field.Path, _, _ any, union *UnionMembership, discriminatorValue T, fieldValues ...any) (errs field.ErrorList) {
	discriminatorStrValue := string(discriminatorValue)
	if len(union.members) != len(fieldValues) {
		return field.ErrorList{
			field.InternalError(fldPath,
				fmt.Errorf("number of field values (%d) does not match number of union members (%d)",
					len(fieldValues), len(union.members))),
		}
	}
	for i, fieldValue := range fieldValues {
		member := union.members[i]
		isDiscriminatedMember := discriminatorStrValue == member.discriminatorValue
		rv := reflect.ValueOf(fieldValue)
		isSpecified := rv.IsValid() && !rv.IsZero()
		if isSpecified && !isDiscriminatedMember {
			errs = append(errs, field.Invalid(fldPath.Child(member.fieldName), "",
				fmt.Sprintf("may only be specified when `%s` is %q", union.discriminatorName, member.discriminatorValue)))
		} else if !isSpecified && isDiscriminatedMember {
			errs = append(errs, field.Invalid(fldPath.Child(member.fieldName), "",
				fmt.Sprintf("must be specified when `%s` is %q", union.discriminatorName, discriminatorValue)))
		}
	}
	return errs
}

type member struct {
	fieldName, discriminatorValue string
}

// UnionMembership represents an ordered list of field union memberships.
type UnionMembership struct {
	discriminatorName string
	members           []member
}

// NewUnionMembership returns a new UnionMembership for the given list of members.
//
// Each member is a [2]string to provide a fieldName and discriminatorValue pair, where
// [0] identifies the field name and [1] identifies the union member Name.
//
// Field names must be unique.
func NewUnionMembership(member ...[2]string) *UnionMembership {
	return NewDiscriminatedUnionMembership("", member...)
}

// NewDiscriminatedUnionMembership returns a new UnionMembership for the given discriminator field and list of members.
// members are provided in the same way as for NewUnionMembership.
func NewDiscriminatedUnionMembership(discriminatorFieldName string, members ...[2]string) *UnionMembership {
	u := &UnionMembership{}
	u.discriminatorName = discriminatorFieldName
	for _, fieldName := range members {
		u.members = append(u.members, member{fieldName: fieldName[0], discriminatorValue: fieldName[1]})
	}
	return u
}

// specifiedFields returns a string listing all the field names of the specified fieldValues for use in error reporting.
func (u UnionMembership) specifiedFields(fieldValues []any) []string {
	var membersSpecified []string
	for i, fieldValue := range fieldValues {
		rv := reflect.ValueOf(fieldValue)
		if rv.IsValid() && !rv.IsZero() {
			f := u.members[i]
			membersSpecified = append(membersSpecified, f.fieldName)
		}
	}
	return membersSpecified
}

// allFields returns a string listing all the field names of the member of a union for use in error reporting.
func (u UnionMembership) allFields() []string {
	memberNames := make([]string, 0, len(u.members))
	for _, f := range u.members {
		memberNames = append(memberNames, fmt.Sprintf("`%s`", f.fieldName))
	}
	return memberNames
}
