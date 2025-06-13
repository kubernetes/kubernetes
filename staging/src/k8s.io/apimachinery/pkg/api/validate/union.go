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
	"context"
	"fmt"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ExtractorFn extracts a member field from a parent object.
type ExtractorFn[T, V any] func(obj T) V

// Union verifies that exactly one member of a union is specified.
//
// UnionMembership must define all the members of the union.
//
// For example:
//
//	var UnionMembershipForABC := validate.NewUnionMembership([2]string{"a", "A"}, [2]string{"b", "B"}, [2]string{"c", "C"})
//	func ValidateABC(ctx context.Context, op operation.Operation, fldPath *field.Path, in *ABC) (errs fields.ErrorList) {
//		errs = append(errs, Union(ctx, op, fldPath, in, oldIn, UnionMembershipForABC,
//			func(in *ABC) any { return in.A },
//			func(in *ABC) any { return in.B },
//			func(in *ABC) any { return in.C },
//		)...)
//		return errs
//	}
func Union[T any](_ context.Context, op operation.Operation, fldPath *field.Path, obj, oldObj T, union *UnionMembership, extractorFns ...ExtractorFn[T, any]) field.ErrorList {
	if len(union.members) != len(extractorFns) {
		return field.ErrorList{
			field.InternalError(fldPath,
				fmt.Errorf("number of field extractors (%d) does not match number of union members (%d)",
					len(extractorFns), len(union.members))),
		}
	}
	var specifiedFields []string
	for i, extractor := range extractorFns {
		fieldValue := extractor(obj)
		rv := reflect.ValueOf(fieldValue)
		if rv.IsValid() && !rv.IsZero() {
			specifiedFields = append(specifiedFields, union.members[i].fieldName)
		}
	}
	if len(specifiedFields) > 1 {
		return field.ErrorList{
			field.Invalid(fldPath, fmt.Sprintf("{%s}", strings.Join(specifiedFields, ", ")),
				fmt.Sprintf("must specify exactly one of: %s", strings.Join(union.allFields(), ", "))),
		}
	}
	if len(specifiedFields) == 0 {
		return field.ErrorList{field.Invalid(fldPath, "",
			fmt.Sprintf("must specify one of: %s",
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
//	var UnionMembershipForABC := validate.NewDiscriminatedUnionMembership("type", [2]string{"a", "A"}, [2]string{"b" "B"}, [2]string{"c", "C"})
//	func ValidateABC(ctx context.Context, op operation.Operation, fldPath, *field.Path, in *ABC) (errs fields.ErrorList) {
//		errs = append(errs, DiscriminatedUnion(ctx, op, fldPath, in, oldIn, UnionMembershipForABC,
//			func(in *ABC) string { return string(in.Type) },
//			func(in *ABC) any { return in.A },
//			func(in *ABC) any { return in.B },
//			func(in *ABC) any { return in.C },
//		)...)
//		return errs
//	}
//
// It is not an error for the discriminatorValue to be unknown.  That must be
// validated on its own.
func DiscriminatedUnion[T any, D ~string](_ context.Context, op operation.Operation, fldPath *field.Path, obj, oldObj T, union *UnionMembership, discriminatorExtractor ExtractorFn[T, D], extractorFns ...ExtractorFn[T, any]) (errs field.ErrorList) {
	if len(union.members) != len(extractorFns) {
		return field.ErrorList{
			field.InternalError(fldPath,
				fmt.Errorf("number of field extractors (%d) does not match number of union members (%d)",
					len(extractorFns), len(union.members))),
		}
	}

	discriminatorValue := discriminatorExtractor(obj)

	for i, extractor := range extractorFns {
		member := union.members[i]
		isDiscriminatedMember := string(discriminatorValue) == member.discriminatorValue
		fieldValue := extractor(obj)
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

// allFields returns a string listing all the field names of the member of a union for use in error reporting.
func (u UnionMembership) allFields() []string {
	memberNames := make([]string, 0, len(u.members))
	for _, f := range u.members {
		memberNames = append(memberNames, fmt.Sprintf("`%s`", f.fieldName))
	}
	return memberNames
}
