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
	"strings"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ExtractorFn extracts a value from a parent object. Depending on the context,
// that could be the value of a field or just whether that field was set or
// not.
// Note: obj is not guaranteed to be non-nil, need to handle nil obj in the
// extractor.
type ExtractorFn[T, V any] func(obj T) V

// UnionValidationOptions configures how union validation behaves
type UnionValidationOptions struct {
	// ErrorForEmpty returns error when no fields are set (nil means no error)
	ErrorForEmpty func(fldPath *field.Path, allFields []string) *field.Error

	// ErrorForMultiple returns error when multiple fields are set (nil means no error)
	ErrorForMultiple func(fldPath *field.Path, specifiedFields []string, allFields []string) *field.Error
}

// Union verifies that exactly one member of a union is specified.
//
// UnionMembership must define all the members of the union.
//
// For example:
//
//	var UnionMembershipForABC := validate.NewUnionMembership(
//	 	validate.NewUnionMember("a"),
//	 	validate.NewUnionMember("b"),
//	 	validate.NewUnionMember("c"),
//	 )
//	func ValidateABC(ctx context.Context, op operation.Operation, fldPath *field.Path, in *ABC) (errs field.ErrorList) {
//		errs = append(errs, Union(ctx, op, fldPath, in, oldIn, UnionMembershipForABC,
//			func(in *ABC) bool { return in.A != nil },
//			func(in *ABC) bool { return in.B != "" },
//			func(in *ABC) bool { return in.C != 0 },
//		)...)
//		return errs
//	}
func Union[T any](_ context.Context, op operation.Operation, fldPath *field.Path, obj, oldObj T, union *UnionMembership, isSetFns ...ExtractorFn[T, bool]) field.ErrorList {
	options := UnionValidationOptions{
		ErrorForEmpty: func(fldPath *field.Path, allFields []string) *field.Error {
			return field.Invalid(fldPath, "",
				fmt.Sprintf("must specify one of: %s", strings.Join(allFields, ", ")))
		},
		ErrorForMultiple: func(fldPath *field.Path, specifiedFields []string, allFields []string) *field.Error {
			return field.Invalid(fldPath, fmt.Sprintf("{%s}", strings.Join(specifiedFields, ", ")),
				fmt.Sprintf("must specify exactly one of: %s", strings.Join(allFields, ", ")))
		},
	}

	return unionValidate(op, fldPath, obj, oldObj, union, options, isSetFns...).WithOrigin("union")
}

// DiscriminatedUnion verifies specified union member matches the discriminator.
//
// UnionMembership must define all the members of the union and the discriminator.
//
// For example:
//
//	var UnionMembershipForABC = validate.NewDiscriminatedUnionMembership("type",
//	 	validate.NewDiscriminatedUnionMember("a", "A"),
//	 	validate.NewDiscriminatedUnionMember("b", "B"),
//	 	validate.NewDiscriminatedUnionMember("c", "C"),
//	)
//	func ValidateABC(ctx context.Context, op operation.Operation, fldPath *field.Path, in *ABC) (errs field.ErrorList) {
//		errs = append(errs, DiscriminatedUnion(ctx, op, fldPath, in, oldIn, UnionMembershipForABC,
//			func(in *ABC) string { return string(in.Type) },
//			func(in *ABC) bool { return in.A != nil },
//			func(in *ABC) bool { return in.B != "" },
//			func(in *ABC) bool { return in.C != 0 },
//		)...)
//		return errs
//	}
//
// It is not an error for the discriminatorValue to be unknown.  That must be
// validated on its own.
func DiscriminatedUnion[T any, D ~string](_ context.Context, op operation.Operation, fldPath *field.Path, obj, oldObj T, union *UnionMembership, discriminatorExtractor ExtractorFn[T, D], isSetFns ...ExtractorFn[T, bool]) (errs field.ErrorList) {
	if len(union.members) != len(isSetFns) {
		return field.ErrorList{
			field.InternalError(fldPath,
				fmt.Errorf("number of extractors (%d) does not match number of union members (%d)",
					len(isSetFns), len(union.members))),
		}
	}
	var changed bool
	discriminatorValue := discriminatorExtractor(obj)
	if op.Type == operation.Update {
		oldDiscriminatorValue := discriminatorExtractor(oldObj)
		changed = discriminatorValue != oldDiscriminatorValue
	}

	for i, fieldIsSet := range isSetFns {
		member := union.members[i]
		isDiscriminatedMember := string(discriminatorValue) == member.discriminatorValue
		newIsSet := fieldIsSet(obj)
		if op.Type == operation.Update && !changed {
			oldIsSet := fieldIsSet(oldObj)
			changed = changed || newIsSet != oldIsSet
		}
		if newIsSet && !isDiscriminatedMember {
			errs = append(errs, field.Invalid(fldPath.Child(member.fieldName), "",
				fmt.Sprintf("may only be specified when `%s` is %q", union.discriminatorName, member.discriminatorValue)))
		} else if !newIsSet && isDiscriminatedMember {
			errs = append(errs, field.Invalid(fldPath.Child(member.fieldName), "",
				fmt.Sprintf("must be specified when `%s` is %q", union.discriminatorName, discriminatorValue)))
		}
	}
	// If the union discriminator and membership is unchanged, we don't need to
	// re-validate.
	if op.Type == operation.Update && !changed {
		return nil
	}
	return errs.WithOrigin("union")
}

// UnionMember represents a member of a union.
type UnionMember struct {
	fieldName          string
	discriminatorValue string
}

// NewUnionMember returns a new UnionMember for the given field name.
func NewUnionMember(fieldName string) UnionMember {
	return UnionMember{fieldName: fieldName}
}

// NewDiscriminatedUnionMember returns a new UnionMember for the given field
// name and discriminator value.
func NewDiscriminatedUnionMember(fieldName, discriminatorValue string) UnionMember {
	return UnionMember{fieldName: fieldName, discriminatorValue: discriminatorValue}
}

// UnionMembership represents an ordered list of field union memberships.
type UnionMembership struct {
	discriminatorName string
	members           []UnionMember
}

// NewUnionMembership returns a new UnionMembership for the given list of members.
// Member names must be unique.
func NewUnionMembership(member ...UnionMember) *UnionMembership {
	return NewDiscriminatedUnionMembership("", member...)
}

// NewDiscriminatedUnionMembership returns a new UnionMembership for the given discriminator field and list of members.
// members are provided in the same way as for NewUnionMembership.
func NewDiscriminatedUnionMembership(discriminatorFieldName string, members ...UnionMember) *UnionMembership {
	return &UnionMembership{
		discriminatorName: discriminatorFieldName,
		members:           members,
	}
}

// allFields returns a string listing all the field names of the member of a union for use in error reporting.
func (u UnionMembership) allFields() []string {
	memberNames := make([]string, 0, len(u.members))
	for _, f := range u.members {
		memberNames = append(memberNames, fmt.Sprintf("`%s`", f.fieldName))
	}
	return memberNames
}

func unionValidate[T any](op operation.Operation, fldPath *field.Path,
	obj, oldObj T, union *UnionMembership, options UnionValidationOptions, isSetFns ...ExtractorFn[T, bool],
) field.ErrorList {
	if len(union.members) != len(isSetFns) {
		return field.ErrorList{
			field.InternalError(fldPath,
				fmt.Errorf("number of extractors (%d) does not match number of union members (%d)",
					len(isSetFns), len(union.members))),
		}
	}

	var specifiedFields []string
	var changed bool
	for i, fieldIsSet := range isSetFns {
		newIsSet := fieldIsSet(obj)
		if op.Type == operation.Update && !changed {
			oldIsSet := fieldIsSet(oldObj)
			changed = changed || newIsSet != oldIsSet
		}
		if newIsSet {
			specifiedFields = append(specifiedFields, union.members[i].fieldName)
		}
	}

	// If the union membership is unchanged, we don't need to re-validate.
	if op.Type == operation.Update && !changed {
		return nil
	}

	var errs field.ErrorList

	if len(specifiedFields) > 1 && options.ErrorForMultiple != nil {
		errs = append(errs, options.ErrorForMultiple(fldPath, specifiedFields, union.allFields()))
	}

	if len(specifiedFields) == 0 && options.ErrorForEmpty != nil {
		errs = append(errs, options.ErrorForEmpty(fldPath, union.allFields()))
	}

	return errs
}
