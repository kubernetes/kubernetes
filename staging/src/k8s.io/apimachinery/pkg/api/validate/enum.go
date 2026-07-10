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
	"slices"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Enum verifies that a given value is a member of a set of enum values.
// Exclude Rules that apply when options are enabled or disabled are also considered.
// If ANY exclude rule matches for a value, that value is excluded from the enum when validating.
func Enum[T ~string](_ context.Context, op operation.Operation, fldPath *field.Path, value, _ *T, validValues sets.Set[T], exclusions []EnumExclusion[T]) field.ErrorList {
	if value == nil {
		return nil
	}
	excluded, err := isExcluded(op, exclusions, *value)
	if err != nil {
		return field.ErrorList{field.InternalError(fldPath, err)}
	}
	if !validValues.Has(*value) || excluded {
		supported, err := supportedValues(op, validValues, exclusions)
		if err != nil {
			return field.ErrorList{field.InternalError(fldPath, err)}
		}
		return field.ErrorList{field.NotSupported[T](fldPath, *value, supported)}
	}
	return nil
}

// supportedValues returns a sorted list of supported values.
// Excluded enum values are not included in the list.
func supportedValues[T ~string](op operation.Operation, values sets.Set[T], exclusions []EnumExclusion[T]) ([]T, error) {
	res := make([]T, 0, len(values))
	for key := range values {
		excluded, err := isExcluded(op, exclusions, key)
		if err != nil {
			return nil, err
		}
		if excluded {
			continue
		}
		res = append(res, key)
	}
	slices.Sort(res)
	return res, nil
}

// EnumExclusion represents a single enum exclusion rule.
type EnumExclusion[T ~string] struct {
	// Value specifies the enum value to be conditionally excluded.
	Value T
	// ExcludeWhen determines the condition for exclusion.
	// If true, the value is excluded if the option is present.
	// If false, the value is excluded if the option is NOT present.
	ExcludeWhen bool
	// Option is the name of the feature option that controls the exclusion.
	Option string
}

func isExcluded[T ~string](op operation.Operation, exclusions []EnumExclusion[T], value T) (bool, error) {
	for _, rule := range exclusions {
		if rule.Value != value {
			continue
		}
		on, defined := op.HasOption(rule.Option)
		if !defined {
			return false, fmt.Errorf("undefined validation option %q", rule.Option)
		}
		if rule.ExcludeWhen == on {
			return true, nil
		}
	}
	return false, nil
}
