/*
Copyright 2025 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// DiscriminatedRule defines a validation to apply for a specific discriminator value.
type DiscriminatedRule[Tfield any, Tdisc comparable] struct {
	Value      Tdisc
	Validation ValidateFunc[Tfield]
}

// Discriminated validates a member field based on a discriminator value.
// It iterates through the rules and applies the first one that matches the discriminator.
// If no rule matches, it applies the defaultValidation if provided.
//
// It performs ratcheting: if the operation is an Update, and neither the discriminator
// nor the value (checked via equiv) have changed, validation is skipped.
func Discriminated[Tfield any, Tdisc comparable, Tstruct any](ctx context.Context, op operation.Operation, structPath *field.Path,
	obj, oldObj *Tstruct, fieldName string, getMemberValue func(*Tstruct) Tfield, getDiscriminator func(*Tstruct) Tdisc,
	equiv MatchFunc[Tfield], defaultValidation ValidateFunc[Tfield], rules []DiscriminatedRule[Tfield, Tdisc],
) field.ErrorList {
	value := getMemberValue(obj)
	discriminator := getDiscriminator(obj)
	var oldValue Tfield
	var oldDiscriminator Tdisc

	if oldObj != nil {
		oldValue = getMemberValue(oldObj)
		oldDiscriminator = getDiscriminator(oldObj)
	}

	if op.Type == operation.Update && oldObj != nil && discriminator == oldDiscriminator && equiv(value, oldValue) {
		return nil
	}

	fldPath := structPath.Child(fieldName)
	for _, rule := range rules {
		if rule.Value == discriminator {
			if rule.Validation == nil {
				return nil
			}
			return rule.Validation(ctx, op, fldPath, value, oldValue)
		}
	}

	if defaultValidation != nil {
		return defaultValidation(ctx, op, fldPath, value, oldValue)
	}

	return nil
}
