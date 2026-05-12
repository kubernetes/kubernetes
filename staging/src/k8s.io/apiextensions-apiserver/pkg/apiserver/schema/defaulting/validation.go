/*
Copyright 2019 The Kubernetes Authors.

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

package defaulting

import (
	"context"
	"fmt"
	"reflect"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel"
	schemaobjectmeta "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/objectmeta"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"
	apiservervalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// ValidateDefaults checks that default values validate and are properly pruned.
// context is passed for supporting context cancellation during cel validation
func ValidateDefaults(ctx context.Context, pth *field.Path, s *structuralschema.Structural, isResourceRoot, requirePrunedDefaults bool) (field.ErrorList, error) {
	f := NewRootObjectFunc().WithTypeMeta(metav1.TypeMeta{APIVersion: "validation/v1", Kind: "Validation"})

	if isResourceRoot {
		if s == nil {
			s = &structuralschema.Structural{}
		}
		if !s.XEmbeddedResource {
			clone := *s
			clone.XEmbeddedResource = true
			s = &clone
		}
	}

	allErr, error, _ := validate(ctx, pth, s, s, f, false, requirePrunedDefaults, celconfig.RuntimeCELCostBudget)
	return allErr, error
}

// validate is the recursive step func for the validation. insideMeta is true if s specifies
// TypeMeta or ObjectMeta. The SurroundingObjectFunc f is used to validate defaults of
// TypeMeta or ObjectMeta fields.
// context is passed for supporting context cancellation during cel validation
func validate(ctx context.Context, pth *field.Path, s *structuralschema.Structural, rootSchema *structuralschema.Structural, f SurroundingObjectFunc, insideMeta, requirePrunedDefaults bool, costBudget int64) (allErrs field.ErrorList, error error, remainingCost int64) {
	remainingCost = costBudget
	if s == nil {
		return nil, nil, remainingCost
	}

	if s.XEmbeddedResource {
		insideMeta = false
		f = NewRootObjectFunc().WithTypeMeta(metav1.TypeMeta{APIVersion: "validation/v1", Kind: "Validation"})
		rootSchema = s
	}

	isResourceRoot := s == rootSchema

	if s.Default.Object != nil {
		validator := apiservervalidation.NewSchemaValidatorFromOpenAPI(s.ToKubeOpenAPI())

		if insideMeta {
			obj, _, err := f(runtime.DeepCopyJSONValue(s.Default.Object))
			if err != nil {
				// this should never happen. f(s.Default.Object) only gives an error if f is the
				// root object func, but the default value is not a map. But then we wouldn't be
				// in this case.
				return nil, fmt.Errorf("failed to validate default value inside metadata: %v", err), remainingCost
			}

			// check ObjectMeta/TypeMeta and everything else
			if err := schemaobjectmeta.Coerce(nil, obj, rootSchema, true, false); err != nil {
				allErrs = append(allErrs, field.Invalid(pth.Child("default"), s.Default.Object, fmt.Sprintf("must result in valid metadata: %v", err)))
			} else if errs := schemaobjectmeta.Validate(nil, obj, rootSchema, true); len(errs) > 0 {
				allErrs = append(allErrs, field.Invalid(pth.Child("default"), s.Default.Object, fmt.Sprintf("must result in valid metadata: %v", errs.ToAggregate())))
			} else if errs := apiservervalidation.ValidateCustomResource(pth.Child("default"), s.Default.Object, validator); len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			} else if celValidator := cel.NewValidator(s, isResourceRoot, celconfig.PerCallLimit); celValidator != nil {
				celErrs, rmCost := celValidator.Validate(ctx, pth.Child("default"), s, s.Default.Object, s.Default.Object, remainingCost)
				allErrs = append(allErrs, celErrs...)

				if len(celErrs) == 0 && utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CRDValidationRatcheting) {
					// If ratcheting is enabled some CEL rules may use optionalOldSelf
					// For such rules the above validation is not sufficient for
					// determining if the default value is a valid value to introduce
					// via create or uncorrelated update.
					//
					// Validate an update from nil to the default value to ensure
					// that the default value pass
					celErrs, rmCostWithoutOldObject := celValidator.Validate(ctx, pth.Child("default"), s, s.Default.Object, nil, remainingCost)
					allErrs = append(allErrs, celErrs...)

					// capture the cost of both types of runs and take whichever
					// leaves less remaining cost
					if rmCostWithoutOldObject < rmCost {
						rmCost = rmCostWithoutOldObject
					}
				}

				remainingCost = rmCost
				if remainingCost < 0 {
					return allErrs, nil, remainingCost
				}
			}
		} else {
			// check whether default is pruned
			if requirePrunedDefaults {
				pruned := runtime.DeepCopyJSONValue(s.Default.Object)
				pruning.Prune(pruned, s, s.XEmbeddedResource)
				if !reflect.DeepEqual(pruned, s.Default.Object) {
					allErrs = append(allErrs, field.Invalid(pth.Child("default"), s.Default.Object, "must not have unknown fields"))
				}
			}

			// check ObjectMeta/TypeMeta and everything else
			if err := schemaobjectmeta.Coerce(pth.Child("default"), s.Default.Object, s, s.XEmbeddedResource, false); err != nil {
				allErrs = append(allErrs, err)
			} else if errs := schemaobjectmeta.Validate(pth.Child("default"), s.Default.Object, s, s.XEmbeddedResource); len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			} else if errs := apiservervalidation.ValidateCustomResource(pth.Child("default"), s.Default.Object, validator); len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			} else if celValidator := cel.NewValidator(s, isResourceRoot, celconfig.PerCallLimit); celValidator != nil {
				celErrs, rmCost := celValidator.Validate(ctx, pth.Child("default"), s, s.Default.Object, s.Default.Object, remainingCost)
				allErrs = append(allErrs, celErrs...)

				if len(celErrs) == 0 && utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CRDValidationRatcheting) {
					// If ratcheting is enabled some CEL rules may use optionalOldSelf
					// For such rules the above validation is not sufficient for
					// determining if the default value is a valid value to introduce
					// via create or uncorrelated update.
					//
					// Validate an update from nil to the default value to ensure
					// that the default value pass
					celErrs, rmCostWithoutOldObject := celValidator.Validate(ctx, pth.Child("default"), s, s.Default.Object, nil, remainingCost)
					allErrs = append(allErrs, celErrs...)

					// capture the cost of both types of runs and take whichever
					// leaves less remaining cost
					if rmCostWithoutOldObject < rmCost {
						rmCost = rmCostWithoutOldObject
					}
				}

				remainingCost = rmCost
				if remainingCost < 0 {
					return allErrs, nil, remainingCost
				}
			}
		}
	}

	// do not follow additionalProperties because defaults are forbidden there

	if s.Items != nil {
		errs, err, rCost := validate(ctx, pth.Child("items"), s.Items, rootSchema, f.Index(), insideMeta, requirePrunedDefaults, remainingCost)
		remainingCost = rCost
		allErrs = append(allErrs, errs...)
		if err != nil {
			return nil, err, remainingCost
		}
		if remainingCost < 0 {
			return allErrs, nil, remainingCost
		}
	}

	for k, subSchema := range s.Properties {
		subInsideMeta := insideMeta
		if s.XEmbeddedResource && (k == "metadata" || k == "apiVersion" || k == "kind") {
			subInsideMeta = true
		}
		errs, err, rCost := validate(ctx, pth.Child("properties").Key(k), &subSchema, rootSchema, f.Child(k), subInsideMeta, requirePrunedDefaults, remainingCost)
		remainingCost = rCost
		allErrs = append(allErrs, errs...)
		if err != nil {
			return nil, err, remainingCost
		}
		if remainingCost < 0 {
			return allErrs, nil, remainingCost
		}
	}

	return allErrs, nil, remainingCost
}
