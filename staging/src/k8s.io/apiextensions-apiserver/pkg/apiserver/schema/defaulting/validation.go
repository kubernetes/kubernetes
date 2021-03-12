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
	"fmt"
	"reflect"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	schemaobjectmeta "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/objectmeta"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"
	apiservervalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
	kubeopenapivalidate "k8s.io/kube-openapi/pkg/validation/validate"
)

// ValidateDefaults checks that default values validate and are properly pruned.
func ValidateDefaults(pth *field.Path, s *structuralschema.Structural, isResourceRoot, requirePrunedDefaults bool) (field.ErrorList, error) {
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

	return validate(pth, s, s, f, false, requirePrunedDefaults)
}

// validate is the recursive step func for the validation. insideMeta is true if s specifies
// TypeMeta or ObjectMeta. The SurroundingObjectFunc f is used to validate defaults of
// TypeMeta or ObjectMeta fields.
func validate(pth *field.Path, s *structuralschema.Structural, rootSchema *structuralschema.Structural, f SurroundingObjectFunc, insideMeta, requirePrunedDefaults bool) (field.ErrorList, error) {
	if s == nil {
		return nil, nil
	}

	if s.XEmbeddedResource {
		insideMeta = false
		f = NewRootObjectFunc().WithTypeMeta(metav1.TypeMeta{APIVersion: "validation/v1", Kind: "Validation"})
		rootSchema = s
	}

	allErrs := field.ErrorList{}

	if s.Default.Object != nil {
		validator := kubeopenapivalidate.NewSchemaValidator(s.ToKubeOpenAPI(), nil, "", strfmt.Default)

		if insideMeta {
			obj, _, err := f(runtime.DeepCopyJSONValue(s.Default.Object))
			if err != nil {
				// this should never happen. f(s.Default.Object) only gives an error if f is the
				// root object func, but the default value is not a map. But then we wouldn't be
				// in this case.
				return nil, fmt.Errorf("failed to validate default value inside metadata: %v", err)
			}

			// check ObjectMeta/TypeMeta and everything else
			if err := schemaobjectmeta.Coerce(nil, obj, rootSchema, true, false); err != nil {
				allErrs = append(allErrs, field.Invalid(pth.Child("default"), s.Default.Object, fmt.Sprintf("must result in valid metadata: %v", err)))
			} else if errs := schemaobjectmeta.Validate(nil, obj, rootSchema, true); len(errs) > 0 {
				allErrs = append(allErrs, field.Invalid(pth.Child("default"), s.Default.Object, fmt.Sprintf("must result in valid metadata: %v", errs.ToAggregate())))
			} else if errs := apiservervalidation.ValidateCustomResource(pth.Child("default"), s.Default.Object, validator); len(errs) > 0 {
				allErrs = append(allErrs, errs...)
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
			}
		}
	}

	// do not follow additionalProperties because defaults are forbidden there

	if s.Items != nil {
		errs, err := validate(pth.Child("items"), s.Items, rootSchema, f.Index(), insideMeta, requirePrunedDefaults)
		if err != nil {
			return nil, err
		}
		allErrs = append(allErrs, errs...)
	}

	for k, subSchema := range s.Properties {
		subInsideMeta := insideMeta
		if s.XEmbeddedResource && (k == "metadata" || k == "apiVersion" || k == "kind") {
			subInsideMeta = true
		}
		errs, err := validate(pth.Child("properties").Key(k), &subSchema, rootSchema, f.Child(k), subInsideMeta, requirePrunedDefaults)
		if err != nil {
			return nil, err
		}
		allErrs = append(allErrs, errs...)
	}

	return allErrs, nil
}
