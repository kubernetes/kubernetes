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

	"github.com/go-openapi/strfmt"
	"github.com/go-openapi/validate"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/objectmeta"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/pruning"
	apiservervalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidateDefaults checks that default values validate and properly pruned.
func ValidateDefaults(pth *field.Path, s *structuralschema.Structural) (field.ErrorList, error) {
	f := NewRootObjectFunc().WithTypeMeta(metav1.TypeMeta{APIVersion: "validation/v1", Kind: "Validation"})
	return validator{s}.validate(pth, s, f, false)
}

type validator struct {
	rootSchema *structuralschema.Structural
}

func (v *validator) validate(pth *field.Path, s *structuralschema.Structural, f SurroundingObjectFunc, insideMetadata bool) (field.ErrorList, error) {
	if s == nil {
		return nil, nil
	}

	if s.XEmbeddedResource {
		insideMetadata = false
		f = NewRootObjectFunc().WithTypeMeta(metav1.TypeMeta{APIVersion: "validation/v1", Kind: "Validation"})
		v2 := *v
		v = &v2
		v.rootSchema = s
	}

	allErrs := field.ErrorList{}

	if s.Default.Object != nil {
		validator := validate.NewSchemaValidator(s.ToGoOpenAPI(), nil, "", strfmt.Default)

		obj, acc, err := f(runtime.DeepCopyJSONValue(s.Default.Object))
		if err != nil {
			return nil, fmt.Errorf("failed to prune default value: %v", err)
		}
		if insideMetadata {
			// check ObjectMeta/TypeMeta and everything else
			if err := objectmeta.Coerce(pth.Child("default"), obj, nil, true, false); err != nil {
				allErrs = append(allErrs, err)
			} else if errs := objectmeta.Validate(pth.Child("default"), obj, v.rootSchema, true); len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			} else if errs := apiservervalidation.ValidateCustomResource(pth.Child("default"), s.Default.Object, validator); len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			}
		} else {
			// check whether default is pruned
			pruning.Prune(obj, s, s.XEmbeddedResource)
			prunedDefault, _, err := acc(obj)
			if err != nil {
				return nil, fmt.Errorf("failed to prune default value: %v", err)
			}
			if !reflect.DeepEqual(prunedDefault, s.Default.Object) {
				allErrs = append(allErrs, field.Invalid(pth.Child("default"), s.Default.Object, "must be pruned"))
			}

			// check ObjectMeta/TypeMeta and everything else
			if err := objectmeta.Coerce(pth.Child("default"), s.Default.Object, s, s.XEmbeddedResource, false); err != nil {
				allErrs = append(allErrs, err)
			} else if errs := objectmeta.Validate(pth.Child("default"), s.Default.Object, s, s.XEmbeddedResource); len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			} else if errs := apiservervalidation.ValidateCustomResource(pth.Child("default"), s.Default.Object, validator); len(errs) > 0 {
				allErrs = append(allErrs, errs...)
			}
		}
	}

	// do not follow additionalProperties because defaults are forbidden there

	if s.Items != nil {
		if errs, err := v.validate(pth.Child("items"), s.Items, f.Index(), insideMetadata); err != nil {
			return nil, err
		} else {
			allErrs = append(allErrs, errs...)
		}
	}

	for k, subSchema := range s.Properties {
		subInsideMetadata := insideMetadata
		if s.XEmbeddedResource && k == "metadata" {
			subInsideMetadata = true
		}
		if errs, err := v.validate(pth.Child("properties"), &subSchema, f.Child(k), subInsideMetadata); err != nil {
			return nil, err
		} else {
			allErrs = append(allErrs, errs...)
		}
	}

	return allErrs, nil
}
