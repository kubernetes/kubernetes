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

package objectmeta

import (
	"strings"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	metavalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/api/validation/path"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Validate validates embedded ObjectMeta and TypeMeta.
// It also validate those at the root if isResourceRoot is true.
func Validate(pth *field.Path, obj interface{}, s *structuralschema.Structural, isResourceRoot bool) field.ErrorList {
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
	return validate(pth, obj, s)
}

func validate(pth *field.Path, x interface{}, s *structuralschema.Structural) field.ErrorList {
	if s == nil {
		return nil
	}

	var allErrs field.ErrorList

	switch x := x.(type) {
	case map[string]interface{}:
		if s.XEmbeddedResource {
			allErrs = append(allErrs, validateEmbeddedResource(pth, x, s)...)
		}

		for k, v := range x {
			prop, ok := s.Properties[k]
			if ok {
				allErrs = append(allErrs, validate(pth.Child(k), v, &prop)...)
			} else if s.AdditionalProperties != nil {
				allErrs = append(allErrs, validate(pth.Key(k), v, s.AdditionalProperties.Structural)...)
			}
		}
	case []interface{}:
		for i, v := range x {
			allErrs = append(allErrs, validate(pth.Index(i), v, s.Items)...)
		}
	default:
		// scalars, do nothing
	}

	return allErrs
}

func validateEmbeddedResource(pth *field.Path, x map[string]interface{}, s *structuralschema.Structural) field.ErrorList {
	var allErrs field.ErrorList

	// require apiVersion and kind, but not metadata
	if _, found := x["apiVersion"]; !found {
		allErrs = append(allErrs, field.Required(pth.Child("apiVersion"), ""))
	}
	if _, found := x["kind"]; !found {
		allErrs = append(allErrs, field.Required(pth.Child("kind"), ""))
	}

	for k, v := range x {
		switch k {
		case "apiVersion":
			if apiVersion, ok := v.(string); !ok {
				allErrs = append(allErrs, field.Invalid(pth.Child("apiVersion"), v, "must be a string"))
			} else if len(apiVersion) == 0 {
				allErrs = append(allErrs, field.Invalid(pth.Child("apiVersion"), apiVersion, "must not be empty"))
			} else if _, err := schema.ParseGroupVersion(apiVersion); err != nil {
				allErrs = append(allErrs, field.Invalid(pth.Child("apiVersion"), apiVersion, err.Error()))
			}
		case "kind":
			if kind, ok := v.(string); !ok {
				allErrs = append(allErrs, field.Invalid(pth.Child("kind"), v, "must be a string"))
			} else if len(kind) == 0 {
				allErrs = append(allErrs, field.Invalid(pth.Child("kind"), kind, "must not be empty"))
			} else if errs := utilvalidation.IsDNS1035Label(strings.ToLower(kind)); len(errs) > 0 {
				allErrs = append(allErrs, field.Invalid(pth.Child("kind"), kind, "may have mixed case, but should otherwise match: "+strings.Join(errs, ",")))
			}
		case "metadata":
			meta, _, err := GetObjectMeta(x, false)
			if err != nil {
				allErrs = append(allErrs, field.Invalid(pth.Child("metadata"), v, err.Error()))
			} else {
				if len(meta.Name) == 0 {
					meta.Name = "fakename" // we have to set something to avoid an error
				}
				allErrs = append(allErrs, metavalidation.ValidateObjectMeta(meta, len(meta.Namespace) > 0, path.ValidatePathSegmentName, pth.Child("metadata"))...)
			}
		}
	}

	return allErrs
}
