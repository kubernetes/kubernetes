/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUTHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package union

import (
	"fmt"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Validate traverses the object and the schema, and validates fields with x-kubernetes-unions.
func Validate(path *field.Path, s *schema.Structural, obj interface{}) field.ErrorList {
	fmt.Println("Enter Validate", path, s, obj)
	if s == nil {
		return nil
	}

	var errs field.ErrorList
	switch x := obj.(type) {
	case map[string]interface{}:
		errs = append(errs, validateObject(path, s, x)...)
	case []interface{}:
		errs = append(errs, validateList(path, s, x)...)
	}

	return errs
}

func validateObject(path *field.Path, s *schema.Structural, obj map[string]interface{}) field.ErrorList {
	fmt.Println("Enter validateObject", path, s, obj)
	var errs field.ErrorList

	for _, union := range s.XUnions {
		fieldNames := make([]string, len(union.Members))
		for i, f := range union.Members {
			fieldNames[i] = f.FieldName
		}

		if union.Discriminator != "" {
			discriminatorValue, discriminatorSet := obj[union.Discriminator]
			if !discriminatorSet {
				// discriminator is not set. If any of the fields are set, this is an error.
				activeFields := 0
				for _, fieldName := range fieldNames {
					if _, ok := obj[fieldName]; ok {
						activeFields++
					}
				}
				if activeFields > 0 {
					errs = append(errs, field.Required(path.Child(union.Discriminator), fmt.Sprintf("discriminator is required when one of %v is set", fieldNames)))
				}
			} else if discriminator, isString := discriminatorValue.(string); !isString {
				errs = append(errs, field.Invalid(path.Child(union.Discriminator), discriminatorValue, "discriminator must be a string"))
			} else {
				// discriminator is set and is a string
				var discriminatorValueFoundInUnion bool
				for _, m := range union.Members {
					if discriminator == m.DiscriminatorValue {
						discriminatorValueFoundInUnion = true
						break
					}
				}
				if !discriminatorValueFoundInUnion {
					discriminatorValues := make([]string, len(union.Members))
					for i, m := range union.Members {
						discriminatorValues[i] = m.DiscriminatorValue
					}
					errs = append(errs, field.Invalid(path.Child(union.Discriminator), discriminator, fmt.Sprintf("discriminator value must be one of %v", discriminatorValues)))
				} else {
					for _, member := range union.Members {
						if discriminator == member.DiscriminatorValue {
							// This is the active member. Its field must be present.
							if _, ok := obj[member.FieldName]; !ok {
								errs = append(errs, field.Invalid(path, obj, fmt.Sprintf("discriminator set to %s, but field %s is not set", discriminator, member.FieldName)))
							}
						} else {
							// This is an inactive member. Its field must NOT be present.
							if _, ok := obj[member.FieldName]; ok {
								errs = append(errs, field.Invalid(path, obj, fmt.Sprintf("field %s is set but discriminator is '%s', not '%s'", member.FieldName, discriminator, member.DiscriminatorValue)))
							}
						}
					}
				}
			}
		} else { // non-discriminated union
			activeFields := 0
			for _, fieldName := range fieldNames {
				if _, ok := obj[fieldName]; ok {
					activeFields++
				}
			}
			if union.ZeroOrOneOf {
				if activeFields > 1 {
					errs = append(errs, field.Invalid(path, obj, fmt.Sprintf("at most one of %v is allowed", fieldNames)))
				}
			} else if activeFields != 1 {
				errs = append(errs, field.Invalid(path, obj, fmt.Sprintf("exactly one of %v is required", fieldNames)))
			}
		}
	}

	for k, v := range obj {
		if prop, ok := s.Properties[k]; ok {
			errs = append(errs, Validate(path.Child(k), &prop, v)...)
		}
		if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil {
			errs = append(errs, Validate(path.Child(k), s.AdditionalProperties.Structural, v)...)
		}
	}

	return errs
}

func validateList(path *field.Path, s *schema.Structural, list []interface{}) field.ErrorList {
	fmt.Println("Enter validateList", path, s, list)
	var errs field.ErrorList
	if s.Items != nil {
		for i, item := range list {
			errs = append(errs, Validate(path.Index(i), s.Items, item)...)
		}
	}
	return errs
}
