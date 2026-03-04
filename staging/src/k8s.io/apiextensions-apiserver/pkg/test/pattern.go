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

package test

import (
	"errors"
	"regexp"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// PatternValidators extracts the pattern validators by version and JSONPath from a CRD and returns
// a validator func for testing against samples.
func PatternValidators(t *testing.T, crd *apiextensionsv1.CustomResourceDefinition) (validatorsByVersionByJSONPath map[string]map[string]PatternValidateFunc) {
	ret := map[string]map[string]PatternValidateFunc{}
	for _, v := range crd.Spec.Versions {
		var internalSchema apiextensions.JSONSchemaProps
		err := apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(v.Schema.OpenAPIV3Schema, &internalSchema, nil)
		require.NoError(t, err, "failed to convert JSONSchemaProps for version %s: %v", v.Name, err)
		structuralSchema, err := schema.NewStructural(&internalSchema)
		require.NoError(t, err, "failed to create StructuralSchema for version %s: %v", v.Name, err)

		versionVals, err := findPattern(t, structuralSchema, field.NewPath("openAPIV3Schema"))
		require.NoError(t, err, "failed to find CEL for version %s: %v", v.Name, err)
		ret[v.Name] = versionVals
	}

	return ret
}

type PatternValidateFunc func(obj interface{}) error

func findPattern(t *testing.T, s *schema.Structural, pth *field.Path) (map[string]PatternValidateFunc, error) {
	ret := map[string]PatternValidateFunc{}

	if len(s.ValueValidation.Pattern) > 0 {
		s := *s
		pth := *pth
		ret[pth.String()] = func(obj interface{}) error {
			p, err := regexp.Compile(s.ValueValidation.Pattern)
			if err != nil {
				return err
			}
			if p.MatchString(obj.(string)) {
				return nil
			}
			return errors.New("pattern mismatch")
		}
	}

	for k, v := range s.Properties {
		v := v
		sub, err := findPattern(t, &v, pth.Child("properties").Child(k))
		if err != nil {
			return nil, err
		}

		for pth, val := range sub {
			ret[pth] = val
		}
	}
	if s.Items != nil {
		sub, err := findPattern(t, s.Items, pth.Child("items"))
		if err != nil {
			return nil, err
		}
		for pth, val := range sub {
			ret[pth] = val
		}
	}
	if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil {
		sub, err := findPattern(t, s.AdditionalProperties.Structural, pth.Child("additionalProperties"))
		if err != nil {
			return nil, err
		}
		for pth, val := range sub {
			ret[pth] = val
		}
	}

	return ret, nil
}
