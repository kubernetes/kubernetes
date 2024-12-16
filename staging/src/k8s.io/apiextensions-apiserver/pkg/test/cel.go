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
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/yaml"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
)

// MustLoadManifest loads a CRD from a file and panics on error.
func MustLoadManifest[T any](t *testing.T, pth string) *T {
	data, err := os.ReadFile(pth)
	require.NoError(t, err)

	var crd T
	err = yaml.Unmarshal(data, &crd)
	require.NoError(t, err)

	return &crd
}

// FieldValidators extracts the CEL validators by version and JSONPath from a CRD and returns
// a validator func for testing against samples.
func FieldValidators(t *testing.T, crd *apiextensionsv1.CustomResourceDefinition) (validatorsByVersionByJSONPath map[string]map[string]CELValidateFunc) {
	ret := map[string]map[string]CELValidateFunc{}
	for _, v := range crd.Spec.Versions {
		var internalSchema apiextensions.JSONSchemaProps
		err := apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(v.Schema.OpenAPIV3Schema, &internalSchema, nil)
		require.NoError(t, err, "failed to convert JSONSchemaProps for version %s: %v", v.Name, err)
		structuralSchema, err := schema.NewStructural(&internalSchema)
		require.NoError(t, err, "failed to create StructuralSchema for version %s: %v", v.Name, err)

		versionVals, err := findCEL(t, structuralSchema, true, field.NewPath("openAPIV3Schema"))
		require.NoError(t, err, "failed to find CEL for version %s: %v", v.Name, err)
		ret[v.Name] = versionVals
	}

	return ret
}

// VersionValidatorsFromFile extracts the CEL validators by version from a CRD file and returns
// a validator func for testing against samples.
func VersionValidatorsFromFile(t *testing.T, crdFilePath string) map[string]CELValidateFunc {
	data, err := os.ReadFile(crdFilePath)
	require.NoError(t, err)

	var crd apiextensionsv1.CustomResourceDefinition
	err = yaml.Unmarshal(data, &crd)
	require.NoError(t, err)

	ret := map[string]CELValidateFunc{}
	for _, v := range crd.Spec.Versions {
		var internalSchema apiextensions.JSONSchemaProps
		err := apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(v.Schema.OpenAPIV3Schema, &internalSchema, nil)
		require.NoError(t, err, "failed to convert JSONSchemaProps for version %s: %v", v.Name, err)
		structuralSchema, err := schema.NewStructural(&internalSchema)
		require.NoError(t, err, "failed to create StructuralSchema for version %s: %v", v.Name, err)
		ret[v.Name] = func(obj, old interface{}) field.ErrorList {
			errs, _ := cel.NewValidator(structuralSchema, true, celconfig.RuntimeCELCostBudget).Validate(context.TODO(), nil, structuralSchema, obj, old, celconfig.PerCallLimit)
			return errs
		}
	}

	return ret
}

// VersionValidatorFromFile extracts the CEL validators for a given version from a CRD file and returns
// a validator func for testing against samples.
func VersionValidatorFromFile(t *testing.T, crdFilePath string, version string) (CELValidateFunc, error) {
	vals := VersionValidatorsFromFile(t, crdFilePath)
	if val, ok := vals[version]; ok {
		return val, nil
	}
	return nil, fmt.Errorf("version %s not found", version)
}

// CELValidateFunc tests a sample object against a CEL validator.
type CELValidateFunc func(obj, old interface{}) field.ErrorList

func findCEL(t *testing.T, s *schema.Structural, root bool, pth *field.Path) (map[string]CELValidateFunc, error) {
	ret := map[string]CELValidateFunc{}

	if len(s.XValidations) > 0 {
		s := *s
		pth := *pth
		ret[pth.String()] = func(obj, old interface{}) field.ErrorList {
			errs, _ := cel.NewValidator(&s, root, celconfig.RuntimeCELCostBudget).Validate(context.TODO(), &pth, &s, obj, old, celconfig.PerCallLimit)
			return errs
		}
	}

	for k, v := range s.Properties {
		v := v
		sub, err := findCEL(t, &v, false, pth.Child("properties").Child(k))
		if err != nil {
			return nil, err
		}

		for pth, val := range sub {
			ret[pth] = val
		}
	}
	if s.Items != nil {
		sub, err := findCEL(t, s.Items, false, pth.Child("items"))
		if err != nil {
			return nil, err
		}
		for pth, val := range sub {
			ret[pth] = val
		}
	}
	if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil {
		sub, err := findCEL(t, s.AdditionalProperties.Structural, false, pth.Child("additionalProperties"))
		if err != nil {
			return nil, err
		}
		for pth, val := range sub {
			ret[pth] = val
		}
	}

	return ret, nil
}
