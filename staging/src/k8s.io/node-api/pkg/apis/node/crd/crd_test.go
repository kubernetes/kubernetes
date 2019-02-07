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

package crd

import (
	"fmt"
	"io/ioutil"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/validation"
	"k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apimachinery/pkg/util/validation/field"
	nodev1alpha1 "k8s.io/node-api/pkg/apis/node/v1alpha1"
)

const (
	runtimeClassCRD = "runtimeclass_crd.yaml"
	addonsRoot      = "../../../../../../../../cluster/addons"

	addonManagerModeLabelKey = "addonmanager.kubernetes.io/mode"
)

func TestRuntimeClassCRD(t *testing.T) {
	crd, err := loadCRD(runtimeClassCRD)
	require.NoError(t, err)

	runtimeClass := reflect.TypeOf(nodev1alpha1.RuntimeClass{})
	skipFields := []string{
		"", // TypeMeta
		"metadata",
	}
	errs := verifyStructProperties(field.NewPath(""), runtimeClass, crd.Spec.Validation.OpenAPIV3Schema, skipFields...)

	require.NoError(t, errs.ToAggregate())
}

// TestCRDSources verifies that all the CRD sources are in sync. The sources include:
// - CRD yaml here (the source of truth)
// - The CRD loaded from bindata, generated from the source of truth
// - The CRD copied into the addons directory
func TestCRDSources(t *testing.T) {
	srcs := []struct {
		truth, addon string
		generator    func() (*apiextensionsv1beta1.CustomResourceDefinition, error)
	}{
		{
			truth:     runtimeClassCRD,
			addon:     filepath.Join(addonsRoot, "runtimeclass", runtimeClassCRD),
			generator: RuntimeClassCRD,
		},
	}

	for _, src := range srcs {
		t.Run(src.truth, func(t *testing.T) {
			trueCRD, err := loadCRD(src.truth)
			require.NoError(t, err, "Failed to load source-of-truth CRD")

			generatedCRD, err := src.generator()
			require.NoError(t, err, "Failed to load generated CRD")

			assert.Equal(t, trueCRD, generatedCRD, "Generated CRD mismatch")

			addonCRD, err := loadCRD(src.addon)
			require.NoError(t, err, "Failed to load addon CRD")

			// Copy the addon manager label over to the source of truth, to enable direct comparison.
			mode, ok := addonCRD.Labels[addonManagerModeLabelKey]
			require.True(t, ok, "Missing addon manager mode label")
			if trueCRD.Labels == nil {
				trueCRD.Labels = map[string]string{}
			}
			trueCRD.Labels[addonManagerModeLabelKey] = mode

			assert.Equal(t, trueCRD, addonCRD, "Addon CRD mismatch")

		})
	}
}

func loadCRD(filename string) (*apiextensionsv1beta1.CustomResourceDefinition, error) {
	if filename == "" {
		return nil, fmt.Errorf("missing filename")
	}
	raw, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file %q: %v", filename, err)
	}

	crd, err := decodeCRD(raw)
	if err != nil {
		return nil, fmt.Errorf("failed to decode %q into CRD: %v", filename, err)
	}

	// Validate the loaded CRD.
	copy := crd.DeepCopy()
	apiserver.Scheme.Default(copy) // Validation requires a defaulted object.
	internal := &apiextensions.CustomResourceDefinition{}
	if err := apiserver.Scheme.Convert(copy, internal, nil); err != nil {
		return nil, fmt.Errorf("failed to convert external CRD to internal version: %v", err)
	}

	if err := validation.ValidateCustomResourceDefinition(internal).ToAggregate(); err != nil {
		return nil, fmt.Errorf("invalid CRD %q: %v", filename, err)
	}

	return crd, nil
}

func verifyStructProperties(path *field.Path, st reflect.Type, props *apiextensionsv1beta1.JSONSchemaProps, skipFields ...string) field.ErrorList {
	var errs field.ErrorList
	fields := map[string]bool{}

	for i := 0; i < st.NumField(); i++ {
		fld := st.Field(i)
		fldName := jsonTagName(fld.Tag)
		if contains(fldName, skipFields) {
			continue
		}

		fldPath := path.Child(fldName)

		prop, found := props.Properties[fldName]
		if !found {
			errs = append(errs, field.Required(fldPath, "CRD missing field"))
			continue
		}
		fields[fldName] = true

		errs = append(errs, verifyFieldProperties(fldPath, fld.Type, &prop)...)
	}

	for key := range props.Properties {
		if _, ok := fields[key]; !ok {
			errs = append(errs, field.Required(path.Child(key), "typed API missing field"))
		}
	}

	return errs
}

func verifyFieldProperties(path *field.Path, fldType reflect.Type, props *apiextensionsv1beta1.JSONSchemaProps) field.ErrorList {
	var expectedType string
	switch fldType.Kind() {
	case reflect.Struct:
		return verifyStructProperties(path, fldType, props)
	case reflect.Ptr:
		// Pointers are just optional fields in JSON
		return verifyFieldProperties(path, fldType.Elem(), props)
	case reflect.Int32, reflect.Int64:
		expectedType = "integer"
	case reflect.Float32, reflect.Float64:
		expectedType = "number"
	case reflect.String:
		expectedType = "string"
	case reflect.Bool:
		expectedType = "boolean"
	default:
		// TODO: Handle slices & maps.
		panic("Unmapped kind verification: " + fldType.Kind().String())
	}
	if expectedType != props.Type {
		return field.ErrorList{field.Invalid(path, expectedType,
			fmt.Sprintf("type mismatch: typed=%s, CRD=%s", expectedType, props.Type))}
	}
	return nil
}

func jsonTagName(tag reflect.StructTag) string {
	json := tag.Get("json")
	// TODO: This doesn't handle inline structs.
	return strings.Split(json, ",")[0]
}

func contains(s string, ss []string) bool {
	for _, val := range ss {
		if s == val {
			return true
		}
	}
	return false
}
