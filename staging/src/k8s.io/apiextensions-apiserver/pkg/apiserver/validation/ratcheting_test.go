/*
Copyright 2023 The Kubernetes Authors.

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

package validation_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
)

var zeroIntSchema *spec.Schema = &spec.Schema{
	SchemaProps: spec.SchemaProps{
		Type:    spec.StringOrArray{"number"},
		Minimum: ptr(float64(0)),
		Maximum: ptr(float64(0)),
	},
}

var smallIntSchema *spec.Schema = &spec.Schema{
	SchemaProps: spec.SchemaProps{
		Type:    spec.StringOrArray{"number"},
		Maximum: ptr(float64(50)),
	},
}

var mediumIntSchema *spec.Schema = &spec.Schema{
	SchemaProps: spec.SchemaProps{
		Type:    spec.StringOrArray{"number"},
		Minimum: ptr(float64(50)),
		Maximum: ptr(float64(10000)),
	},
}

var largeIntSchema *spec.Schema = &spec.Schema{
	SchemaProps: spec.SchemaProps{
		Type:    spec.StringOrArray{"number"},
		Minimum: ptr(float64(10000)),
	},
}

func TestScalarRatcheting(t *testing.T) {
	validator := validation.NewRatchetingSchemaValidator(mediumIntSchema, nil, "", strfmt.Default)
	require.True(t, validator.ValidateUpdate(1, 1, validation.WithRatcheting(nil)).IsValid())
	require.False(t, validator.ValidateUpdate(1, 2, validation.WithRatcheting(nil)).IsValid())
}

var objectSchema *spec.Schema = &spec.Schema{
	SchemaProps: spec.SchemaProps{
		Type: spec.StringOrArray{"object"},
		Properties: map[string]spec.Schema{
			"zero":   *zeroIntSchema,
			"small":  *smallIntSchema,
			"medium": *mediumIntSchema,
			"large":  *largeIntSchema,
		},
	},
}

var objectObjectSchema *spec.Schema = &spec.Schema{
	SchemaProps: spec.SchemaProps{
		Type: spec.StringOrArray{"object"},
		Properties: map[string]spec.Schema{
			"nested": *objectSchema,
		},
	},
}

// Shows scalar fields of objects can be ratcheted
func TestObjectScalarFieldsRatcheting(t *testing.T) {
	validator := validation.NewRatchetingSchemaValidator(objectSchema, nil, "", strfmt.Default)
	assert.True(t, validator.ValidateUpdate(map[string]interface{}{
		"small": 500,
	}, map[string]interface{}{
		"small": 500,
	}, validation.WithRatcheting(nil)).IsValid())
	assert.True(t, validator.ValidateUpdate(map[string]interface{}{
		"small": 501,
	}, map[string]interface{}{
		"small":  501,
		"medium": 500,
	}, validation.WithRatcheting(nil)).IsValid())
	assert.False(t, validator.ValidateUpdate(map[string]interface{}{
		"small": 500,
	}, map[string]interface{}{
		"small": 501,
	}, validation.WithRatcheting(nil)).IsValid())
}

// Shows schemas with object fields which themselves are ratcheted can be ratcheted
func TestObjectObjectFieldsRatcheting(t *testing.T) {
	validator := validation.NewRatchetingSchemaValidator(objectObjectSchema, nil, "", strfmt.Default)
	assert.True(t, validator.ValidateUpdate(map[string]interface{}{
		"nested": map[string]interface{}{
			"small": 500,
		}}, map[string]interface{}{
		"nested": map[string]interface{}{
			"small": 500,
		}}, validation.WithRatcheting(nil)).IsValid())
	assert.True(t, validator.ValidateUpdate(map[string]interface{}{
		"nested": map[string]interface{}{
			"small": 501,
		}}, map[string]interface{}{
		"nested": map[string]interface{}{
			"small":  501,
			"medium": 500,
		}}, validation.WithRatcheting(nil)).IsValid())
	assert.False(t, validator.ValidateUpdate(map[string]interface{}{
		"nested": map[string]interface{}{
			"small": 500,
		}}, map[string]interface{}{
		"nested": map[string]interface{}{
			"small": 501,
		}}, validation.WithRatcheting(nil)).IsValid())
}

func ptr[T any](v T) *T {
	return &v
}
