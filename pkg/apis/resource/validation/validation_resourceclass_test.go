/*
Copyright 2022 The Kubernetes Authors.

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

package validation

import (
	"testing"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/pointer"
)

func testClass(name, driverName string) *resource.ResourceClass {
	return &resource.ResourceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		DriverName: driverName,
	}
}

func TestValidateClass(t *testing.T) {
	goodName := "foo"
	now := metav1.Now()
	goodParameters := resource.ResourceClassParametersReference{
		Name:      "valid",
		Namespace: "valid",
		Kind:      "foo",
	}
	badName := "!@#$%^"
	badValue := "spaces not allowed"
	badAPIGroup := "example.com/v1"
	goodAPIGroup := "example.com"

	scenarios := map[string]struct {
		class        *resource.ResourceClass
		wantFailures field.ErrorList
	}{
		"good-class": {
			class: testClass(goodName, goodName),
		},
		"good-long-driver-name": {
			class: testClass(goodName, "acme.example.com"),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			class:        testClass("", goodName),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			class:        testClass(badName, goodName),
		},
		"generate-name": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.GenerateName = "pvc-"
				return class
			}(),
		},
		"uid": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return class
			}(),
		},
		"resource-version": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.ResourceVersion = "1"
				return class
			}(),
		},
		"generation": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.Generation = 100
				return class
			}(),
		},
		"creation-timestamp": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.CreationTimestamp = now
				return class
			}(),
		},
		"deletion-grace-period-seconds": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.DeletionGracePeriodSeconds = pointer.Int64(10)
				return class
			}(),
		},
		"owner-references": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "pod",
						Name:       "foo",
						UID:        "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d",
					},
				}
				return class
			}(),
		},
		"finalizers": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.Finalizers = []string{
					"example.com/foo",
				}
				return class
			}(),
		},
		"managed-fields": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						FieldsType: "FieldsV1",
						Operation:  "Apply",
						APIVersion: "apps/v1",
						Manager:    "foo",
					},
				}
				return class
			}(),
		},
		"good-labels": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return class
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.Labels = map[string]string{
					"hello-world": badValue,
				}
				return class
			}(),
		},
		"good-annotations": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.Annotations = map[string]string{
					"foo": "bar",
				}
				return class
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.Annotations = map[string]string{
					badName: "hello world",
				}
				return class
			}(),
		},
		"missing-driver-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("driverName"), ""),
				field.Invalid(field.NewPath("driverName"), "", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
			class: testClass(goodName, ""),
		},
		"invalid-driver-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("driverName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			class:        testClass(goodName, badName),
		},
		"invalid-qualified-driver-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("driverName"), goodName+"/path", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			class:        testClass(goodName, goodName+"/path"),
		},
		"good-parameters": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.ParametersRef = goodParameters.DeepCopy()
				return class
			}(),
		},
		"good-parameters-apigroup": {
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.ParametersRef = goodParameters.DeepCopy()
				class.ParametersRef.APIGroup = goodAPIGroup
				return class
			}(),
		},
		"bad-parameters-apigroup": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("parametersRef", "apiGroup"), badAPIGroup, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.ParametersRef = goodParameters.DeepCopy()
				class.ParametersRef.APIGroup = badAPIGroup
				return class
			}(),
		},
		"missing-parameters-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("parametersRef", "name"), "")},
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.ParametersRef = goodParameters.DeepCopy()
				class.ParametersRef.Name = ""
				return class
			}(),
		},
		"bad-parameters-namespace": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("parametersRef", "namespace"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')")},
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.ParametersRef = goodParameters.DeepCopy()
				class.ParametersRef.Namespace = badName
				return class
			}(),
		},
		"missing-parameters-kind": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("parametersRef", "kind"), "")},
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.ParametersRef = goodParameters.DeepCopy()
				class.ParametersRef.Kind = ""
				return class
			}(),
		},
		"invalid-node-selector": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("suitableNodes", "nodeSelectorTerms"), "must have at least one node selector term")},
			class: func() *resource.ResourceClass {
				class := testClass(goodName, goodName)
				class.SuitableNodes = &core.NodeSelector{
					// Must not be empty.
				}
				return class
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateClass(scenario.class)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateClassUpdate(t *testing.T) {
	validClass := testClass("foo", "valid")

	scenarios := map[string]struct {
		oldClass     *resource.ResourceClass
		update       func(class *resource.ResourceClass) *resource.ResourceClass
		wantFailures field.ErrorList
	}{
		"valid-no-op-update": {
			oldClass: validClass,
			update:   func(class *resource.ResourceClass) *resource.ResourceClass { return class },
		},
		"update-driver": {
			oldClass: validClass,
			update: func(class *resource.ResourceClass) *resource.ResourceClass {
				class.DriverName += "2"
				return class
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldClass.ResourceVersion = "1"
			errs := ValidateClassUpdate(scenario.update(scenario.oldClass.DeepCopy()), scenario.oldClass)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}
