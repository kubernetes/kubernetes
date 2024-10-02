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
	"k8s.io/utils/ptr"
)

func testClass(name string) *resource.DeviceClass {
	return &resource.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}
}

func TestValidateClass(t *testing.T) {
	goodName := "foo"
	now := metav1.Now()
	badName := "!@#$%^"
	badValue := "spaces not allowed"

	scenarios := map[string]struct {
		class        *resource.DeviceClass
		wantFailures field.ErrorList
	}{
		"good-class": {
			class: testClass(goodName),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			class:        testClass(""),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			class:        testClass(badName),
		},
		"generate-name": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.GenerateName = "pvc-"
				return class
			}(),
		},
		"uid": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return class
			}(),
		},
		"resource-version": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.ResourceVersion = "1"
				return class
			}(),
		},
		"generation": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Generation = 100
				return class
			}(),
		},
		"creation-timestamp": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.CreationTimestamp = now
				return class
			}(),
		},
		"deletion-grace-period-seconds": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.DeletionGracePeriodSeconds = ptr.To(int64(10))
				return class
			}(),
		},
		"owner-references": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
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
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Finalizers = []string{
					"example.com/foo",
				}
				return class
			}(),
		},
		"managed-fields": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
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
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return class
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Labels = map[string]string{
					"hello-world": badValue,
				}
				return class
			}(),
		},
		"good-annotations": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Annotations = map[string]string{
					"foo": "bar",
				}
				return class
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Annotations = map[string]string{
					badName: "hello world",
				}
				return class
			}(),
		},
		"invalid-node-selector": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("suitableNodes", "nodeSelectorTerms"), "must have at least one node selector term")},
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Spec.SuitableNodes = &core.NodeSelector{
					// Must not be empty.
				}
				return class
			}(),
		},
		"valid-node-selector": {
			class: func() *resource.DeviceClass {
				class := testClass(goodName)
				class.Spec.SuitableNodes = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{
						MatchExpressions: []core.NodeSelectorRequirement{{
							Key:      "foo",
							Operator: core.NodeSelectorOpDoesNotExist,
						}},
					}},
				}
				return class
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateDeviceClass(scenario.class)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateClassUpdate(t *testing.T) {
	validClass := testClass(goodName)

	scenarios := map[string]struct {
		oldClass     *resource.DeviceClass
		update       func(class *resource.DeviceClass) *resource.DeviceClass
		wantFailures field.ErrorList
	}{
		"valid-no-op-update": {
			oldClass: validClass,
			update:   func(class *resource.DeviceClass) *resource.DeviceClass { return class },
		},
		"update-node-selector": {
			oldClass: validClass,
			update: func(class *resource.DeviceClass) *resource.DeviceClass {
				class = class.DeepCopy()
				class.Spec.SuitableNodes = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{
						MatchExpressions: []core.NodeSelectorRequirement{{
							Key:      "foo",
							Operator: core.NodeSelectorOpDoesNotExist,
						}},
					}},
				}
				return class
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldClass.ResourceVersion = "1"
			errs := ValidateDeviceClassUpdate(scenario.update(scenario.oldClass.DeepCopy()), scenario.oldClass)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}
