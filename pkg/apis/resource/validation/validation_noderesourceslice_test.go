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
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/ptr"
)

func testNodeResourceSlice(name, nodeName, driverName string) *resource.NodeResourceSlice {
	return &resource.NodeResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		NodeName:   nodeName,
		DriverName: driverName,
		NodeResourceModel: resource.NodeResourceModel{
			NamedResources: &resource.NamedResourcesResources{},
		},
	}
}

func TestValidateNodeResourceSlice(t *testing.T) {
	goodName := "foo"
	badName := "!@#$%^"
	driverName := "test.example.com"
	now := metav1.Now()
	badValue := "spaces not allowed"

	scenarios := map[string]struct {
		slice        *resource.NodeResourceSlice
		wantFailures field.ErrorList
	}{
		"good": {
			slice: testNodeResourceSlice(goodName, goodName, driverName),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			slice:        testNodeResourceSlice("", goodName, driverName),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			slice:        testNodeResourceSlice(badName, goodName, driverName),
		},
		"generate-name": {
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.GenerateName = "prefix-"
				return slice
			}(),
		},
		"uid": {
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return slice
			}(),
		},
		"resource-version": {
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.ResourceVersion = "1"
				return slice
			}(),
		},
		"generation": {
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.Generation = 100
				return slice
			}(),
		},
		"creation-timestamp": {
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.CreationTimestamp = now
				return slice
			}(),
		},
		"deletion-grace-period-seconds": {
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.DeletionGracePeriodSeconds = ptr.To[int64](10)
				return slice
			}(),
		},
		"owner-references": {
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "pod",
						Name:       "foo",
						UID:        "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d",
					},
				}
				return slice
			}(),
		},
		"finalizers": {
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.Finalizers = []string{
					"example.com/foo",
				}
				return slice
			}(),
		},
		"managed-fields": {
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						FieldsType: "FieldsV1",
						Operation:  "Apply",
						APIVersion: "apps/v1",
						Manager:    "foo",
					},
				}
				return slice
			}(),
		},
		"good-labels": {
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return slice
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.Labels = map[string]string{
					"hello-world": badValue,
				}
				return slice
			}(),
		},
		"good-annotations": {
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.Annotations = map[string]string{
					"foo": "bar",
				}
				return slice
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.Annotations = map[string]string{
					badName: "hello world",
				}
				return slice
			}(),
		},
		"bad-nodename": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("nodeName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			slice:        testNodeResourceSlice(goodName, badName, driverName),
		},
		"bad-drivername": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("driverName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			slice:        testNodeResourceSlice(goodName, goodName, badName),
		},

		"empty-model": {
			wantFailures: field.ErrorList{field.Required(nil, "exactly one structured model field must be set")},
			slice: func() *resource.NodeResourceSlice {
				slice := testNodeResourceSlice(goodName, goodName, driverName)
				slice.NodeResourceModel = resource.NodeResourceModel{}
				return slice
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateNodeResourceSlice(scenario.slice)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateNodeResourceSliceUpdate(t *testing.T) {
	name := "valid"
	validNodeResourceSlice := testNodeResourceSlice(name, name, name)

	scenarios := map[string]struct {
		oldNodeResourceSlice *resource.NodeResourceSlice
		update               func(slice *resource.NodeResourceSlice) *resource.NodeResourceSlice
		wantFailures         field.ErrorList
	}{
		"valid-no-op-update": {
			oldNodeResourceSlice: validNodeResourceSlice,
			update:               func(slice *resource.NodeResourceSlice) *resource.NodeResourceSlice { return slice },
		},
		"invalid-name-update": {
			oldNodeResourceSlice: validNodeResourceSlice,
			update: func(slice *resource.NodeResourceSlice) *resource.NodeResourceSlice {
				slice.Name += "-update"
				return slice
			},
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), name+"-update", "field is immutable")},
		},
		"invalid-update-nodename": {
			wantFailures:         field.ErrorList{field.Invalid(field.NewPath("nodeName"), name+"-updated", "field is immutable")},
			oldNodeResourceSlice: validNodeResourceSlice,
			update: func(slice *resource.NodeResourceSlice) *resource.NodeResourceSlice {
				slice.NodeName += "-updated"
				return slice
			},
		},
		"invalid-update-drivername": {
			wantFailures:         field.ErrorList{field.Invalid(field.NewPath("driverName"), name+"-updated", "field is immutable")},
			oldNodeResourceSlice: validNodeResourceSlice,
			update: func(slice *resource.NodeResourceSlice) *resource.NodeResourceSlice {
				slice.DriverName += "-updated"
				return slice
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldNodeResourceSlice.ResourceVersion = "1"
			errs := ValidateNodeResourceSliceUpdate(scenario.update(scenario.oldNodeResourceSlice.DeepCopy()), scenario.oldNodeResourceSlice)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}
