/*
Copyright 2025 The Kubernetes Authors.

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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/ptr"
)

func testResourceSlicePatch(name string) *resource.ResourceSlicePatch {
	return &resource.ResourceSlicePatch{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}
}

func TestValidateResourceSlicePatch(t *testing.T) {
	goodName := "foo"
	now := metav1.Now()
	badName := "!@#$%^"
	badValue := "spaces not allowed"

	scenarios := map[string]struct {
		patch        *resource.ResourceSlicePatch
		wantFailures field.ErrorList
	}{
		"good-patch": {
			patch: testResourceSlicePatch(goodName),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			patch:        testResourceSlicePatch(""),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			patch:        testResourceSlicePatch(badName),
		},
		"generate-name": {
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.GenerateName = "pvc-"
				return patch
			}(),
		},
		"uid": {
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return patch
			}(),
		},
		"resource-version": {
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.ResourceVersion = "1"
				return patch
			}(),
		},
		"generation": {
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.Generation = 100
				return patch
			}(),
		},
		"creation-timestamp": {
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.CreationTimestamp = now
				return patch
			}(),
		},
		"deletion-grace-period-seconds": {
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.DeletionGracePeriodSeconds = ptr.To(int64(10))
				return patch
			}(),
		},
		"owner-references": {
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "pod",
						Name:       "foo",
						UID:        "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d",
					},
				}
				return patch
			}(),
		},
		"finalizers": {
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.Finalizers = []string{
					"example.com/foo",
				}
				return patch
			}(),
		},
		"managed-fields": {
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						FieldsType: "FieldsV1",
						Operation:  "Apply",
						APIVersion: "apps/v1",
						Manager:    "foo",
					},
				}
				return patch
			}(),
		},
		"good-labels": {
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return patch
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.Labels = map[string]string{
					"hello-world": badValue,
				}
				return patch
			}(),
		},
		"good-annotations": {
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.Annotations = map[string]string{
					"foo": "bar",
				}
				return patch
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			patch: func() *resource.ResourceSlicePatch {
				patch := testResourceSlicePatch(goodName)
				patch.Annotations = map[string]string{
					badName: "hello world",
				}
				return patch
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateResourceSlicePatch(scenario.patch)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateResourceSlicePatchUpdate(t *testing.T) {
	validPatch := testResourceSlicePatch(goodName)

	scenarios := map[string]struct {
		oldPatch     *resource.ResourceSlicePatch
		update       func(patch *resource.ResourceSlicePatch) *resource.ResourceSlicePatch
		wantFailures field.ErrorList
	}{
		"valid-no-op-update": {
			oldPatch: validPatch,
			update:   func(patch *resource.ResourceSlicePatch) *resource.ResourceSlicePatch { return patch },
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldPatch.ResourceVersion = "1"
			errs := ValidateResourceSlicePatchUpdate(scenario.update(scenario.oldPatch.DeepCopy()), scenario.oldPatch)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}
