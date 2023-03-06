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
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/pointer"
)

func testPodSchedulingHints(name, namespace string, spec resource.PodSchedulingHintsSpec) *resource.PodSchedulingHints {
	return &resource.PodSchedulingHints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: spec,
	}
}

func TestValidatePodSchedulingHints(t *testing.T) {
	goodName := "foo"
	goodNS := "ns"
	goodPodSchedulingSpec := resource.PodSchedulingHintsSpec{}
	now := metav1.Now()
	badName := "!@#$%^"
	badValue := "spaces not allowed"

	scenarios := map[string]struct {
		hints        *resource.PodSchedulingHints
		wantFailures field.ErrorList
	}{
		"good-hints": {
			hints: testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			hints:        testPodSchedulingHints("", goodNS, goodPodSchedulingSpec),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			hints:        testPodSchedulingHints(badName, goodNS, goodPodSchedulingSpec),
		},
		"missing-namespace": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "namespace"), "")},
			hints:        testPodSchedulingHints(goodName, "", goodPodSchedulingSpec),
		},
		"generate-name": {
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.GenerateName = "pvc-"
				return hints
			}(),
		},
		"uid": {
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return hints
			}(),
		},
		"resource-version": {
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.ResourceVersion = "1"
				return hints
			}(),
		},
		"generation": {
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.Generation = 100
				return hints
			}(),
		},
		"creation-timestamp": {
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.CreationTimestamp = now
				return hints
			}(),
		},
		"deletion-grace-period-seconds": {
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.DeletionGracePeriodSeconds = pointer.Int64(10)
				return hints
			}(),
		},
		"owner-references": {
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "pod",
						Name:       "foo",
						UID:        "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d",
					},
				}
				return hints
			}(),
		},
		"finalizers": {
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.Finalizers = []string{
					"example.com/foo",
				}
				return hints
			}(),
		},
		"managed-fields": {
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						FieldsType: "FieldsV1",
						Operation:  "Apply",
						APIVersion: "apps/v1",
						Manager:    "foo",
					},
				}
				return hints
			}(),
		},
		"good-labels": {
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return hints
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.Labels = map[string]string{
					"hello-world": badValue,
				}
				return hints
			}(),
		},
		"good-annotations": {
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.Annotations = map[string]string{
					"foo": "bar",
				}
				return hints
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			hints: func() *resource.PodSchedulingHints {
				hints := testPodSchedulingHints(goodName, goodNS, goodPodSchedulingSpec)
				hints.Annotations = map[string]string{
					badName: "hello world",
				}
				return hints
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidatePodSchedulingHints(scenario.hints)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidatePodSchedulingHintsUpdate(t *testing.T) {
	validScheduling := testPodSchedulingHints("foo", "ns", resource.PodSchedulingHintsSpec{})
	badName := "!@#$%^"

	scenarios := map[string]struct {
		oldScheduling *resource.PodSchedulingHints
		update        func(hints *resource.PodSchedulingHints) *resource.PodSchedulingHints
		wantFailures  field.ErrorList
	}{
		"valid-no-op-update": {
			oldScheduling: validScheduling,
			update:        func(hints *resource.PodSchedulingHints) *resource.PodSchedulingHints { return hints },
		},
		"add-selected-node": {
			oldScheduling: validScheduling,
			update: func(hints *resource.PodSchedulingHints) *resource.PodSchedulingHints {
				hints.Spec.SelectedNode = "worker1"
				return hints
			},
		},
		"add-potential-nodes": {
			oldScheduling: validScheduling,
			update: func(hints *resource.PodSchedulingHints) *resource.PodSchedulingHints {
				for i := 0; i < resource.PodSchedulingNodeListMaxSize; i++ {
					hints.Spec.PotentialNodes = append(hints.Spec.PotentialNodes, fmt.Sprintf("worker%d", i))
				}
				return hints
			},
		},
		"invalid-potential-nodes-too-long": {
			wantFailures:  field.ErrorList{field.TooLongMaxLength(field.NewPath("spec", "potentialNodes"), 129, resource.PodSchedulingNodeListMaxSize)},
			oldScheduling: validScheduling,
			update: func(hints *resource.PodSchedulingHints) *resource.PodSchedulingHints {
				for i := 0; i < resource.PodSchedulingNodeListMaxSize+1; i++ {
					hints.Spec.PotentialNodes = append(hints.Spec.PotentialNodes, fmt.Sprintf("worker%d", i))
				}
				return hints
			},
		},
		"invalid-potential-nodes-name": {
			wantFailures:  field.ErrorList{field.Invalid(field.NewPath("spec", "potentialNodes").Index(0), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			oldScheduling: validScheduling,
			update: func(hints *resource.PodSchedulingHints) *resource.PodSchedulingHints {
				hints.Spec.PotentialNodes = append(hints.Spec.PotentialNodes, badName)
				return hints
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldScheduling.ResourceVersion = "1"
			errs := ValidatePodSchedulingHintsUpdate(scenario.update(scenario.oldScheduling.DeepCopy()), scenario.oldScheduling)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidatePodSchedulingHintsStatusUpdate(t *testing.T) {
	validScheduling := testPodSchedulingHints("foo", "ns", resource.PodSchedulingHintsSpec{})
	badName := "!@#$%^"

	scenarios := map[string]struct {
		oldScheduling *resource.PodSchedulingHints
		update        func(hints *resource.PodSchedulingHints) *resource.PodSchedulingHints
		wantFailures  field.ErrorList
	}{
		"valid-no-op-update": {
			oldScheduling: validScheduling,
			update:        func(hints *resource.PodSchedulingHints) *resource.PodSchedulingHints { return hints },
		},
		"add-claim-status": {
			oldScheduling: validScheduling,
			update: func(hints *resource.PodSchedulingHints) *resource.PodSchedulingHints {
				hints.Status.ResourceClaims = append(hints.Status.ResourceClaims,
					resource.ResourceClaimSchedulingStatus{
						Name: "my-claim",
					},
				)
				for i := 0; i < resource.PodSchedulingNodeListMaxSize; i++ {
					hints.Status.ResourceClaims[0].UnsuitableNodes = append(
						hints.Status.ResourceClaims[0].UnsuitableNodes,
						fmt.Sprintf("worker%d", i),
					)
				}
				return hints
			},
		},
		"invalid-duplicated-claim-status": {
			wantFailures:  field.ErrorList{field.Duplicate(field.NewPath("status", "claims").Index(1), "my-claim")},
			oldScheduling: validScheduling,
			update: func(hints *resource.PodSchedulingHints) *resource.PodSchedulingHints {
				for i := 0; i < 2; i++ {
					hints.Status.ResourceClaims = append(hints.Status.ResourceClaims,
						resource.ResourceClaimSchedulingStatus{Name: "my-claim"},
					)
				}
				return hints
			},
		},
		"invalid-too-long-claim-status": {
			wantFailures:  field.ErrorList{field.TooLongMaxLength(field.NewPath("status", "claims").Index(0).Child("unsuitableNodes"), 129, resource.PodSchedulingNodeListMaxSize)},
			oldScheduling: validScheduling,
			update: func(hints *resource.PodSchedulingHints) *resource.PodSchedulingHints {
				hints.Status.ResourceClaims = append(hints.Status.ResourceClaims,
					resource.ResourceClaimSchedulingStatus{
						Name: "my-claim",
					},
				)
				for i := 0; i < resource.PodSchedulingNodeListMaxSize+1; i++ {
					hints.Status.ResourceClaims[0].UnsuitableNodes = append(
						hints.Status.ResourceClaims[0].UnsuitableNodes,
						fmt.Sprintf("worker%d", i),
					)
				}
				return hints
			},
		},
		"invalid-node-name": {
			wantFailures:  field.ErrorList{field.Invalid(field.NewPath("status", "claims").Index(0).Child("unsuitableNodes").Index(0), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			oldScheduling: validScheduling,
			update: func(hints *resource.PodSchedulingHints) *resource.PodSchedulingHints {
				hints.Status.ResourceClaims = append(hints.Status.ResourceClaims,
					resource.ResourceClaimSchedulingStatus{
						Name: "my-claim",
					},
				)
				hints.Status.ResourceClaims[0].UnsuitableNodes = append(
					hints.Status.ResourceClaims[0].UnsuitableNodes,
					badName,
				)
				return hints
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldScheduling.ResourceVersion = "1"
			errs := ValidatePodSchedulingHintsStatusUpdate(scenario.update(scenario.oldScheduling.DeepCopy()), scenario.oldScheduling)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}
