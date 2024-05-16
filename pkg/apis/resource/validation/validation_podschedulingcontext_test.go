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

func testPodSchedulingContexts(name, namespace string, spec resource.PodSchedulingContextSpec) *resource.PodSchedulingContext {
	return &resource.PodSchedulingContext{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: spec,
	}
}

func TestValidatePodSchedulingContexts(t *testing.T) {
	goodName := "foo"
	goodNS := "ns"
	goodPodSchedulingSpec := resource.PodSchedulingContextSpec{}
	now := metav1.Now()
	badName := "!@#$%^"
	badValue := "spaces not allowed"

	scenarios := map[string]struct {
		schedulingCtx *resource.PodSchedulingContext
		wantFailures  field.ErrorList
	}{
		"good-schedulingCtx": {
			schedulingCtx: testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec),
		},
		"missing-name": {
			wantFailures:  field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			schedulingCtx: testPodSchedulingContexts("", goodNS, goodPodSchedulingSpec),
		},
		"bad-name": {
			wantFailures:  field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			schedulingCtx: testPodSchedulingContexts(badName, goodNS, goodPodSchedulingSpec),
		},
		"missing-namespace": {
			wantFailures:  field.ErrorList{field.Required(field.NewPath("metadata", "namespace"), "")},
			schedulingCtx: testPodSchedulingContexts(goodName, "", goodPodSchedulingSpec),
		},
		"generate-name": {
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.GenerateName = "pvc-"
				return schedulingCtx
			}(),
		},
		"uid": {
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return schedulingCtx
			}(),
		},
		"resource-version": {
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.ResourceVersion = "1"
				return schedulingCtx
			}(),
		},
		"generation": {
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.Generation = 100
				return schedulingCtx
			}(),
		},
		"creation-timestamp": {
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.CreationTimestamp = now
				return schedulingCtx
			}(),
		},
		"deletion-grace-period-seconds": {
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.DeletionGracePeriodSeconds = pointer.Int64(10)
				return schedulingCtx
			}(),
		},
		"owner-references": {
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "pod",
						Name:       "foo",
						UID:        "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d",
					},
				}
				return schedulingCtx
			}(),
		},
		"finalizers": {
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.Finalizers = []string{
					"example.com/foo",
				}
				return schedulingCtx
			}(),
		},
		"managed-fields": {
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						FieldsType: "FieldsV1",
						Operation:  "Apply",
						APIVersion: "apps/v1",
						Manager:    "foo",
					},
				}
				return schedulingCtx
			}(),
		},
		"good-labels": {
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return schedulingCtx
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.Labels = map[string]string{
					"hello-world": badValue,
				}
				return schedulingCtx
			}(),
		},
		"good-annotations": {
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.Annotations = map[string]string{
					"foo": "bar",
				}
				return schedulingCtx
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			schedulingCtx: func() *resource.PodSchedulingContext {
				schedulingCtx := testPodSchedulingContexts(goodName, goodNS, goodPodSchedulingSpec)
				schedulingCtx.Annotations = map[string]string{
					badName: "hello world",
				}
				return schedulingCtx
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidatePodSchedulingContexts(scenario.schedulingCtx)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidatePodSchedulingUpdate(t *testing.T) {
	validScheduling := testPodSchedulingContexts("foo", "ns", resource.PodSchedulingContextSpec{})
	badName := "!@#$%^"

	scenarios := map[string]struct {
		oldScheduling *resource.PodSchedulingContext
		update        func(schedulingCtx *resource.PodSchedulingContext) *resource.PodSchedulingContext
		wantFailures  field.ErrorList
	}{
		"valid-no-op-update": {
			oldScheduling: validScheduling,
			update: func(schedulingCtx *resource.PodSchedulingContext) *resource.PodSchedulingContext {
				return schedulingCtx
			},
		},
		"add-selected-node": {
			oldScheduling: validScheduling,
			update: func(schedulingCtx *resource.PodSchedulingContext) *resource.PodSchedulingContext {
				schedulingCtx.Spec.SelectedNode = "worker1"
				return schedulingCtx
			},
		},
		"add-potential-nodes": {
			oldScheduling: validScheduling,
			update: func(schedulingCtx *resource.PodSchedulingContext) *resource.PodSchedulingContext {
				for i := 0; i < resource.PodSchedulingNodeListMaxSize; i++ {
					schedulingCtx.Spec.PotentialNodes = append(schedulingCtx.Spec.PotentialNodes, fmt.Sprintf("worker%d", i))
				}
				return schedulingCtx
			},
		},
		"invalid-potential-nodes-too-long": {
			wantFailures:  field.ErrorList{field.TooLongMaxLength(field.NewPath("spec", "potentialNodes"), 129, resource.PodSchedulingNodeListMaxSize)},
			oldScheduling: validScheduling,
			update: func(schedulingCtx *resource.PodSchedulingContext) *resource.PodSchedulingContext {
				for i := 0; i < resource.PodSchedulingNodeListMaxSize+1; i++ {
					schedulingCtx.Spec.PotentialNodes = append(schedulingCtx.Spec.PotentialNodes, fmt.Sprintf("worker%d", i))
				}
				return schedulingCtx
			},
		},
		"invalid-potential-nodes-name": {
			wantFailures:  field.ErrorList{field.Invalid(field.NewPath("spec", "potentialNodes").Index(0), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			oldScheduling: validScheduling,
			update: func(schedulingCtx *resource.PodSchedulingContext) *resource.PodSchedulingContext {
				schedulingCtx.Spec.PotentialNodes = append(schedulingCtx.Spec.PotentialNodes, badName)
				return schedulingCtx
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldScheduling.ResourceVersion = "1"
			errs := ValidatePodSchedulingContextUpdate(scenario.update(scenario.oldScheduling.DeepCopy()), scenario.oldScheduling)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidatePodSchedulingStatusUpdate(t *testing.T) {
	validScheduling := testPodSchedulingContexts("foo", "ns", resource.PodSchedulingContextSpec{})
	badName := "!@#$%^"

	scenarios := map[string]struct {
		oldScheduling *resource.PodSchedulingContext
		update        func(schedulingCtx *resource.PodSchedulingContext) *resource.PodSchedulingContext
		wantFailures  field.ErrorList
	}{
		"valid-no-op-update": {
			oldScheduling: validScheduling,
			update: func(schedulingCtx *resource.PodSchedulingContext) *resource.PodSchedulingContext {
				return schedulingCtx
			},
		},
		"add-claim-status": {
			oldScheduling: validScheduling,
			update: func(schedulingCtx *resource.PodSchedulingContext) *resource.PodSchedulingContext {
				schedulingCtx.Status.ResourceClaims = append(schedulingCtx.Status.ResourceClaims,
					resource.ResourceClaimSchedulingStatus{
						Name: "my-claim",
					},
				)
				for i := 0; i < resource.PodSchedulingNodeListMaxSize; i++ {
					schedulingCtx.Status.ResourceClaims[0].UnsuitableNodes = append(
						schedulingCtx.Status.ResourceClaims[0].UnsuitableNodes,
						fmt.Sprintf("worker%d", i),
					)
				}
				return schedulingCtx
			},
		},
		"invalid-duplicated-claim-status": {
			wantFailures:  field.ErrorList{field.Duplicate(field.NewPath("status", "claims").Index(1), "my-claim")},
			oldScheduling: validScheduling,
			update: func(schedulingCtx *resource.PodSchedulingContext) *resource.PodSchedulingContext {
				for i := 0; i < 2; i++ {
					schedulingCtx.Status.ResourceClaims = append(schedulingCtx.Status.ResourceClaims,
						resource.ResourceClaimSchedulingStatus{Name: "my-claim"},
					)
				}
				return schedulingCtx
			},
		},
		"invalid-too-long-claim-status": {
			wantFailures:  field.ErrorList{field.TooLongMaxLength(field.NewPath("status", "claims").Index(0).Child("unsuitableNodes"), 129, resource.PodSchedulingNodeListMaxSize)},
			oldScheduling: validScheduling,
			update: func(schedulingCtx *resource.PodSchedulingContext) *resource.PodSchedulingContext {
				schedulingCtx.Status.ResourceClaims = append(schedulingCtx.Status.ResourceClaims,
					resource.ResourceClaimSchedulingStatus{
						Name: "my-claim",
					},
				)
				for i := 0; i < resource.PodSchedulingNodeListMaxSize+1; i++ {
					schedulingCtx.Status.ResourceClaims[0].UnsuitableNodes = append(
						schedulingCtx.Status.ResourceClaims[0].UnsuitableNodes,
						fmt.Sprintf("worker%d", i),
					)
				}
				return schedulingCtx
			},
		},
		"invalid-node-name": {
			wantFailures:  field.ErrorList{field.Invalid(field.NewPath("status", "claims").Index(0).Child("unsuitableNodes").Index(0), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			oldScheduling: validScheduling,
			update: func(schedulingCtx *resource.PodSchedulingContext) *resource.PodSchedulingContext {
				schedulingCtx.Status.ResourceClaims = append(schedulingCtx.Status.ResourceClaims,
					resource.ResourceClaimSchedulingStatus{
						Name: "my-claim",
					},
				)
				schedulingCtx.Status.ResourceClaims[0].UnsuitableNodes = append(
					schedulingCtx.Status.ResourceClaims[0].UnsuitableNodes,
					badName,
				)
				return schedulingCtx
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldScheduling.ResourceVersion = "1"
			errs := ValidatePodSchedulingContextStatusUpdate(scenario.update(scenario.oldScheduling.DeepCopy()), scenario.oldScheduling)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}
