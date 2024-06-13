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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/pointer"
)

func testClaim(name, namespace string, spec resource.ResourceClaimSpec) *resource.ResourceClaim {
	return &resource.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: spec,
	}
}

func TestValidateClaim(t *testing.T) {
	goodName := "foo"
	badName := "!@#$%^"
	goodNS := "ns"
	goodClaimSpec := resource.ResourceClaimSpec{
		ResourceClassName: goodName,
	}
	now := metav1.Now()
	badValue := "spaces not allowed"
	badAPIGroup := "example.com/v1"
	goodAPIGroup := "example.com"

	scenarios := map[string]struct {
		claim        *resource.ResourceClaim
		wantFailures field.ErrorList
	}{
		"good-claim": {
			claim: testClaim(goodName, goodNS, goodClaimSpec),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			claim:        testClaim("", goodNS, goodClaimSpec),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			claim:        testClaim(badName, goodNS, goodClaimSpec),
		},
		"missing-namespace": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "namespace"), "")},
			claim:        testClaim(goodName, "", goodClaimSpec),
		},
		"generate-name": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.GenerateName = "pvc-"
				return claim
			}(),
		},
		"uid": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return claim
			}(),
		},
		"resource-version": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.ResourceVersion = "1"
				return claim
			}(),
		},
		"generation": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.Generation = 100
				return claim
			}(),
		},
		"creation-timestamp": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.CreationTimestamp = now
				return claim
			}(),
		},
		"deletion-grace-period-seconds": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.DeletionGracePeriodSeconds = pointer.Int64(10)
				return claim
			}(),
		},
		"owner-references": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "pod",
						Name:       "foo",
						UID:        "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d",
					},
				}
				return claim
			}(),
		},
		"finalizers": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.Finalizers = []string{
					"example.com/foo",
				}
				return claim
			}(),
		},
		"managed-fields": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						FieldsType: "FieldsV1",
						Operation:  "Apply",
						APIVersion: "apps/v1",
						Manager:    "foo",
					},
				}
				return claim
			}(),
		},
		"good-labels": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return claim
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.Labels = map[string]string{
					"hello-world": badValue,
				}
				return claim
			}(),
		},
		"good-annotations": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.Annotations = map[string]string{
					"foo": "bar",
				}
				return claim
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.Annotations = map[string]string{
					badName: "hello world",
				}
				return claim
			}(),
		},
		"bad-classname": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "resourceClassName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.Spec.ResourceClassName = badName
				return claim
			}(),
		},
		"good-parameters": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.Spec.ParametersRef = &resource.ResourceClaimParametersReference{
					Kind: "foo",
					Name: "bar",
				}
				return claim
			}(),
		},
		"good-parameters-apigroup": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.Spec.ParametersRef = &resource.ResourceClaimParametersReference{
					APIGroup: goodAPIGroup,
					Kind:     "foo",
					Name:     "bar",
				}
				return claim
			}(),
		},
		"bad-parameters-apigroup": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "parametersRef", "apiGroup"), badAPIGroup, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.Spec.ParametersRef = &resource.ResourceClaimParametersReference{
					APIGroup: badAPIGroup,
					Kind:     "foo",
					Name:     "bar",
				}
				return claim
			}(),
		},
		"missing-parameters-kind": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("spec", "parametersRef", "kind"), "")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.Spec.ParametersRef = &resource.ResourceClaimParametersReference{
					Name: "bar",
				}
				return claim
			}(),
		},
		"missing-parameters-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("spec", "parametersRef", "name"), "")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, goodClaimSpec)
				claim.Spec.ParametersRef = &resource.ResourceClaimParametersReference{
					Kind: "foo",
				}
				return claim
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateClaim(scenario.claim)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateClaimUpdate(t *testing.T) {
	name := "valid"
	parameters := &resource.ResourceClaimParametersReference{
		Kind: "foo",
		Name: "bar",
	}
	validClaim := testClaim("foo", "ns", resource.ResourceClaimSpec{
		ResourceClassName: name,
		ParametersRef:     parameters,
	})

	scenarios := map[string]struct {
		oldClaim     *resource.ResourceClaim
		update       func(claim *resource.ResourceClaim) *resource.ResourceClaim
		wantFailures field.ErrorList
	}{
		"valid-no-op-update": {
			oldClaim: validClaim,
			update:   func(claim *resource.ResourceClaim) *resource.ResourceClaim { return claim },
		},
		"invalid-update-class": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec"), func() resource.ResourceClaimSpec {
				spec := validClaim.Spec.DeepCopy()
				spec.ResourceClassName += "2"
				return *spec
			}(), "field is immutable")},
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Spec.ResourceClassName += "2"
				return claim
			},
		},
		"invalid-update-remove-parameters": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec"), func() resource.ResourceClaimSpec {
				spec := validClaim.Spec.DeepCopy()
				spec.ParametersRef = nil
				return *spec
			}(), "field is immutable")},
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Spec.ParametersRef = nil
				return claim
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldClaim.ResourceVersion = "1"
			errs := ValidateClaimUpdate(scenario.update(scenario.oldClaim.DeepCopy()), scenario.oldClaim)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateClaimStatusUpdate(t *testing.T) {
	invalidName := "!@#$%^"
	validClaim := testClaim("foo", "ns", resource.ResourceClaimSpec{
		ResourceClassName: "valid",
	})

	validAllocatedClaim := validClaim.DeepCopy()
	validAllocatedClaim.Status = resource.ResourceClaimStatus{
		DriverName: "valid",
		Allocation: &resource.AllocationResult{
			ResourceHandles: func() []resource.ResourceHandle {
				var handles []resource.ResourceHandle
				for i := 0; i < resource.AllocationResultResourceHandlesMaxSize; i++ {
					handle := resource.ResourceHandle{
						DriverName: "valid",
						Data:       strings.Repeat(" ", resource.ResourceHandleDataMaxSize),
					}
					handles = append(handles, handle)
				}
				return handles
			}(),
			Shareable: true,
		},
	}

	scenarios := map[string]struct {
		oldClaim     *resource.ResourceClaim
		update       func(claim *resource.ResourceClaim) *resource.ResourceClaim
		wantFailures field.ErrorList
	}{
		"valid-no-op-update": {
			oldClaim: validClaim,
			update:   func(claim *resource.ResourceClaim) *resource.ResourceClaim { return claim },
		},
		"add-driver": {
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DriverName = "valid"
				return claim
			},
		},
		"invalid-add-allocation": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("status", "driverName"), "must be specified when `allocation` is set")},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				// DriverName must also get set here!
				claim.Status.Allocation = &resource.AllocationResult{}
				return claim
			},
		},
		"valid-add-allocation": {
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DriverName = "valid"
				claim.Status.Allocation = &resource.AllocationResult{
					ResourceHandles: []resource.ResourceHandle{
						{
							DriverName: "valid",
							Data:       strings.Repeat(" ", resource.ResourceHandleDataMaxSize),
						},
					},
				}
				return claim
			},
		},
		"valid-add-empty-allocation-structured": {
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DriverName = "valid"
				claim.Status.Allocation = &resource.AllocationResult{
					ResourceHandles: []resource.ResourceHandle{
						{
							DriverName:     "valid",
							StructuredData: &resource.StructuredResourceHandle{},
						},
					},
				}
				return claim
			},
		},
		"valid-add-allocation-structured": {
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DriverName = "valid"
				claim.Status.Allocation = &resource.AllocationResult{
					ResourceHandles: []resource.ResourceHandle{
						{
							DriverName: "valid",
							StructuredData: &resource.StructuredResourceHandle{
								NodeName: "worker",
							},
						},
					},
				}
				return claim
			},
		},
		"invalid-add-allocation-structured": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "resourceHandles").Index(0).Child("structuredData", "nodeName"), "&^!", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
				field.Required(field.NewPath("status", "allocation", "resourceHandles").Index(0).Child("structuredData", "results").Index(1), "exactly one structured model field must be set"),
			},
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DriverName = "valid"
				claim.Status.Allocation = &resource.AllocationResult{
					ResourceHandles: []resource.ResourceHandle{
						{
							DriverName: "valid",
							StructuredData: &resource.StructuredResourceHandle{
								NodeName: "&^!",
								Results: []resource.DriverAllocationResult{
									{
										AllocationResultModel: resource.AllocationResultModel{
											NamedResources: &resource.NamedResourcesAllocationResult{
												Name: "some-resource-instance",
											},
										},
									},
									{
										AllocationResultModel: resource.AllocationResultModel{}, // invalid
									},
								},
							},
						},
					},
				}
				return claim
			},
		},
		"invalid-duplicated-data": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("status", "allocation", "resourceHandles").Index(0), nil, "data and structuredData are mutually exclusive")},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DriverName = "valid"
				claim.Status.Allocation = &resource.AllocationResult{
					ResourceHandles: []resource.ResourceHandle{
						{
							DriverName: "valid",
							Data:       "something",
							StructuredData: &resource.StructuredResourceHandle{
								NodeName: "worker",
							},
						},
					},
				}
				return claim
			},
		},
		"invalid-allocation-resourceHandles": {
			wantFailures: field.ErrorList{field.TooLongMaxLength(field.NewPath("status", "allocation", "resourceHandles"), resource.AllocationResultResourceHandlesMaxSize+1, resource.AllocationResultResourceHandlesMaxSize)},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DriverName = "valid"
				claim.Status.Allocation = &resource.AllocationResult{
					ResourceHandles: func() []resource.ResourceHandle {
						var handles []resource.ResourceHandle
						for i := 0; i < resource.AllocationResultResourceHandlesMaxSize+1; i++ {
							handles = append(handles, resource.ResourceHandle{DriverName: "valid"})
						}
						return handles
					}(),
				}
				return claim
			},
		},
		"invalid-allocation-resource-handle-drivername": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("status", "allocation", "resourceHandles[0]", "driverName"), invalidName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DriverName = "valid"
				claim.Status.Allocation = &resource.AllocationResult{
					ResourceHandles: []resource.ResourceHandle{
						{
							DriverName: invalidName,
						},
					},
				}
				return claim
			},
		},
		"invalid-allocation-resource-handle-data": {
			wantFailures: field.ErrorList{field.TooLongMaxLength(field.NewPath("status", "allocation", "resourceHandles").Index(0).Child("data"), resource.ResourceHandleDataMaxSize+1, resource.ResourceHandleDataMaxSize)},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DriverName = "valid"
				claim.Status.Allocation = &resource.AllocationResult{
					ResourceHandles: []resource.ResourceHandle{
						{
							DriverName: "valid",
							Data:       strings.Repeat(" ", resource.ResourceHandleDataMaxSize+1),
						},
					},
				}
				return claim
			},
		},
		"invalid-node-selector": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("status", "allocation", "availableOnNodes", "nodeSelectorTerms"), "must have at least one node selector term")},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DriverName = "valid"
				claim.Status.Allocation = &resource.AllocationResult{
					AvailableOnNodes: &core.NodeSelector{
						// Must not be empty.
					},
				}
				return claim
			},
		},
		"add-reservation": {
			oldClaim: validAllocatedClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				for i := 0; i < resource.ResourceClaimReservedForMaxSize; i++ {
					claim.Status.ReservedFor = append(claim.Status.ReservedFor,
						resource.ResourceClaimConsumerReference{
							Resource: "pods",
							Name:     fmt.Sprintf("foo-%d", i),
							UID:      types.UID(fmt.Sprintf("%d", i)),
						})
				}
				return claim
			},
		},
		"add-reservation-and-allocation": {
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status = *validAllocatedClaim.Status.DeepCopy()
				for i := 0; i < resource.ResourceClaimReservedForMaxSize; i++ {
					claim.Status.ReservedFor = append(claim.Status.ReservedFor,
						resource.ResourceClaimConsumerReference{
							Resource: "pods",
							Name:     fmt.Sprintf("foo-%d", i),
							UID:      types.UID(fmt.Sprintf("%d", i)),
						})
				}
				return claim
			},
		},
		"invalid-reserved-for-too-large": {
			wantFailures: field.ErrorList{field.TooLongMaxLength(field.NewPath("status", "reservedFor"), resource.ResourceClaimReservedForMaxSize+1, resource.ResourceClaimReservedForMaxSize)},
			oldClaim:     validAllocatedClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				for i := 0; i < resource.ResourceClaimReservedForMaxSize+1; i++ {
					claim.Status.ReservedFor = append(claim.Status.ReservedFor,
						resource.ResourceClaimConsumerReference{
							Resource: "pods",
							Name:     fmt.Sprintf("foo-%d", i),
							UID:      types.UID(fmt.Sprintf("%d", i)),
						})
				}
				return claim
			},
		},
		"invalid-reserved-for-duplicate": {
			wantFailures: field.ErrorList{field.Duplicate(field.NewPath("status", "reservedFor").Index(1).Child("uid"), types.UID("1"))},
			oldClaim:     validAllocatedClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				for i := 0; i < 2; i++ {
					claim.Status.ReservedFor = append(claim.Status.ReservedFor,
						resource.ResourceClaimConsumerReference{
							Resource: "pods",
							Name:     "foo",
							UID:      "1",
						})
				}
				return claim
			},
		},
		"invalid-reserved-for-not-shared": {
			wantFailures: field.ErrorList{field.Forbidden(field.NewPath("status", "reservedFor"), "may not be reserved more than once")},
			oldClaim: func() *resource.ResourceClaim {
				claim := validAllocatedClaim.DeepCopy()
				claim.Status.Allocation.Shareable = false
				return claim
			}(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				for i := 0; i < 2; i++ {
					claim.Status.ReservedFor = append(claim.Status.ReservedFor,
						resource.ResourceClaimConsumerReference{
							Resource: "pods",
							Name:     fmt.Sprintf("foo-%d", i),
							UID:      types.UID(fmt.Sprintf("%d", i)),
						})
				}
				return claim
			},
		},
		"invalid-reserved-for-no-allocation": {
			wantFailures: field.ErrorList{field.Forbidden(field.NewPath("status", "reservedFor"), "may not be specified when `allocated` is not set")},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DriverName = "valid"
				claim.Status.ReservedFor = []resource.ResourceClaimConsumerReference{
					{
						Resource: "pods",
						Name:     "foo",
						UID:      "1",
					},
				}
				return claim
			},
		},
		"invalid-reserved-for-no-resource": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("status", "reservedFor").Index(0).Child("resource"), "")},
			oldClaim:     validAllocatedClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.ReservedFor = []resource.ResourceClaimConsumerReference{
					{
						Name: "foo",
						UID:  "1",
					},
				}
				return claim
			},
		},
		"invalid-reserved-for-no-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("status", "reservedFor").Index(0).Child("name"), "")},
			oldClaim:     validAllocatedClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.ReservedFor = []resource.ResourceClaimConsumerReference{
					{
						Resource: "pods",
						UID:      "1",
					},
				}
				return claim
			},
		},
		"invalid-reserved-for-no-uid": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("status", "reservedFor").Index(0).Child("uid"), "")},
			oldClaim:     validAllocatedClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.ReservedFor = []resource.ResourceClaimConsumerReference{
					{
						Resource: "pods",
						Name:     "foo",
					},
				}
				return claim
			},
		},
		"invalid-reserved-deleted": {
			wantFailures: field.ErrorList{field.Forbidden(field.NewPath("status", "reservedFor"), "new entries may not be added while `deallocationRequested` or `deletionTimestamp` are set")},
			oldClaim: func() *resource.ResourceClaim {
				claim := validAllocatedClaim.DeepCopy()
				var deletionTimestamp metav1.Time
				claim.DeletionTimestamp = &deletionTimestamp
				return claim
			}(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.ReservedFor = []resource.ResourceClaimConsumerReference{
					{
						Resource: "pods",
						Name:     "foo",
						UID:      "1",
					},
				}
				return claim
			},
		},
		"invalid-reserved-deallocation-requested": {
			wantFailures: field.ErrorList{field.Forbidden(field.NewPath("status", "reservedFor"), "new entries may not be added while `deallocationRequested` or `deletionTimestamp` are set")},
			oldClaim: func() *resource.ResourceClaim {
				claim := validAllocatedClaim.DeepCopy()
				claim.Status.DeallocationRequested = true
				return claim
			}(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.ReservedFor = []resource.ResourceClaimConsumerReference{
					{
						Resource: "pods",
						Name:     "foo",
						UID:      "1",
					},
				}
				return claim
			},
		},
		"add-deallocation-requested": {
			oldClaim: validAllocatedClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DeallocationRequested = true
				return claim
			},
		},
		"remove-allocation": {
			oldClaim: func() *resource.ResourceClaim {
				claim := validAllocatedClaim.DeepCopy()
				claim.Status.DeallocationRequested = true
				return claim
			}(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DeallocationRequested = false
				claim.Status.Allocation = nil
				return claim
			},
		},
		"invalid-deallocation-requested-removal": {
			wantFailures: field.ErrorList{field.Forbidden(field.NewPath("status", "deallocationRequested"), "may not be cleared when `allocation` is set")},
			oldClaim: func() *resource.ResourceClaim {
				claim := validAllocatedClaim.DeepCopy()
				claim.Status.DeallocationRequested = true
				return claim
			}(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DeallocationRequested = false
				return claim
			},
		},
		"invalid-allocation-modification": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("status.allocation"), func() *resource.AllocationResult {
				claim := validAllocatedClaim.DeepCopy()
				claim.Status.Allocation.ResourceHandles = []resource.ResourceHandle{
					{
						DriverName: "valid",
						Data:       strings.Repeat(" ", resource.ResourceHandleDataMaxSize/2),
					},
				}
				return claim.Status.Allocation
			}(), "field is immutable")},
			oldClaim: func() *resource.ResourceClaim {
				claim := validAllocatedClaim.DeepCopy()
				claim.Status.DeallocationRequested = false
				return claim
			}(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation.ResourceHandles = []resource.ResourceHandle{
					{
						DriverName: "valid",
						Data:       strings.Repeat(" ", resource.ResourceHandleDataMaxSize/2),
					},
				}
				return claim
			},
		},
		"invalid-deallocation-requested-in-use": {
			wantFailures: field.ErrorList{field.Forbidden(field.NewPath("status", "deallocationRequested"), "deallocation cannot be requested while `reservedFor` is set")},
			oldClaim: func() *resource.ResourceClaim {
				claim := validAllocatedClaim.DeepCopy()
				claim.Status.ReservedFor = []resource.ResourceClaimConsumerReference{
					{
						Resource: "pods",
						Name:     "foo",
						UID:      "1",
					},
				}
				return claim
			}(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DeallocationRequested = true
				return claim
			},
		},
		"invalid-deallocation-not-allocated": {
			wantFailures: field.ErrorList{field.Forbidden(field.NewPath("status"), "`allocation` must be set when `deallocationRequested` is set")},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.DeallocationRequested = true
				return claim
			},
		},
		"invalid-allocation-removal-not-reset": {
			wantFailures: field.ErrorList{field.Forbidden(field.NewPath("status"), "`allocation` must be set when `deallocationRequested` is set")},
			oldClaim: func() *resource.ResourceClaim {
				claim := validAllocatedClaim.DeepCopy()
				claim.Status.DeallocationRequested = true
				return claim
			}(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = nil
				return claim
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldClaim.ResourceVersion = "1"
			errs := ValidateClaimStatusUpdate(scenario.update(scenario.oldClaim.DeepCopy()), scenario.oldClaim)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}
