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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/pointer"
	"k8s.io/utils/ptr"
)

func testClaim(name, namespace string, spec resource.ResourceClaimSpec) *resource.ResourceClaim {
	return &resource.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: *spec.DeepCopy(),
	}
}

const (
	goodName = "foo"
	badName  = "!@#$%^"
	goodNS   = "ns"
)

var (
	validClaimSpec = resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{{
				Name:            goodName,
				DeviceClassName: goodName,
				AllocationMode:  resource.DeviceAllocationModeExactCount,
				Count:           1,
			}},
		},
	}
	validClaim = testClaim(goodName, goodNS, validClaimSpec)
)

func TestValidateClaim(t *testing.T) {
	now := metav1.Now()
	badValue := "spaces not allowed"

	scenarios := map[string]struct {
		claim        *resource.ResourceClaim
		wantFailures field.ErrorList
	}{
		"good-claim": {
			claim: testClaim(goodName, goodNS, validClaimSpec),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			claim:        testClaim("", goodNS, validClaimSpec),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			claim:        testClaim(badName, goodNS, validClaimSpec),
		},
		"missing-namespace": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "namespace"), "")},
			claim:        testClaim(goodName, "", validClaimSpec),
		},
		"generate-name": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.GenerateName = "pvc-"
				return claim
			}(),
		},
		"uid": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return claim
			}(),
		},
		"resource-version": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.ResourceVersion = "1"
				return claim
			}(),
		},
		"generation": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Generation = 100
				return claim
			}(),
		},
		"creation-timestamp": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.CreationTimestamp = now
				return claim
			}(),
		},
		"deletion-grace-period-seconds": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.DeletionGracePeriodSeconds = pointer.Int64(10)
				return claim
			}(),
		},
		"owner-references": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
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
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Finalizers = []string{
					"example.com/foo",
				}
				return claim
			}(),
		},
		"managed-fields": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
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
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return claim
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Labels = map[string]string{
					"hello-world": badValue,
				}
				return claim
			}(),
		},
		"good-annotations": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Annotations = map[string]string{
					"foo": "bar",
				}
				return claim
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Annotations = map[string]string{
					badName: "hello world",
				}
				return claim
			}(),
		},
		"bad-classname": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "devices", "requests").Index(0).Child("deviceClassName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Requests[0].DeviceClassName = badName
				return claim
			}(),
		},
		"invalid-request-name": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("requests").Index(1), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("requests").Index(1), badName, "must be the name of a request in the claim"),
				field.Invalid(field.NewPath("spec", "devices", "config").Index(0).Child("requests").Index(1), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Invalid(field.NewPath("spec", "devices", "config").Index(0).Child("requests").Index(1), badName, "must be the name of a request in the claim"),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Constraints = []resource.DeviceConstraint{{
					Requests:       []string{claim.Spec.Devices.Requests[0].Name, badName},
					MatchAttribute: ptr.To(resource.FullyQualifiedName("dra.example.com/numa")),
				}}
				claim.Spec.Devices.Config = []resource.DeviceClaimConfiguration{{
					Requests: []string{claim.Spec.Devices.Requests[0].Name, badName},
					DeviceConfiguration: resource.DeviceConfiguration{
						Opaque: &resource.OpaqueDeviceConfiguration{
							Driver: "dra.example.com",
							Parameters: runtime.RawExtension{
								Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
							},
						},
					},
				}}
				return claim
			}(),
		},
		"invalid-config-json": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "devices", "config").Index(0).Child("opaque", "parameters"), ""),
				field.Invalid(field.NewPath("spec", "devices", "config").Index(1).Child("opaque", "parameters"), "<value omitted>", "error parsing data: unexpected end of JSON input"),
				field.Invalid(field.NewPath("spec", "devices", "config").Index(2).Child("opaque", "parameters"), "<value omitted>", "parameters must be a valid JSON object"),
				field.Required(field.NewPath("spec", "devices", "config").Index(3).Child("opaque", "parameters"), ""),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Config = []resource.DeviceClaimConfiguration{
					{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(``),
								},
							},
						},
					},
					{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`{`),
								},
							},
						},
					},
					{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`"hello-world"`),
								},
							},
						},
					},
					{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`null`),
								},
							},
						},
					},
				}
				return claim
			}(),
		},
		"CEL-compile-errors": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "requests").Index(1).Child("selectors").Index(1).Child("cel", "expression"), `device.attributes[true].someBoolean`, "compilation failed: ERROR: <input>:1:18: found no matching overload for '_[_]' applied to '(map(string, map(string, any)), bool)'\n | device.attributes[true].someBoolean\n | .................^"),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, claim.Spec.Devices.Requests[0])
				claim.Spec.Devices.Requests[1].Name += "-2"
				claim.Spec.Devices.Requests[1].Selectors = []resource.DeviceSelector{
					{
						// Good selector.
						CEL: &resource.CELDeviceSelector{
							Expression: `device.driver == "dra.example.com"`,
						},
					},
					{
						// Bad selector.
						CEL: &resource.CELDeviceSelector{
							Expression: `device.attributes[true].someBoolean`,
						},
					},
				}
				return claim
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateResourceClaim(scenario.claim)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateClaimUpdate(t *testing.T) {
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
				spec.Devices.Requests[0].DeviceClassName += "2"
				return *spec
			}(), "field is immutable")},
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Spec.Devices.Requests[0].DeviceClassName += "2"
				return claim
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldClaim.ResourceVersion = "1"
			errs := ValidateResourceClaimUpdate(scenario.update(scenario.oldClaim.DeepCopy()), scenario.oldClaim)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateClaimStatusUpdate(t *testing.T) {
	validAllocatedClaim := validClaim.DeepCopy()
	validAllocatedClaim.Status = resource.ResourceClaimStatus{
		Allocation: &resource.AllocationResult{
			Devices: resource.DeviceAllocationResult{
				Results: []resource.DeviceRequestAllocationResult{{
					Request: goodName,
					Driver:  goodName,
					Pool:    goodName,
					Device:  goodName,
				}},
			},
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
		"valid-add-allocation-empty": {
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{}
				return claim
			},
		},
		"valid-add-allocation-non-empty": {
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request: goodName,
							Driver:  goodName,
							Pool:    goodName,
							Device:  goodName,
						}},
					},
				}
				return claim
			},
		},
		"invalid-add-allocation-bad-request": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("request"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("request"), badName, "must be the name of a request in the claim"),
			},
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request: badName,
							Driver:  goodName,
							Pool:    goodName,
							Device:  goodName,
						}},
					},
				}
				return claim
			},
		},
		"invalid-node-selector": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("status", "allocation", "nodeSelector", "nodeSelectorTerms"), "must have at least one node selector term")},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					NodeSelector: &core.NodeSelector{
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
		"invalid-reserved-for-no-allocation": {
			wantFailures: field.ErrorList{field.Forbidden(field.NewPath("status", "reservedFor"), "may not be specified when `allocated` is not set")},
			oldClaim:     validClaim,
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
				claim.Status.Allocation.Devices.Results[0].Driver += "-2"
				return claim.Status.Allocation
			}(), "field is immutable")},
			oldClaim: validAllocatedClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation.Devices.Results[0].Driver += "-2"
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
		"invalid-request-name": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "devices", "config").Index(0).Child("requests").Index(1), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Invalid(field.NewPath("status", "allocation", "devices", "config").Index(0).Child("requests").Index(1), badName, "must be the name of a request in the claim"),
			},
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim = claim.DeepCopy()
				claim.Status.Allocation = validAllocatedClaim.Status.Allocation.DeepCopy()
				claim.Status.Allocation.Devices.Config = []resource.DeviceAllocationConfiguration{{
					Source:   resource.AllocationConfigSourceClaim,
					Requests: []string{claim.Spec.Devices.Requests[0].Name, badName},
					DeviceConfiguration: resource.DeviceConfiguration{
						Opaque: &resource.OpaqueDeviceConfiguration{
							Driver: "dra.example.com",
							Parameters: runtime.RawExtension{
								Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
							},
						},
					},
				}}
				return claim
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldClaim.ResourceVersion = "1"
			errs := ValidateResourceClaimStatusUpdate(scenario.update(scenario.oldClaim.DeepCopy()), scenario.oldClaim)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}
