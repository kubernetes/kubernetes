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

	apiresource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
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
	goodName          = "foo"
	goodName2         = "bar"
	badName           = "!@#$%^"
	goodNS            = "ns"
	badSubrequestName = "&^%$"
)

var (
	badRequestFormat      = fmt.Sprintf("%s/%s/%s", goodName, goodName, goodName)
	badFullSubrequestName = fmt.Sprintf("%s/%s", badName, badSubrequestName)
	validClaimSpec        = resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{{
				Name: goodName,
				Exactly: &resource.ExactDeviceRequest{
					DeviceClassName: goodName,
					AllocationMode:  resource.DeviceAllocationModeExactCount,
					Count:           1,
				},
			}},
		},
	}
	validClaimSpecWithFirstAvailable = resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{{
				Name: goodName,
				FirstAvailable: []resource.DeviceSubRequest{
					{
						Name:            goodName,
						DeviceClassName: goodName,
						AllocationMode:  resource.DeviceAllocationModeExactCount,
						Count:           1,
					},
					{
						Name:            goodName2,
						DeviceClassName: goodName,
						AllocationMode:  resource.DeviceAllocationModeExactCount,
						Count:           1,
					},
				},
			}},
		},
	}
	validSelector = []resource.DeviceSelector{
		{
			CEL: &resource.CELDeviceSelector{
				Expression: `device.driver == "dra.example.com"`,
			},
		},
	}
	validClaim                    = testClaim(goodName, goodNS, validClaimSpec)
	conditionValidationErrMessage = "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')"
)

func TestValidateClaim(t *testing.T) {
	now := metav1.Now()
	badValue := "spaces not allowed"

	scenarios := map[string]struct {
		claim                         *resource.ResourceClaim
		wantFailures                  field.ErrorList
		consumableCapacityFeatureGate bool
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
				claim.DeletionGracePeriodSeconds = ptr.To[int64](10)
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
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "devices", "requests").Index(0).Child("exactly", "deviceClassName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Requests[0].Exactly.DeviceClassName = badName
				return claim
			}(),
		},
		"missing-classname": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("spec", "devices", "requests").Index(0).Child("exactly", "deviceClassName"), "")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Requests[0].Exactly.DeviceClassName = ""
				return claim
			}(),
		},
		"invalid-request": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices", "requests"), resource.DeviceRequestsMaxSize+1, resource.DeviceRequestsMaxSize),
				field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("requests").Index(1), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("requests").Index(1), badName, "must be the name of a request in the claim or the name of a request and a subrequest separated by '/'"),
				field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("matchAttribute"), "missing-domain", "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName', regex used for validation is '[A-Za-z_][A-Za-z0-9_]*')"),
				field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("matchAttribute"), resource.FullyQualifiedName("missing-domain"), "must include a domain"),
				field.Required(field.NewPath("spec", "devices", "constraints").Index(1).Child("matchAttribute"), "name required"),
				field.Required(field.NewPath("spec", "devices", "constraints").Index(2).Child("matchAttribute"), ""),
				field.TooMany(field.NewPath("spec", "devices", "constraints"), resource.DeviceConstraintsMaxSize+1, resource.DeviceConstraintsMaxSize),
				field.Invalid(field.NewPath("spec", "devices", "config").Index(0).Child("requests").Index(1), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Invalid(field.NewPath("spec", "devices", "config").Index(0).Child("requests").Index(1), badName, "must be the name of a request in the claim or the name of a request and a subrequest separated by '/'"),
				field.TooMany(field.NewPath("spec", "devices", "config"), resource.DeviceConfigMaxSize+1, resource.DeviceConfigMaxSize),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Constraints = []resource.DeviceConstraint{
					{
						Requests:       []string{claim.Spec.Devices.Requests[0].Name, badName},
						MatchAttribute: ptr.To(resource.FullyQualifiedName("missing-domain")),
					},
					{
						MatchAttribute: ptr.To(resource.FullyQualifiedName("")),
					},
					{
						MatchAttribute: nil,
					},
				}
				for i := len(claim.Spec.Devices.Constraints); i < resource.DeviceConstraintsMaxSize+1; i++ {
					claim.Spec.Devices.Constraints = append(claim.Spec.Devices.Constraints, resource.DeviceConstraint{MatchAttribute: ptr.To(resource.FullyQualifiedName("foo/bar"))})
				}
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
				for i := len(claim.Spec.Devices.Config); i < resource.DeviceConfigMaxSize+1; i++ {
					claim.Spec.Devices.Config = append(claim.Spec.Devices.Config, resource.DeviceClaimConfiguration{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
								},
							},
						},
					})
				}
				for i := len(claim.Spec.Devices.Requests); i < resource.DeviceRequestsMaxSize+1; i++ {
					req := claim.Spec.Devices.Requests[0].DeepCopy()
					req.Name += fmt.Sprintf("%d", i)
					claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, *req)
				}
				return claim
			}(),
		},
		"invalid-distinct-constraint": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("distinctAttribute"), "missing-domain", "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName', regex used for validation is '[A-Za-z_][A-Za-z0-9_]*')"),
				field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("distinctAttribute"), resource.FullyQualifiedName("missing-domain"), "must include a domain"),
				field.Required(field.NewPath("spec", "devices", "constraints").Index(1).Child("distinctAttribute"), "name required"),
				field.Required(field.NewPath("spec", "devices", "constraints").Index(2), `exactly one of "matchAttribute" or "distinctAttribute" is required, but multiple fields are set`)},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Constraints = []resource.DeviceConstraint{
					{
						Requests:          []string{claim.Spec.Devices.Requests[0].Name},
						DistinctAttribute: ptr.To(resource.FullyQualifiedName("missing-domain")),
					},
					{
						DistinctAttribute: ptr.To(resource.FullyQualifiedName("")),
					},
					{
						MatchAttribute:    nil,
						DistinctAttribute: nil,
					},
				}
				return claim
			}(),
			consumableCapacityFeatureGate: true,
		},
		"valid-request": {
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				for i := len(claim.Spec.Devices.Constraints); i < resource.DeviceConstraintsMaxSize; i++ {
					claim.Spec.Devices.Constraints = append(claim.Spec.Devices.Constraints, resource.DeviceConstraint{MatchAttribute: ptr.To(resource.FullyQualifiedName("foo/bar"))})
				}
				for i := len(claim.Spec.Devices.Config); i < resource.DeviceConfigMaxSize; i++ {
					claim.Spec.Devices.Config = append(claim.Spec.Devices.Config, resource.DeviceClaimConfiguration{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
								},
							},
						},
					})
				}
				for i := len(claim.Spec.Devices.Requests); i < resource.DeviceRequestsMaxSize; i++ {
					req := claim.Spec.Devices.Requests[0].DeepCopy()
					req.Name += fmt.Sprintf("%d", i)
					claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, *req)
				}
				return claim
			}(),
		},
		"invalid-spec": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("requests").Index(1), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("requests").Index(1), badName, "must be the name of a request in the claim or the name of a request and a subrequest separated by '/'"),
				field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("matchAttribute"), "missing-domain", "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName', regex used for validation is '[A-Za-z_][A-Za-z0-9_]*')"),
				field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("matchAttribute"), resource.FullyQualifiedName("missing-domain"), "must include a domain"),
				field.Required(field.NewPath("spec", "devices", "constraints").Index(1).Child("matchAttribute"), "name required"),
				field.Required(field.NewPath("spec", "devices", "constraints").Index(2).Child("matchAttribute"), ""),
				field.Invalid(field.NewPath("spec", "devices", "config").Index(0).Child("requests").Index(1), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Invalid(field.NewPath("spec", "devices", "config").Index(0).Child("requests").Index(1), badName, "must be the name of a request in the claim or the name of a request and a subrequest separated by '/'"),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Constraints = []resource.DeviceConstraint{
					{
						Requests:       []string{claim.Spec.Devices.Requests[0].Name, badName},
						MatchAttribute: ptr.To(resource.FullyQualifiedName("missing-domain")),
					},
					{
						MatchAttribute: ptr.To(resource.FullyQualifiedName("")),
					},
					{
						MatchAttribute: nil,
					},
				}
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
		"allocation-mode": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "requests").Index(2).Child("exactly", "count"), int64(-1), "must be greater than zero"),
				field.NotSupported(field.NewPath("spec", "devices", "requests").Index(3).Child("exactly", "allocationMode"), resource.DeviceAllocationMode("other"), []resource.DeviceAllocationMode{resource.DeviceAllocationModeAll, resource.DeviceAllocationModeExactCount}),
				field.Invalid(field.NewPath("spec", "devices", "requests").Index(4).Child("exactly", "count"), int64(2), "must not be specified when allocationMode is 'All'"),
				field.Duplicate(field.NewPath("spec", "devices", "requests").Index(5).Child("name"), "foo"),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)

				goodReq := &claim.Spec.Devices.Requests[0]
				goodReq.Name = "foo"
				goodReq.Exactly.AllocationMode = resource.DeviceAllocationModeExactCount
				goodReq.Exactly.Count = 1

				req := goodReq.DeepCopy()
				req.Name += "2"
				req.Exactly.AllocationMode = resource.DeviceAllocationModeAll
				req.Exactly.Count = 0
				claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, *req)

				req = goodReq.DeepCopy()
				req.Name += "3"
				req.Exactly.AllocationMode = resource.DeviceAllocationModeExactCount
				req.Exactly.Count = -1
				claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, *req)

				req = goodReq.DeepCopy()
				req.Name += "4"
				req.Exactly.AllocationMode = resource.DeviceAllocationMode("other")
				claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, *req)

				req = goodReq.DeepCopy()
				req.Name += "5"
				req.Exactly.AllocationMode = resource.DeviceAllocationModeAll
				req.Exactly.Count = 2
				claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, *req)

				req = goodReq.DeepCopy()
				// Same name -> duplicate.
				goodReq.Name = "foo"
				claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, *req)

				return claim
			}(),
		},
		"configuration": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("spec", "devices", "config").Index(0).Child("opaque", "parameters"), ""),
				field.Invalid(field.NewPath("spec", "devices", "config").Index(1).Child("opaque", "parameters"), "<value omitted>", "error parsing data as JSON: unexpected end of JSON input"),
				field.Invalid(field.NewPath("spec", "devices", "config").Index(2).Child("opaque", "parameters"), "<value omitted>", "must be a valid JSON object"),
				field.Required(field.NewPath("spec", "devices", "config").Index(3).Child("opaque", "parameters"), ""),
				field.TooLong(field.NewPath("spec", "devices", "config").Index(5).Child("opaque", "parameters"), "" /* unused */, resource.OpaqueParametersMaxLength),
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
					{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`{"str": "` + strings.Repeat("x", resource.OpaqueParametersMaxLength-9-2) + `"}`),
								},
							},
						},
					},
					{
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`{"str": "` + strings.Repeat("x", resource.OpaqueParametersMaxLength-9-2+1 /* too large by one */) + `"}`),
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
				field.Invalid(field.NewPath("spec", "devices", "requests").Index(1).Child("exactly", "selectors").Index(1).Child("cel", "expression"), `device.attributes[true].someBoolean`, "compilation failed: ERROR: <input>:1:18: found no matching overload for '_[_]' applied to '(map(string, map(string, any)), bool)'\n | device.attributes[true].someBoolean\n | .................^"),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, *claim.Spec.Devices.Requests[0].DeepCopy())
				claim.Spec.Devices.Requests[1].Name += "-2"
				claim.Spec.Devices.Requests[1].Exactly.Selectors = []resource.DeviceSelector{
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
		"CEL-length": {
			wantFailures: field.ErrorList{
				field.TooLong(field.NewPath("spec", "devices", "requests").Index(1).Child("exactly", "selectors").Index(1).Child("cel", "expression"), "" /*unused*/, resource.CELSelectorExpressionMaxLength),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, *claim.Spec.Devices.Requests[0].DeepCopy())
				claim.Spec.Devices.Requests[1].Name += "-2"
				expression := `device.driver == ""`
				claim.Spec.Devices.Requests[1].Exactly.Selectors = []resource.DeviceSelector{
					{
						// Good selector.
						CEL: &resource.CELDeviceSelector{
							Expression: strings.ReplaceAll(expression, `""`, `"`+strings.Repeat("x", resource.CELSelectorExpressionMaxLength-len(expression))+`"`),
						},
					},
					{
						// Too long by one selector.
						CEL: &resource.CELDeviceSelector{
							Expression: strings.ReplaceAll(expression, `""`, `"`+strings.Repeat("x", resource.CELSelectorExpressionMaxLength-len(expression)+1)+`"`),
						},
					},
				}
				return claim
			}(),
		},
		"CEL-cost": {
			wantFailures: field.ErrorList{
				field.Forbidden(field.NewPath("spec", "devices", "requests").Index(0).Child("exactly", "selectors").Index(0).Child("cel", "expression"), "too complex, exceeds cost limit"),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Requests[0].Exactly.Selectors = []resource.DeviceSelector{
					{
						CEL: &resource.CELDeviceSelector{
							// From https://github.com/kubernetes/kubernetes/blob/50fc400f178d2078d0ca46aee955ee26375fc437/test/integration/apiserver/cel/validatingadmissionpolicy_test.go#L2150.
							Expression: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(x, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(y, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z4, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].all(z5, int('1'.find('[0-9]*')) < 100)))))))`,
						},
					},
				}
				return claim
			}(),
		},
		"prioritized-list-valid": {
			wantFailures: nil,
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable)
				return claim
			}(),
		},
		"prioritized-list-both-first-available-and-exactly-set": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "requests").Index(0), nil, "exactly one of `exactly` or `firstAvailable` is required, but multiple fields are set"),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable)

				claim.Spec.Devices.Requests[0].Exactly = &resource.ExactDeviceRequest{
					DeviceClassName: goodName,
					Selectors:       validSelector,
					AllocationMode:  resource.DeviceAllocationModeExactCount,
					Count:           2,
					AdminAccess:     ptr.To(true),
				}
				return claim
			}(),
		},
		"prioritized-list-invalid-nested-request": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("name"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Required(field.NewPath("spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("deviceClassName"), ""),
				field.NotSupported(field.NewPath("spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("allocationMode"), resource.DeviceAllocationMode(""), []resource.DeviceAllocationMode{resource.DeviceAllocationModeAll, resource.DeviceAllocationModeExactCount}),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable)
				claim.Spec.Devices.Requests[0].FirstAvailable[0] = resource.DeviceSubRequest{
					Name: badName,
				}
				return claim
			}(),
		},
		"prioritized-list-nested-requests-same-name": {
			wantFailures: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "devices", "requests").Index(0).Child("firstAvailable").Index(1).Child("name"), "foo"),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable)
				claim.Spec.Devices.Requests[0].FirstAvailable[1].Name = goodName
				return claim
			}(),
		},
		"prioritized-list-too-many-subrequests": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("spec", "devices", "requests").Index(0).Child("firstAvailable"), 9, 8),
			},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpec)
				claim.Spec.Devices.Requests[0].Exactly = nil
				var subRequests []resource.DeviceSubRequest
				for i := 0; i <= 8; i++ {
					subRequests = append(subRequests, resource.DeviceSubRequest{
						Name:            fmt.Sprintf("subreq-%d", i),
						DeviceClassName: goodName,
						AllocationMode:  resource.DeviceAllocationModeExactCount,
						Count:           1,
					})
				}
				claim.Spec.Devices.Requests[0].FirstAvailable = subRequests
				return claim
			}(),
		},
		"prioritized-list-config-requests-with-subrequest-reference": {
			wantFailures: nil,
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable)
				claim.Spec.Devices.Config = []resource.DeviceClaimConfiguration{
					{
						Requests: []string{"foo/bar"},
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
								},
							},
						},
					},
				}
				return claim
			}(),
		},
		"prioritized-list-config-requests-with-parent-request-reference": {
			wantFailures: nil,
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable)
				claim.Spec.Devices.Config = []resource.DeviceClaimConfiguration{
					{
						Requests: []string{"foo"},
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
								},
							},
						},
					},
				}
				return claim
			}(),
		},
		"prioritized-list-config-requests-with-invalid-subrequest-reference": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "devices", "config").Index(0).Child("requests").Index(0), "foo/baz", "must be the name of a request in the claim or the name of a request and a subrequest separated by '/'")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable)
				claim.Spec.Devices.Config = []resource.DeviceClaimConfiguration{
					{
						Requests: []string{"foo/baz"},
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
								},
							},
						},
					},
				}
				return claim
			}(),
		},
		"prioritized-list-constraints-requests-with-subrequest-reference": {
			wantFailures: nil,
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable)
				claim.Spec.Devices.Constraints = []resource.DeviceConstraint{
					{
						Requests:       []string{"foo/bar"},
						MatchAttribute: ptr.To(resource.FullyQualifiedName("dra.example.com/driverVersion")),
					},
				}
				return claim
			}(),
		},
		"prioritized-list-constraints-requests-with-parent-request-reference": {
			wantFailures: nil,
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable)
				claim.Spec.Devices.Constraints = []resource.DeviceConstraint{
					{
						Requests:       []string{"foo"},
						MatchAttribute: ptr.To(resource.FullyQualifiedName("dra.example.com/driverVersion")),
					},
				}
				return claim
			}(),
		},
		"prioritized-list-constraints-requests-with-invalid-subrequest-reference": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec", "devices", "constraints").Index(0).Child("requests").Index(0), "foo/baz", "must be the name of a request in the claim or the name of a request and a subrequest separated by '/'")},
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable)
				claim.Spec.Devices.Constraints = []resource.DeviceConstraint{
					{
						Requests:       []string{"foo/baz"},
						MatchAttribute: ptr.To(resource.FullyQualifiedName("dra.example.com/driverVersion")),
					},
				}
				return claim
			}(),
		},
		"tolerations": {
			wantFailures: func() field.ErrorList {
				var allErrs field.ErrorList
				fldPath := field.NewPath("spec", "devices", "requests").Index(0).Child("firstAvailable").Index(0).Child("tolerations")
				allErrs = append(allErrs,
					field.Required(fldPath.Index(0).Child("operator"), ""),
				)
				fldPath = field.NewPath("spec", "devices", "requests").Index(1).Child("exactly", "tolerations")
				allErrs = append(allErrs,
					field.Required(fldPath.Index(3).Child("operator"), ""),

					field.NotSupported(fldPath.Index(4).Child("operator"), resource.DeviceTolerationOperator("some-other-op"), []resource.DeviceTolerationOperator{resource.DeviceTolerationOpEqual, resource.DeviceTolerationOpExists}),

					field.Invalid(fldPath.Index(5).Child("key"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')"),
					field.Invalid(fldPath.Index(5).Child("value"), badName, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')"),
					field.NotSupported(fldPath.Index(5).Child("effect"), resource.DeviceTaintEffect("some-other-effect"), []resource.DeviceTaintEffect{resource.DeviceTaintEffectNoExecute, resource.DeviceTaintEffectNoSchedule}),
				)
				return allErrs
			}(),
			claim: func() *resource.ResourceClaim {
				claim := testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable)
				claim.Spec.Devices.Requests[0].FirstAvailable[0].Tolerations = []resource.DeviceToleration{
					{
						// One invalid case to verify the field path.
						// Full test for validateDeviceToleration is below.
						Operator: "",
					},
				}
				claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, *validClaimSpec.Devices.Requests[0].DeepCopy())
				claim.Spec.Devices.Requests[1].Name += "-other"
				claim.Spec.Devices.Requests[1].Exactly.Tolerations = []resource.DeviceToleration{
					{
						// Minimal valid toleration: match all taints.
						Operator: resource.DeviceTolerationOpExists,
					},
					{
						Key:      "example.com/taint",
						Operator: resource.DeviceTolerationOpExists,
						Effect:   resource.DeviceTaintEffectNoExecute,
					},
					{
						Key:      "example.com/taint",
						Operator: resource.DeviceTolerationOpEqual,
						Value:    "tainted",
						Effect:   resource.DeviceTaintEffectNoSchedule,
					},
					{
						// Invalid, operator is required.
						Operator: "",
					},
					{
						Key:      goodName,
						Operator: "some-other-op",
					},
					{
						Key:      badName,
						Operator: resource.DeviceTolerationOpEqual,
						Value:    badName,
						Effect:   "some-other-effect",
					},
				}
				return claim
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAConsumableCapacity, scenario.consumableCapacityFeatureGate)
			errs := ValidateResourceClaim(scenario.claim)
			assertFailures(t, scenario.wantFailures, errs)
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
		"invalid-update": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("spec"), func() resource.ResourceClaimSpec {
				spec := validClaim.Spec.DeepCopy()
				spec.Devices.Requests[0].Exactly.DeviceClassName += "2"
				return *spec
			}(), "field is immutable")},
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Spec.Devices.Requests[0].Exactly.DeviceClassName += "2"
				return claim
			},
		},
		"too-large-config-valid-if-stored": {
			oldClaim: func() *resource.ResourceClaim {
				claim := validClaim.DeepCopy()
				claim.Spec.Devices.Config = []resource.DeviceClaimConfiguration{{
					DeviceConfiguration: resource.DeviceConfiguration{
						Opaque: &resource.OpaqueDeviceConfiguration{
							Driver:     goodName,
							Parameters: runtime.RawExtension{Raw: []byte(`{"str": "` + strings.Repeat("x", resource.OpaqueParametersMaxLength-9-2+1 /* too large by one */) + `"}`)},
						},
					},
				}}
				return claim
			}(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				// No changes -> remains valid.
				return claim
			},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldClaim.ResourceVersion = "1"
			errs := ValidateResourceClaimUpdate(scenario.update(scenario.oldClaim.DeepCopy()), scenario.oldClaim)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateClaimStatusUpdate(t *testing.T) {
	goodShareID := types.UID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
	goodShareID2 := types.UID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
	badLengthShareID := types.UID("a")
	badFormatShareID := types.UID("Aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
	goodShareIDStr := string(goodShareID)
	goodShareID2Str := string(goodShareID2)
	badLengthShareIDStr := string(badLengthShareID)
	badFormatShareIDStr := string(badFormatShareID)
	validAllocatedClaim := validClaim.DeepCopy()
	validAllocatedClaim.Status = resource.ResourceClaimStatus{
		Allocation: &resource.AllocationResult{
			Devices: resource.DeviceAllocationResult{
				Results: []resource.DeviceRequestAllocationResult{{
					Request:     goodName,
					Driver:      goodName,
					Pool:        goodName,
					Device:      goodName,
					AdminAccess: ptr.To(false), // Required for new allocations.
				}},
			},
		},
	}
	validAllocatedClaimOld := validAllocatedClaim.DeepCopy()
	validAllocatedClaimOld.Status.Allocation.Devices.Results[0].AdminAccess = nil // Not required in 1.31.

	scenarios := map[string]struct {
		adminAccess                   bool
		deviceStatusFeatureGate       bool
		prioritizedListFeatureGate    bool
		consumableCapacityFeatureGate bool
		oldClaim                      *resource.ResourceClaim
		update                        func(claim *resource.ResourceClaim) *resource.ResourceClaim
		wantFailures                  field.ErrorList
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
							Request:     goodName,
							Driver:      goodName,
							Pool:        goodName,
							Device:      goodName,
							AdminAccess: ptr.To(false),
						}},
					},
				}
				return claim
			},
		},
		"invalid-add-allocation-bad-request": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("request"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("request"), badName, "must be the name of a request in the claim or the name of a request and a subrequest separated by '/'"),
			},
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:     badName,
							Driver:      goodName,
							Pool:        goodName,
							Device:      goodName,
							AdminAccess: ptr.To(false),
						}},
					},
				}
				return claim
			},
		},
		"okay-add-allocation-missing-admin-access": {
			adminAccess: false,
			oldClaim:    validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:     goodName,
							Driver:      goodName,
							Pool:        goodName,
							Device:      goodName,
							AdminAccess: nil, // Intentionally not set.
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
		"add-reservation-old-claim": {
			oldClaim: validAllocatedClaimOld,
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
			wantFailures: field.ErrorList{field.TooMany(field.NewPath("status", "reservedFor"), resource.ResourceClaimReservedForMaxSize+1, resource.ResourceClaimReservedForMaxSize)},
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
		"invalid-request-name": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "devices", "config").Index(0).Child("requests").Index(1), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Invalid(field.NewPath("status", "allocation", "devices", "config").Index(0).Child("requests").Index(1), badName, "must be the name of a request in the claim or the name of a request and a subrequest separated by '/'"),
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
		"configuration": {
			wantFailures: field.ErrorList{
				field.Required(field.NewPath("status", "allocation", "devices", "config").Index(1).Child("source"), ""),
				field.NotSupported(field.NewPath("status", "allocation", "devices", "config").Index(2).Child("source"), resource.AllocationConfigSource("no-such-source"), []resource.AllocationConfigSource{resource.AllocationConfigSourceClaim, resource.AllocationConfigSourceClass}),
				field.Required(field.NewPath("status", "allocation", "devices", "config").Index(3).Child("opaque"), ""),
				field.Required(field.NewPath("status", "allocation", "devices", "config").Index(4).Child("opaque", "driver"), ""),
				field.Invalid(field.NewPath("status", "allocation", "devices", "config").Index(4).Child("opaque", "driver"), "", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
				field.Required(field.NewPath("status", "allocation", "devices", "config").Index(4).Child("opaque", "parameters"), ""),
				field.TooLong(field.NewPath("status", "allocation", "devices", "config").Index(6).Child("opaque", "parameters"), "" /* unused */, resource.OpaqueParametersMaxLength),
			},
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim = claim.DeepCopy()
				claim.Status.Allocation = validAllocatedClaim.Status.Allocation.DeepCopy()
				claim.Status.Allocation.Devices.Config = []resource.DeviceAllocationConfiguration{
					{
						Source: resource.AllocationConfigSourceClaim,
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
								},
							},
						},
					},
					{
						Source: "", /* Empty! */
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
								},
							},
						},
					},
					{
						Source: resource.AllocationConfigSource("no-such-source"),
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver: "dra.example.com",
								Parameters: runtime.RawExtension{
									Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
								},
							},
						},
					},
					{
						Source:              resource.AllocationConfigSourceClaim,
						DeviceConfiguration: resource.DeviceConfiguration{ /* Empty! */ },
					},
					{
						Source: resource.AllocationConfigSourceClaim,
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{ /* Empty! */ },
						},
					},
					{
						Source: resource.AllocationConfigSourceClaim,
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver:     goodName,
								Parameters: runtime.RawExtension{Raw: []byte(`{"str": "` + strings.Repeat("x", resource.OpaqueParametersMaxLength-9-2) + `"}`)},
							},
						},
					},
					{
						Source: resource.AllocationConfigSourceClaim,
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver:     goodName,
								Parameters: runtime.RawExtension{Raw: []byte(`{"str": "` + strings.Repeat("x", resource.OpaqueParametersMaxLength-9-2+1 /* too large by one */) + `"}`)},
							},
						},
					},
					// Other invalid resource.DeviceConfiguration are covered elsewhere. */
				}
				return claim
			},
		},
		"valid-configuration-update": {
			oldClaim: func() *resource.ResourceClaim {
				claim := validClaim.DeepCopy()
				claim.Status.Allocation = validAllocatedClaim.Status.Allocation.DeepCopy()
				claim.Status.Allocation.Devices.Config = []resource.DeviceAllocationConfiguration{
					{
						Source: resource.AllocationConfigSourceClaim,
						DeviceConfiguration: resource.DeviceConfiguration{
							Opaque: &resource.OpaqueDeviceConfiguration{
								Driver:     goodName,
								Parameters: runtime.RawExtension{Raw: []byte(`{"str": "` + strings.Repeat("x", resource.OpaqueParametersMaxLength-9-2+1 /* too large by one */) + `"}`)},
							},
						},
					},
				}
				return claim
			}(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				// No change -> remains valid.
				return claim
			},
		},
		"valid-network-device-status": {
			oldClaim: func() *resource.ResourceClaim { return validAllocatedClaim }(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver: goodName,
						Pool:   goodName,
						Device: goodName,
						Conditions: []metav1.Condition{
							{Type: "test-0", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-1", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-2", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-3", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-4", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-5", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-6", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-7", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
						},
						Data: &runtime.RawExtension{
							Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
						},
						NetworkData: &resource.NetworkDeviceData{
							InterfaceName:   strings.Repeat("x", 256),
							HardwareAddress: strings.Repeat("x", 128),
							IPs: []string{
								"10.9.8.0/24",
								"2001:db8::/64",
								"10.9.8.1/24",
								"2001:db8::1/64",
								"10.9.8.2/24", "10.9.8.3/24", "10.9.8.4/24", "10.9.8.5/24", "10.9.8.6/24", "10.9.8.7/24",
								"10.9.8.8/24", "10.9.8.9/24", "10.9.8.10/24", "10.9.8.11/24", "10.9.8.12/24", "10.9.8.13/24",
							},
						},
					},
				}
				return claim
			},
			deviceStatusFeatureGate: true,
		},
		"invalid-device-status-duplicate": {
			wantFailures: field.ErrorList{
				field.Duplicate(field.NewPath("status", "devices").Index(0).Child("networkData", "ips").Index(1), "2001:db8::1/64"),
				field.Duplicate(field.NewPath("status", "devices").Index(1).Child("deviceID"), structured.MakeSharedDeviceID(structured.MakeDeviceID(goodName, goodName, goodName), nil)),
			},
			oldClaim: func() *resource.ResourceClaim { return validAllocatedClaim }(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver: goodName,
						Pool:   goodName,
						Device: goodName,
						NetworkData: &resource.NetworkDeviceData{
							IPs: []string{
								"2001:db8::1/64",
								"2001:db8::1/64",
							},
						},
					},
					{
						Driver: goodName,
						Pool:   goodName,
						Device: goodName,
					},
				}
				return claim
			},
			deviceStatusFeatureGate: true,
		},
		"invalid-device-status-duplicate-with-share-id": {
			wantFailures: field.ErrorList{
				field.Duplicate(field.NewPath("status", "devices").Index(1).Child("deviceID"),
					structured.MakeSharedDeviceID(structured.MakeDeviceID(goodName, goodName, goodName), ptr.To(goodShareID))),
			},
			oldClaim: func() *resource.ResourceClaim {
				claim := validAllocatedClaim.DeepCopy()
				claim.Status.Allocation.Devices.Results[0].ShareID = ptr.To(goodShareID)
				return claim
			}(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver:  goodName,
						Pool:    goodName,
						Device:  goodName,
						ShareID: ptr.To(goodShareIDStr),
					},
					{
						Driver:  goodName,
						Pool:    goodName,
						Device:  goodName,
						ShareID: ptr.To(goodShareIDStr),
					},
				}
				return claim
			},
			deviceStatusFeatureGate:       true,
			consumableCapacityFeatureGate: true,
		},
		"invalid-network-device-status": {
			wantFailures: field.ErrorList{
				field.TooLong(field.NewPath("status", "devices").Index(0).Child("networkData", "interfaceName"), "", resource.NetworkDeviceDataInterfaceNameMaxLength),
				field.TooLong(field.NewPath("status", "devices").Index(0).Child("networkData", "hardwareAddress"), "", resource.NetworkDeviceDataHardwareAddressMaxLength),
				field.Invalid(field.NewPath("status", "devices").Index(0).Child("networkData", "ips").Index(0), "300.9.8.0/24", "must be a valid address in CIDR form, (e.g. 10.9.8.7/24 or 2001:db8::1/64)"),
				field.Invalid(field.NewPath("status", "devices").Index(0).Child("networkData", "ips").Index(1), "010.009.008.000/24", "must be in canonical form (\"10.9.8.0/24\")"),
				field.Invalid(field.NewPath("status", "devices").Index(0).Child("networkData", "ips").Index(2), "2001:0db8::1/64", "must be in canonical form (\"2001:db8::1/64\")"),
			},
			oldClaim: func() *resource.ResourceClaim { return validAllocatedClaim }(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver: goodName,
						Pool:   goodName,
						Device: goodName,
						NetworkData: &resource.NetworkDeviceData{
							InterfaceName:   strings.Repeat("x", resource.NetworkDeviceDataInterfaceNameMaxLength+1),
							HardwareAddress: strings.Repeat("x", resource.NetworkDeviceDataHardwareAddressMaxLength+1),
							IPs: []string{
								"300.9.8.0/24",
								"010.009.008.000/24",
								"2001:0db8::1/64",
							},
						},
					},
				}
				return claim
			},
			deviceStatusFeatureGate: true,
		},
		"invalid-data-device-status": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "devices").Index(0).Child("data"), "<value omitted>", "error parsing data as JSON: invalid character 'o' in literal false (expecting 'a')"),
			},
			oldClaim: func() *resource.ResourceClaim { return validAllocatedClaim }(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver: goodName,
						Pool:   goodName,
						Device: goodName,
						Data: &runtime.RawExtension{
							Raw: []byte(`foo`),
						},
					},
				}
				return claim
			},
			deviceStatusFeatureGate: true,
		},
		"invalid-data-device-status-limits": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("status", "devices").Index(0).Child("conditions"), resource.AllocatedDeviceStatusMaxConditions+1, resource.AllocatedDeviceStatusMaxConditions),
				field.TooLong(field.NewPath("status", "devices").Index(0).Child("data"), "" /* unused */, resource.AllocatedDeviceStatusDataMaxLength),
				field.TooMany(field.NewPath("status", "devices").Index(0).Child("networkData", "ips"), resource.NetworkDeviceDataMaxIPs+1, resource.NetworkDeviceDataMaxIPs),
			},
			oldClaim: func() *resource.ResourceClaim { return validAllocatedClaim }(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver: goodName,
						Pool:   goodName,
						Device: goodName,
						Data:   &runtime.RawExtension{Raw: []byte(`{"str": "` + strings.Repeat("x", resource.AllocatedDeviceStatusDataMaxLength-9-2+1 /* too large by one */) + `"}`)},
						Conditions: []metav1.Condition{
							{Type: "test-0", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-1", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-2", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-3", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-4", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-5", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-6", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-7", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-8", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
						},
						NetworkData: &resource.NetworkDeviceData{
							IPs: []string{
								"10.9.8.0/24", "10.9.8.1/24", "10.9.8.2/24", "10.9.8.3/24", "10.9.8.4/24", "10.9.8.5/24", "10.9.8.6/24", "10.9.8.7/24", "10.9.8.8/24",
								"10.9.8.9/24", "10.9.8.10/24", "10.9.8.11/24", "10.9.8.12/24", "10.9.8.13/24", "10.9.8.14/24", "10.9.8.15/24", "10.9.8.16/24",
							},
						},
					},
				}
				return claim
			},
			deviceStatusFeatureGate: true,
		},
		"invalid-device-status-no-device": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "devices").Index(0), structured.MakeSharedDeviceID(structured.MakeDeviceID("b", "a", "r"), nil), "must be an allocated device in the claim"),
			},
			oldClaim: func() *resource.ResourceClaim { return validAllocatedClaim }(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver: "b",
						Pool:   "a",
						Device: "r",
					},
				}
				return claim
			},
			deviceStatusFeatureGate: true,
		},
		"invalid-device-status-duplicate-disabled-feature-gate": {
			wantFailures: field.ErrorList{
				field.Duplicate(field.NewPath("status", "devices").Index(0).Child("networkData", "ips").Index(1), "2001:db8::1/64"),
				field.Duplicate(field.NewPath("status", "devices").Index(1).Child("deviceID"), structured.MakeSharedDeviceID(structured.MakeDeviceID(goodName, goodName, goodName), nil)),
			},
			oldClaim: func() *resource.ResourceClaim { return validAllocatedClaim }(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver: goodName,
						Pool:   goodName,
						Device: goodName,
						NetworkData: &resource.NetworkDeviceData{
							IPs: []string{
								"2001:db8::1/64",
								"2001:db8::1/64",
							},
						},
					},
					{
						Driver: goodName,
						Pool:   goodName,
						Device: goodName,
					},
				}
				return claim
			},
			deviceStatusFeatureGate: false,
		},
		"invalid-device-status-duplicate-with-share-id-disabled-feature-gate": {
			wantFailures: field.ErrorList{
				field.Duplicate(field.NewPath("status", "devices").Index(1).Child("deviceID"),
					structured.MakeSharedDeviceID(structured.MakeDeviceID(goodName, goodName, goodName), ptr.To(goodShareID))),
			},
			oldClaim: func() *resource.ResourceClaim {
				claim := validAllocatedClaim.DeepCopy()
				claim.Status.Allocation.Devices.Results[0].ShareID = ptr.To(goodShareID)
				return claim
			}(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver:  goodName,
						Pool:    goodName,
						Device:  goodName,
						ShareID: ptr.To(goodShareIDStr),
					},
					{
						Driver:  goodName,
						Pool:    goodName,
						Device:  goodName,
						ShareID: ptr.To(goodShareIDStr),
					},
				}
				return claim
			},
			deviceStatusFeatureGate:       false,
			consumableCapacityFeatureGate: false,
		},
		"invalid-network-device-status-disabled-feature-gate": {
			wantFailures: field.ErrorList{
				field.TooLong(field.NewPath("status", "devices").Index(0).Child("networkData", "interfaceName"), "", resource.NetworkDeviceDataInterfaceNameMaxLength),
				field.TooLong(field.NewPath("status", "devices").Index(0).Child("networkData", "hardwareAddress"), "", resource.NetworkDeviceDataHardwareAddressMaxLength),
				field.Invalid(field.NewPath("status", "devices").Index(0).Child("networkData", "ips").Index(0), "300.9.8.0/24", "must be a valid address in CIDR form, (e.g. 10.9.8.7/24 or 2001:db8::1/64)"),
			},
			oldClaim: func() *resource.ResourceClaim { return validAllocatedClaim }(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver: goodName,
						Pool:   goodName,
						Device: goodName,
						NetworkData: &resource.NetworkDeviceData{
							InterfaceName:   strings.Repeat("x", resource.NetworkDeviceDataInterfaceNameMaxLength+1),
							HardwareAddress: strings.Repeat("x", resource.NetworkDeviceDataHardwareAddressMaxLength+1),
							IPs: []string{
								"300.9.8.0/24",
							},
						},
					},
				}
				return claim
			},
			deviceStatusFeatureGate: false,
		},
		"invalid-data-device-status-disabled-feature-gate": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "devices").Index(0).Child("data"), "<value omitted>", "error parsing data as JSON: invalid character 'o' in literal false (expecting 'a')"),
			},
			oldClaim: func() *resource.ResourceClaim { return validAllocatedClaim }(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver: goodName,
						Pool:   goodName,
						Device: goodName,
						Data: &runtime.RawExtension{
							Raw: []byte(`foo`),
						},
					},
				}
				return claim
			},
			deviceStatusFeatureGate: false,
		},
		"invalid-data-device-status-limits-feature-gate": {
			wantFailures: field.ErrorList{
				field.TooMany(field.NewPath("status", "devices").Index(0).Child("conditions"), resource.AllocatedDeviceStatusMaxConditions+1, resource.AllocatedDeviceStatusMaxConditions),
				field.TooLong(field.NewPath("status", "devices").Index(0).Child("data"), "" /* unused */, resource.AllocatedDeviceStatusDataMaxLength),
				field.TooMany(field.NewPath("status", "devices").Index(0).Child("networkData", "ips"), resource.NetworkDeviceDataMaxIPs+1, resource.NetworkDeviceDataMaxIPs),
			},
			oldClaim: func() *resource.ResourceClaim { return validAllocatedClaim }(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver: goodName,
						Pool:   goodName,
						Device: goodName,
						Data:   &runtime.RawExtension{Raw: []byte(`{"str": "` + strings.Repeat("x", resource.AllocatedDeviceStatusDataMaxLength-9-2+1 /* too large by one */) + `"}`)},
						Conditions: []metav1.Condition{
							{Type: "test-0", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-1", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-2", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-3", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-4", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-5", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-6", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-7", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
							{Type: "test-8", Status: metav1.ConditionTrue, Reason: "test_reason", LastTransitionTime: metav1.Now(), ObservedGeneration: 0},
						},
						NetworkData: &resource.NetworkDeviceData{
							IPs: []string{
								"10.9.8.0/24", "10.9.8.1/24", "10.9.8.2/24", "10.9.8.3/24", "10.9.8.4/24", "10.9.8.5/24", "10.9.8.6/24", "10.9.8.7/24", "10.9.8.8/24",
								"10.9.8.9/24", "10.9.8.10/24", "10.9.8.11/24", "10.9.8.12/24", "10.9.8.13/24", "10.9.8.14/24", "10.9.8.15/24", "10.9.8.16/24",
							},
						},
					},
				}
				return claim
			},
			deviceStatusFeatureGate: false,
		},
		"invalid-device-status-no-device-disabled-feature-gate": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "devices").Index(0), structured.MakeSharedDeviceID(structured.MakeDeviceID("b", "a", "r"), nil), "must be an allocated device in the claim"),
			},
			oldClaim: func() *resource.ResourceClaim { return validAllocatedClaim }(),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver: "b",
						Pool:   "a",
						Device: "r",
					},
				}
				return claim
			},
			deviceStatusFeatureGate: false,
		},
		"invalid-update-invalid-label-value": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "nodeSelector", "nodeSelectorTerms").Index(0).Child("matchExpressions").Index(0).Child("values").Index(0), "-1", "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')"),
			},
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim = claim.DeepCopy()
				claim.Status.Allocation = validAllocatedClaim.Status.Allocation.DeepCopy()
				claim.Status.Allocation.NodeSelector = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{
						MatchExpressions: []core.NodeSelectorRequirement{{
							Key:      "foo",
							Operator: core.NodeSelectorOpIn,
							Values:   []string{"-1"},
						}},
					}},
				}
				return claim
			},
		},
		"valid-update-with-invalid-label-value": {
			oldClaim: func() *resource.ResourceClaim {
				claim := validAllocatedClaim.DeepCopy()
				claim.Status.Allocation = validAllocatedClaim.Status.Allocation.DeepCopy()
				claim.Status.Allocation.NodeSelector = &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{{
						MatchExpressions: []core.NodeSelectorRequirement{{
							Key:      "foo",
							Operator: core.NodeSelectorOpIn,
							Values:   []string{"-1"},
						}},
					}},
				}
				return claim
			}(),
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
		"valid-add-allocation-with-sub-requests": {
			oldClaim: testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:     fmt.Sprintf("%s/%s", goodName, goodName),
							Driver:      goodName,
							Pool:        goodName,
							Device:      goodName,
							AdminAccess: ptr.To(false),
						}},
					},
				}
				return claim
			},
			prioritizedListFeatureGate: true,
		},
		"invalid-add-allocation-with-sub-requests-invalid-format": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("request"), badRequestFormat, "must be the name of a request in the claim or the name of a request and a subrequest separated by '/'"),
			},
			oldClaim: testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:     badRequestFormat,
							Driver:      goodName,
							Pool:        goodName,
							Device:      goodName,
							AdminAccess: ptr.To(false),
						}},
					},
				}
				return claim
			},
			prioritizedListFeatureGate: true,
		},
		"invalid-add-allocation-with-sub-requests-no-corresponding-sub-request": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("request"), "foo/baz", "must be the name of a request in the claim or the name of a request and a subrequest separated by '/'"),
			},
			oldClaim: testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:     "foo/baz",
							Driver:      goodName,
							Pool:        goodName,
							Device:      goodName,
							AdminAccess: ptr.To(false),
						}},
					},
				}
				return claim
			},
			prioritizedListFeatureGate: true,
		},
		"invalid-add-allocation-with-sub-requests-invalid-request-names": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("request"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("request"), badSubrequestName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')"),
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("request"), badFullSubrequestName, "must be the name of a request in the claim or the name of a request and a subrequest separated by '/'"),
			},
			oldClaim: testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:     badFullSubrequestName,
							Driver:      goodName,
							Pool:        goodName,
							Device:      goodName,
							AdminAccess: ptr.To(false),
						}},
					},
				}
				return claim
			},
			prioritizedListFeatureGate: true,
		},
		"add-allocation-old-claim-with-prioritized-list": {
			wantFailures: nil,
			oldClaim:     testClaim(goodName, goodNS, validClaimSpecWithFirstAvailable),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:     "foo/bar",
							Driver:      goodName,
							Pool:        goodName,
							Device:      goodName,
							AdminAccess: ptr.To(false),
						}},
					},
				}
				return claim
			},
			prioritizedListFeatureGate: false,
		},
		"good-binding-conditions": {
			oldClaim: validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:                  goodName,
							Driver:                   goodName,
							Pool:                     goodName,
							Device:                   goodName,
							AdminAccess:              ptr.To(false),
							BindingConditions:        []string{"example.com/condition1", "condition2", "condition3", "condition4"},
							BindingFailureConditions: []string{"example.com/condition5", "condition6", "condition7", "condition8"},
						}},
					},
				}
				return claim
			},
		},
		"too-many-binding-conditions": {
			wantFailures: field.ErrorList{field.TooMany(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("bindingConditions"), 5, 4)},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:                  goodName,
							Driver:                   goodName,
							Pool:                     goodName,
							Device:                   goodName,
							AdminAccess:              ptr.To(false),
							BindingConditions:        []string{"condition1", "condition2", "condition3", "condition4", "condition5"},
							BindingFailureConditions: []string{"condition6", "condition7"},
						}},
					},
				}
				return claim
			},
		},
		"too-many-binding-failure-conditions": {
			wantFailures: field.ErrorList{field.TooMany(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("bindingFailureConditions"), 5, 4)},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:                  goodName,
							Driver:                   goodName,
							Pool:                     goodName,
							Device:                   goodName,
							AdminAccess:              ptr.To(false),
							BindingConditions:        []string{"condition1", "condition2"},
							BindingFailureConditions: []string{"condition3", "condition4", "condition5", "condition6", "condition7"},
						}},
					},
				}
				return claim
			},
		},
		"invalid-binding-conditions": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("bindingConditions").Index(1), "condition2!", conditionValidationErrMessage)},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:                  goodName,
							Driver:                   goodName,
							Pool:                     goodName,
							Device:                   goodName,
							AdminAccess:              ptr.To(false),
							BindingConditions:        []string{"condition1", "condition2!"},
							BindingFailureConditions: []string{"condition3", "condition4"},
						}},
					},
				}
				return claim
			},
		},
		"invalid-binding-failure-conditions": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("bindingFailureConditions").Index(1), "condition4!", conditionValidationErrMessage)},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:                  goodName,
							Driver:                   goodName,
							Pool:                     goodName,
							Device:                   goodName,
							AdminAccess:              ptr.To(false),
							BindingConditions:        []string{"condition1", "condition2"},
							BindingFailureConditions: []string{"condition3", "condition4!"},
						}},
					},
				}
				return claim
			},
		},
		"lacking-binding-conditions-claim-has-binding-failure-conditions": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("bindingConditions"), []string(nil), "bindingConditions are required to use bindingFailureConditions")},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:                  goodName,
							Driver:                   goodName,
							Pool:                     goodName,
							Device:                   goodName,
							AdminAccess:              ptr.To(false),
							BindingConditions:        nil,
							BindingFailureConditions: []string{"condition3", "condition4"},
						}},
					},
				}
				return claim
			},
		},
		"lacking-binding-failure-conditions-claim-has-binding-conditions": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("bindingFailureConditions"), []string(nil), "bindingFailureConditions are required to use bindingConditions")},
			oldClaim:     validClaim,
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:                  goodName,
							Driver:                   goodName,
							Pool:                     goodName,
							Device:                   goodName,
							AdminAccess:              ptr.To(false),
							BindingConditions:        []string{"condition1", "condition2"},
							BindingFailureConditions: nil,
						}},
					},
				}
				return claim
			},
		},
		"valid-add-allocation-with-consumable-capacity-feature-on": {
			oldClaim: testClaim(goodName, goodNS, validClaimSpec),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{{
							Request:     goodName,
							Driver:      goodName,
							Pool:        goodName,
							Device:      goodName,
							AdminAccess: ptr.To(false),
						}},
					},
				}
				return claim
			},
			consumableCapacityFeatureGate: true,
		},
		"valid-add-multi-alloc-devices": {
			oldClaim: testClaim(goodName, goodNS, func() resource.ResourceClaimSpec {
				spec := validClaimSpec.DeepCopy()
				spec.Devices.Requests[0].Exactly.Count = 2
				return *spec
			}()),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{
							{
								Request:     goodName,
								Driver:      goodName,
								Pool:        goodName,
								Device:      goodName,
								ShareID:     ptr.To(goodShareID),
								AdminAccess: ptr.To(false),
							},
							{
								Request:     goodName,
								Driver:      goodName,
								Pool:        goodName,
								Device:      goodName,
								ShareID:     ptr.To(goodShareID2),
								AdminAccess: ptr.To(false),
							},
						},
					},
				}
				return claim
			},
			consumableCapacityFeatureGate: true,
		},
		"valid-add-multi-alloc-devices-with-device-status": {
			oldClaim: testClaim(goodName, goodNS, func() resource.ResourceClaimSpec {
				spec := validClaimSpec.DeepCopy()
				spec.Devices.Requests[0].Exactly.Count = 2
				return *spec
			}()),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{
							{
								Request:     goodName,
								Driver:      goodName,
								Pool:        goodName,
								Device:      goodName,
								ShareID:     ptr.To(goodShareID),
								AdminAccess: ptr.To(false),
							},
							{
								Request:     goodName,
								Driver:      goodName,
								Pool:        goodName,
								Device:      goodName,
								ShareID:     ptr.To(goodShareID2),
								AdminAccess: ptr.To(false),
							},
						},
					},
				}
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver:  goodName,
						Pool:    goodName,
						Device:  goodName,
						ShareID: ptr.To(goodShareIDStr),
					},
					{
						Driver:  goodName,
						Pool:    goodName,
						Device:  goodName,
						ShareID: ptr.To(goodShareID2Str),
					},
				}
				return claim
			},
			consumableCapacityFeatureGate: true,
			deviceStatusFeatureGate:       true,
		},
		"valid-add-multi-alloc-devices-with-consumed-capacity": {
			oldClaim: testClaim(goodName, goodNS, validClaimSpec),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{
							{
								Request:     goodName,
								Driver:      goodName,
								Pool:        goodName,
								Device:      goodName,
								ShareID:     ptr.To(goodShareID),
								AdminAccess: ptr.To(false),
								ConsumedCapacity: map[resource.QualifiedName]apiresource.Quantity{
									goodName: apiresource.MustParse("1"),
								},
							},
						},
					},
				}
				return claim
			},
			consumableCapacityFeatureGate: true,
		},
		"valid-add-enable-both-device-status-and-consumable-capacity-features": {
			oldClaim: testClaim(goodName, goodNS, validClaimSpec),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{
							{
								Request:     goodName,
								Driver:      goodName,
								Pool:        goodName,
								Device:      goodName,
								ShareID:     ptr.To(goodShareID),
								AdminAccess: ptr.To(false),
							},
						},
					},
				}
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver:  goodName,
						Pool:    goodName,
						Device:  goodName,
						ShareID: ptr.To(goodShareIDStr),
					},
				}
				return claim
			},
			consumableCapacityFeatureGate: true,
			deviceStatusFeatureGate:       true,
		},
		"invalid-add-allocated-status-no-share-id": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "devices").Index(0), structured.MakeSharedDeviceID(structured.MakeDeviceID(goodName, goodName, goodName), nil), "must be an allocated device in the claim"),
			},
			oldClaim: testClaim(goodName, goodNS, validClaimSpec),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{
							{
								Request:     goodName,
								Driver:      goodName,
								Pool:        goodName,
								Device:      goodName,
								ShareID:     ptr.To(goodShareID),
								AdminAccess: ptr.To(false),
							},
						},
					},
				}
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver: goodName,
						Pool:   goodName,
						Device: goodName,
					},
				}
				return claim
			},
			consumableCapacityFeatureGate: true,
			deviceStatusFeatureGate:       true,
		},
		"invalid-add-allocated-status-mismatch-share-id": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "devices").Index(0), structured.MakeSharedDeviceID(structured.MakeDeviceID(goodName, goodName, goodName), ptr.To(goodShareID2)), "must be an allocated device in the claim"),
			},
			oldClaim: testClaim(goodName, goodNS, validClaimSpec),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{
							{
								Request:     goodName,
								Driver:      goodName,
								Pool:        goodName,
								Device:      goodName,
								ShareID:     ptr.To(goodShareID),
								AdminAccess: ptr.To(false),
							},
						},
					},
				}
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver:  goodName,
						Pool:    goodName,
						Device:  goodName,
						ShareID: &goodShareID2Str,
					},
				}
				return claim
			},
			consumableCapacityFeatureGate: true,
			deviceStatusFeatureGate:       true,
		},
		"invalid-add-allocated-status-invalid-share-id": {
			wantFailures: field.ErrorList{
				field.Invalid(field.NewPath("status", "devices").Index(0).Child("shareID"), badLengthShareIDStr, "error validating uid: invalid UUID length: 1"),
				field.Invalid(field.NewPath("status", "devices").Index(1).Child("shareID"), badFormatShareIDStr, "uid must be in RFC 4122 normalized form, `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` with lowercase hexadecimal characters"),
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(0).Child("shareID"), badLengthShareIDStr, "error validating uid: invalid UUID length: 1"),
				field.Invalid(field.NewPath("status", "allocation", "devices", "results").Index(1).Child("shareID"), badFormatShareIDStr, "uid must be in RFC 4122 normalized form, `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` with lowercase hexadecimal characters"),
			},
			oldClaim: testClaim(goodName, goodNS, validClaimSpec),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{
							{
								Request:     goodName,
								Driver:      goodName,
								Pool:        goodName,
								Device:      goodName,
								ShareID:     ptr.To(badLengthShareID),
								AdminAccess: ptr.To(false),
							},
							{
								Request:     goodName,
								Driver:      goodName,
								Pool:        goodName,
								Device:      goodName,
								ShareID:     ptr.To(badFormatShareID),
								AdminAccess: ptr.To(false),
							},
						},
					},
				}
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver:  goodName,
						Pool:    goodName,
						Device:  goodName,
						ShareID: &badLengthShareIDStr,
					},
					{
						Driver:  goodName,
						Pool:    goodName,
						Device:  goodName,
						ShareID: &badFormatShareIDStr,
					},
				}
				return claim
			},
			consumableCapacityFeatureGate: true,
			deviceStatusFeatureGate:       true,
		},
		"valid-add-allocated-status-with-share-id-disabled-feature-gate": {
			oldClaim: testClaim(goodName, goodNS, validClaimSpec),
			update: func(claim *resource.ResourceClaim) *resource.ResourceClaim {
				claim.Status.Allocation = &resource.AllocationResult{
					Devices: resource.DeviceAllocationResult{
						Results: []resource.DeviceRequestAllocationResult{
							{
								Request:     goodName,
								Driver:      goodName,
								Pool:        goodName,
								Device:      goodName,
								ShareID:     ptr.To(goodShareID),
								AdminAccess: ptr.To(false),
							},
						},
					},
				}
				claim.Status.Devices = []resource.AllocatedDeviceStatus{
					{
						Driver:  goodName,
						Pool:    goodName,
						Device:  "foo",
						ShareID: ptr.To(goodShareIDStr),
					},
				}
				return claim
			},
			consumableCapacityFeatureGate: false,
			deviceStatusFeatureGate:       true,
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAAdminAccess, scenario.adminAccess)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAResourceClaimDeviceStatus, scenario.deviceStatusFeatureGate)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPrioritizedList, scenario.prioritizedListFeatureGate)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAConsumableCapacity, scenario.consumableCapacityFeatureGate)

			scenario.oldClaim.ResourceVersion = "1"
			errs := ValidateResourceClaimStatusUpdate(scenario.update(scenario.oldClaim.DeepCopy()), scenario.oldClaim)
			assertFailures(t, scenario.wantFailures, errs)
		})
	}
}
