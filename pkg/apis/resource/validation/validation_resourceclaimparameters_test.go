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

func testResourceClaimParameters(name, namespace string, requests []resource.DriverRequests) *resource.ResourceClaimParameters {
	return &resource.ResourceClaimParameters{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		DriverRequests: requests,
	}
}

var goodRequests []resource.DriverRequests

func TestValidateResourceClaimParameters(t *testing.T) {
	goodName := "foo"
	badName := "!@#$%^"
	badValue := "spaces not allowed"
	now := metav1.Now()

	scenarios := map[string]struct {
		parameters   *resource.ResourceClaimParameters
		wantFailures field.ErrorList
	}{
		"good": {
			parameters: testResourceClaimParameters(goodName, goodName, goodRequests),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			parameters:   testResourceClaimParameters("", goodName, goodRequests),
		},
		"missing-namespace": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "namespace"), "")},
			parameters:   testResourceClaimParameters(goodName, "", goodRequests),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			parameters:   testResourceClaimParameters(badName, goodName, goodRequests),
		},
		"bad-namespace": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "namespace"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')")},
			parameters:   testResourceClaimParameters(goodName, badName, goodRequests),
		},
		"generate-name": {
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.GenerateName = "prefix-"
				return parameters
			}(),
		},
		"uid": {
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return parameters
			}(),
		},
		"resource-version": {
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.ResourceVersion = "1"
				return parameters
			}(),
		},
		"generation": {
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.Generation = 100
				return parameters
			}(),
		},
		"creation-timestamp": {
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.CreationTimestamp = now
				return parameters
			}(),
		},
		"deletion-grace-period-seconds": {
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.DeletionGracePeriodSeconds = ptr.To[int64](10)
				return parameters
			}(),
		},
		"owner-references": {
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: "v1",
						Kind:       "pod",
						Name:       "foo",
						UID:        "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d",
					},
				}
				return parameters
			}(),
		},
		"finalizers": {
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.Finalizers = []string{
					"example.com/foo",
				}
				return parameters
			}(),
		},
		"managed-fields": {
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						FieldsType: "FieldsV1",
						Operation:  "Apply",
						APIVersion: "apps/v1",
						Manager:    "foo",
					},
				}
				return parameters
			}(),
		},
		"good-labels": {
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return parameters
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.Labels = map[string]string{
					"hello-world": badValue,
				}
				return parameters
			}(),
		},
		"good-annotations": {
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.Annotations = map[string]string{
					"foo": "bar",
				}
				return parameters
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.Annotations = map[string]string{
					badName: "hello world",
				}
				return parameters
			}(),
		},

		"empty-model": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("driverRequests").Index(0).Child("requests").Index(0), "exactly one structured model field must be set")},
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.DriverRequests = []resource.DriverRequests{{DriverName: goodName, Requests: []resource.ResourceRequest{{}}}}
				return parameters
			}(),
		},

		"empty-requests": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("driverRequests").Index(0).Child("requests"), "empty entries with no requests are not allowed")},
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.DriverRequests = []resource.DriverRequests{{DriverName: goodName}}
				return parameters
			}(),
		},

		"invalid-driver": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("driverRequests").Index(1).Child("driverName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.DriverRequests = []resource.DriverRequests{
					{
						DriverName: goodName,
						Requests: []resource.ResourceRequest{
							{
								ResourceRequestModel: resource.ResourceRequestModel{
									NamedResources: &resource.NamedResourcesRequest{Selector: "true"},
								},
							},
						},
					},
					{
						DriverName: badName,
						Requests: []resource.ResourceRequest{
							{
								ResourceRequestModel: resource.ResourceRequestModel{
									NamedResources: &resource.NamedResourcesRequest{Selector: "true"},
								},
							},
						},
					},
				}
				return parameters
			}(),
		},

		"duplicate-driver": {
			wantFailures: field.ErrorList{field.Duplicate(field.NewPath("driverRequests").Index(1).Child("driverName"), goodName)},
			parameters: func() *resource.ResourceClaimParameters {
				parameters := testResourceClaimParameters(goodName, goodName, goodRequests)
				parameters.DriverRequests = []resource.DriverRequests{
					{
						DriverName: goodName,
						Requests: []resource.ResourceRequest{
							{
								ResourceRequestModel: resource.ResourceRequestModel{
									NamedResources: &resource.NamedResourcesRequest{Selector: "true"},
								},
							},
						},
					},
					{
						DriverName: goodName,
						Requests: []resource.ResourceRequest{
							{
								ResourceRequestModel: resource.ResourceRequestModel{
									NamedResources: &resource.NamedResourcesRequest{Selector: "true"},
								},
							},
						},
					},
				}
				return parameters
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateResourceClaimParameters(scenario.parameters)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateResourceClaimParametersUpdate(t *testing.T) {
	name := "valid"
	validResourceClaimParameters := testResourceClaimParameters(name, name, nil)

	scenarios := map[string]struct {
		oldResourceClaimParameters *resource.ResourceClaimParameters
		update                     func(claim *resource.ResourceClaimParameters) *resource.ResourceClaimParameters
		wantFailures               field.ErrorList
	}{
		"valid-no-op-update": {
			oldResourceClaimParameters: validResourceClaimParameters,
			update:                     func(claim *resource.ResourceClaimParameters) *resource.ResourceClaimParameters { return claim },
		},
		"invalid-name-update": {
			oldResourceClaimParameters: validResourceClaimParameters,
			update: func(claim *resource.ResourceClaimParameters) *resource.ResourceClaimParameters {
				claim.Name += "-update"
				return claim
			},
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), name+"-update", "field is immutable")},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldResourceClaimParameters.ResourceVersion = "1"
			errs := ValidateResourceClaimParametersUpdate(scenario.update(scenario.oldResourceClaimParameters.DeepCopy()), scenario.oldResourceClaimParameters)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}
