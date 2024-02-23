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

func testResourceClassParameters(name, namespace string, filters []resource.ResourceFilter) *resource.ResourceClassParameters {
	return &resource.ResourceClassParameters{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Filters: filters,
	}
}

var goodFilters []resource.ResourceFilter

func TestValidateResourceClassParameters(t *testing.T) {
	goodName := "foo"
	badName := "!@#$%^"
	badValue := "spaces not allowed"
	now := metav1.Now()

	scenarios := map[string]struct {
		parameters   *resource.ResourceClassParameters
		wantFailures field.ErrorList
	}{
		"good": {
			parameters: testResourceClassParameters(goodName, goodName, goodFilters),
		},
		"missing-name": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "name"), "name or generateName is required")},
			parameters:   testResourceClassParameters("", goodName, goodFilters),
		},
		"missing-namespace": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("metadata", "namespace"), "")},
			parameters:   testResourceClassParameters(goodName, "", goodFilters),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			parameters:   testResourceClassParameters(badName, goodName, goodFilters),
		},
		"bad-namespace": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "namespace"), badName, "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')")},
			parameters:   testResourceClassParameters(goodName, badName, goodFilters),
		},
		"generate-name": {
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.GenerateName = "prefix-"
				return parameters
			}(),
		},
		"uid": {
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.UID = "ac051fac-2ead-46d9-b8b4-4e0fbeb7455d"
				return parameters
			}(),
		},
		"resource-version": {
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.ResourceVersion = "1"
				return parameters
			}(),
		},
		"generation": {
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.Generation = 100
				return parameters
			}(),
		},
		"creation-timestamp": {
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.CreationTimestamp = now
				return parameters
			}(),
		},
		"deletion-grace-period-seconds": {
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.DeletionGracePeriodSeconds = ptr.To[int64](10)
				return parameters
			}(),
		},
		"owner-references": {
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
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
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.Finalizers = []string{
					"example.com/foo",
				}
				return parameters
			}(),
		},
		"managed-fields": {
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
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
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.Labels = map[string]string{
					"apps.kubernetes.io/name": "test",
				}
				return parameters
			}(),
		},
		"bad-labels": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "labels"), badValue, "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyValue',  or 'my_value',  or '12345', regex used for validation is '(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?')")},
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.Labels = map[string]string{
					"hello-world": badValue,
				}
				return parameters
			}(),
		},
		"good-annotations": {
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.Annotations = map[string]string{
					"foo": "bar",
				}
				return parameters
			}(),
		},
		"bad-annotations": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "annotations"), badName, "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')")},
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.Annotations = map[string]string{
					badName: "hello world",
				}
				return parameters
			}(),
		},

		"empty-model": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("filters").Index(0), "exactly one structured model field must be set")},
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.Filters = []resource.ResourceFilter{{DriverName: goodName}}
				return parameters
			}(),
		},

		"filters-invalid-driver": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("filters").Index(1).Child("driverName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.Filters = []resource.ResourceFilter{
					{
						DriverName: goodName,
						ResourceFilterModel: resource.ResourceFilterModel{
							NamedResources: &resource.NamedResourcesFilter{Selector: "true"},
						},
					},
					{
						DriverName: badName,
						ResourceFilterModel: resource.ResourceFilterModel{
							NamedResources: &resource.NamedResourcesFilter{Selector: "true"},
						},
					},
				}
				return parameters
			}(),
		},

		"filters-duplicate-driver": {
			wantFailures: field.ErrorList{field.Duplicate(field.NewPath("filters").Index(1).Child("driverName"), goodName)},
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.Filters = []resource.ResourceFilter{
					{
						DriverName: goodName,
						ResourceFilterModel: resource.ResourceFilterModel{
							NamedResources: &resource.NamedResourcesFilter{Selector: "true"},
						},
					},
					{
						DriverName: goodName,
						ResourceFilterModel: resource.ResourceFilterModel{
							NamedResources: &resource.NamedResourcesFilter{Selector: "true"},
						},
					},
				}
				return parameters
			}(),
		},

		"parameters-invalid-driver": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("vendorParameters").Index(1).Child("driverName"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.VendorParameters = []resource.VendorParameters{
					{
						DriverName: goodName,
					},
					{
						DriverName: badName,
					},
				}
				return parameters
			}(),
		},

		"parameters-duplicate-driver": {
			wantFailures: field.ErrorList{field.Duplicate(field.NewPath("vendorParameters").Index(1).Child("driverName"), goodName)},
			parameters: func() *resource.ResourceClassParameters {
				parameters := testResourceClassParameters(goodName, goodName, goodFilters)
				parameters.VendorParameters = []resource.VendorParameters{
					{
						DriverName: goodName,
					},
					{
						DriverName: goodName,
					},
				}
				return parameters
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateResourceClassParameters(scenario.parameters)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateResourceClassParametersUpdate(t *testing.T) {
	name := "valid"
	validResourceClassParameters := testResourceClassParameters(name, name, nil)

	scenarios := map[string]struct {
		oldResourceClassParameters *resource.ResourceClassParameters
		update                     func(class *resource.ResourceClassParameters) *resource.ResourceClassParameters
		wantFailures               field.ErrorList
	}{
		"valid-no-op-update": {
			oldResourceClassParameters: validResourceClassParameters,
			update:                     func(class *resource.ResourceClassParameters) *resource.ResourceClassParameters { return class },
		},
		"invalid-name-update": {
			oldResourceClassParameters: validResourceClassParameters,
			update: func(class *resource.ResourceClassParameters) *resource.ResourceClassParameters {
				class.Name += "-update"
				return class
			},
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("metadata", "name"), name+"-update", "field is immutable")},
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			scenario.oldResourceClassParameters.ResourceVersion = "1"
			errs := ValidateResourceClassParametersUpdate(scenario.update(scenario.oldResourceClassParameters.DeepCopy()), scenario.oldResourceClassParameters)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}
