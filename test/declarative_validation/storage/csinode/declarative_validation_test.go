/*
Copyright The Kubernetes Authors.

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

package csinode

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	storage "k8s.io/kubernetes/pkg/apis/storage"
	registry "k8s.io/kubernetes/pkg/registry/storage/csinode"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "storage.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "csinodes",
		IsResourceRequest: true,
		Verb:              "create",
	})

	obj := mkCSINode()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "storage.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "csinodes",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	})

	updateObj := mkCSINode()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy,
		meta.WithStringentFinalizerValidation())
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIPrefix:         "apis",
			APIGroup:          "storage.k8s.io",
			APIVersion:        apiVersion,
			Resource:          "csinodes",
			Subresource:       "status",
			Name:              "valid-obj",
			IsResourceRequest: true,
			Verb:              "update",
		})

		conditionsPath := field.NewPath("status", "storageHealth").Index(0).Child("healthConditions")
		conditionReasonPath := field.NewPath("status", "storageHealth").Index(0).Child("healthConditions").Index(0).Child("reason")
		conditionMessagePath := field.NewPath("status", "storageHealth").Index(0).Child("healthConditions").Index(0).Child("message")
		conditionStatusTypePath := field.NewPath("status", "storageHealth").Index(0).Child("healthConditions").Index(0).Child("status")

		validStorageHealthStatusTypes := []storage.StorageHealthStatusType{
			storage.StorageDegraded,
			storage.StorageUnreachable,
		}
		tests := map[string]struct {
			storageHealth []storage.StorageHealth
			expectedErrs  field.ErrorList
		}{
			"valid": {
				storageHealth: []storage.StorageHealth{{Name: "foo"}},
			},
			"name required": {
				storageHealth: []storage.StorageHealth{{}},
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("status", "storageHealth").Index(0).Child("name"), "").MarkAlpha(),
				},
			},
			"reason valid": {
				storageHealth: []storage.StorageHealth{{
					Name: "foo",
					HealthConditions: []storage.StorageHealthCondition{{
						Status: storage.StorageUnreachable,
						Reason: "BackendDown",
					}},
				}},
			},
			"reason is required": {
				storageHealth: []storage.StorageHealth{{
					Name: "foo",
					HealthConditions: []storage.StorageHealthCondition{{
						Status: storage.StorageUnreachable,
						Reason: "",
					}},
				}},
				expectedErrs: field.ErrorList{
					field.Required(conditionReasonPath, "").MarkAlpha(),
				},
			},
			"reason length can not be greater than 256": {
				storageHealth: []storage.StorageHealth{{
					Name: "foo",
					HealthConditions: []storage.StorageHealthCondition{{
						Status: storage.StorageUnreachable,
						Reason: strings.Repeat("a", 257),
					}},
				}},
				expectedErrs: field.ErrorList{
					field.TooLong(conditionReasonPath, "", 256).WithOrigin("maxBytes").MarkAlpha(),
				},
			},
			"message length can not be greater than 1024": {
				storageHealth: []storage.StorageHealth{{
					Name: "foo",
					HealthConditions: []storage.StorageHealthCondition{{
						Status:  storage.StorageUnreachable,
						Reason:  "BackendDown",
						Message: strings.Repeat("a", 1025),
					}},
				}},
				expectedErrs: field.ErrorList{
					field.TooLong(conditionMessagePath, "", 1024).WithOrigin("maxBytes").MarkAlpha(),
				},
			},
			"reason format is invalid": {
				storageHealth: []storage.StorageHealth{{
					Name: "foo",
					HealthConditions: []storage.StorageHealthCondition{{
						Status: storage.StorageUnreachable,
						Reason: "1-not-valid!",
					}},
				}},
				expectedErrs: field.ErrorList{
					field.Invalid(conditionReasonPath, "", "").WithOrigin("format=k8s-condition-reason").MarkAlpha(),
				},
			},
			"storage health status type is required": {
				storageHealth: []storage.StorageHealth{{
					Name: "foo",
					HealthConditions: []storage.StorageHealthCondition{{
						Reason: "BackendDown",
					}},
				}},
				expectedErrs: field.ErrorList{
					field.Required(conditionStatusTypePath, "").MarkAlpha(),
				},
			},
			"invalid storage health status type": {
				storageHealth: []storage.StorageHealth{{
					Name: "foo",
					HealthConditions: []storage.StorageHealthCondition{{
						Status: "badvalue",
						Reason: "BackendDown",
					}},
				}},
				expectedErrs: field.ErrorList{
					field.NotSupported(conditionStatusTypePath, "", validStorageHealthStatusTypes).MarkAlpha(),
				},
			},
			"16 health conditions is allowed": {
				storageHealth: mkStorageHealthWithValidConditions(16),
			},
			"more than 16 health conditions": {
				storageHealth: mkStorageHealthWithValidConditions(17),
				expectedErrs: field.ErrorList{
					field.TooMany(conditionsPath, 17, 16).WithOrigin("maxItems").MarkAlpha(),
				},
			},
			"more than 16 health conditions with one invalid condition": {
				storageHealth: mkStorageHealthWithOneInvalidCondition(17),
				expectedErrs: field.ErrorList{
					field.TooMany(conditionsPath, 17, 16).WithOrigin("maxItems").MarkAlpha(),
				},
			},
		}

		for name, tc := range tests {
			t.Run(apiVersion+"/"+name, func(t *testing.T) {
				oldObj := mkCSINode()
				oldObj.ResourceVersion = "1"
				updateObj := oldObj.DeepCopy()
				updateObj.Status.StorageHealth = tc.storageHealth
				apitesting.VerifyUpdateValidationEquivalence(t, ctx, updateObj, &oldObj,
					registry.StatusStrategy, tc.expectedErrs,
					apitesting.WithSubResources("status"))
			})
		}
	}
}

// mkStorageHealthWithValidConditions returns one StorageHealth entry carrying n valid conditions.
func mkStorageHealthWithValidConditions(n int) []storage.StorageHealth {
	conditions := make([]storage.StorageHealthCondition, n)
	for i := range conditions {
		conditions[i] = storage.StorageHealthCondition{
			Status: storage.StorageUnreachable,
			Reason: "BackendDown",
		}
	}
	return []storage.StorageHealth{{Name: "foo", HealthConditions: conditions}}
}

// mkStorageHealthWithOneInvalidCondition returns n valid conditions with the first one missing its status.
func mkStorageHealthWithOneInvalidCondition(n int) []storage.StorageHealth {
	health := mkStorageHealthWithValidConditions(n)
	health[0].HealthConditions[0].Status = ""
	return health
}

func mkCSINode(tweaks ...func(node *storage.CSINode)) storage.CSINode {
	node := storage.CSINode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-obj",
		},
		Spec: storage.CSINodeSpec{
			Drivers: []storage.CSINodeDriver{
				{
					Name:   "foo",
					NodeID: "bar",
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&node)
	}
	return node
}
