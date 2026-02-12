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

package csidriver

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/storage"
)

func TestDeclarativeValidateUpdate(t *testing.T) {
	testCases := map[string]struct {
		oldObj       storage.CSIDriver
		updateObj    storage.CSIDriver
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    makeValidCSIDriver(),
			updateObj: makeValidCSIDriver(),
		},
		"invalid update: attachRequired changed": {
			oldObj:    makeValidCSIDriver(),
			updateObj: makeValidCSIDriver(tweakAttachRequired(false)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "attachRequired"), boolPtr(false), "field is immutable").WithOrigin("immutable"),
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "storage.k8s.io",
				APIVersion:        "v1",
				Resource:          "csidrivers",
				Name:              "test-driver",
				IsResourceRequest: true,
				Verb:              "update",
			})
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func makeValidCSIDriver(mutators ...func(*storage.CSIDriver)) storage.CSIDriver {
	trueVal := true
	falseVal := false
	driver := storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-driver",
		},
		Spec: storage.CSIDriverSpec{
			AttachRequired:    &trueVal,
			PodInfoOnMount:    &trueVal,
			RequiresRepublish: &falseVal,
			StorageCapacity:   &falseVal,
			SELinuxMount:      &falseVal,
		},
	}
	for _, mutate := range mutators {
		mutate(&driver)
	}
	return driver
}

func tweakAttachRequired(val bool) func(*storage.CSIDriver) {
	return func(d *storage.CSIDriver) {
		d.Spec.AttachRequired = &val
	}
}

func boolPtr(b bool) *bool {
	return &b
}
