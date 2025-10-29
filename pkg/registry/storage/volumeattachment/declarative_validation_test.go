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

package volumeattachment

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	storage "k8s.io/kubernetes/pkg/apis/storage"
)

func TestDeclarativeValidate(t *testing.T) {
	// VolumeAttachment existed as v1beta1 and v1
	apiVersions := []string{"v1beta1", "v1"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "storage.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "volumeattachments",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        storage.VolumeAttachment
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidVolumeAttachment(),
		},
		"invalid attacher (required)": {
			input: mkValidVolumeAttachment(func(obj *storage.VolumeAttachment) {
				obj.Spec.Attacher = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "attacher"), ""),
			},
		},
		// TODO: Add more test cases (e.g., attacher format constraints) if desired.
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func mkValidVolumeAttachment(tweaks ...func(obj *storage.VolumeAttachment)) storage.VolumeAttachment {
	pvName := "pv-001"
	obj := storage.VolumeAttachment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-volume-attachment",
		},
		Spec: storage.VolumeAttachmentSpec{
			// Use an in-tree style qualified name to satisfy legacy validation rules.
			Attacher: "example.com",
			Source: storage.VolumeAttachmentSource{
				PersistentVolumeName: &pvName,
			},
			NodeName: "node-1",
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
