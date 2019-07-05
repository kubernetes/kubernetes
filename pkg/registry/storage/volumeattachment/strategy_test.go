/*
Copyright 2017 The Kubernetes Authors.

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

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/features"
)

func getValidVolumeAttachment(name string) *storage.VolumeAttachment {
	return &storage.VolumeAttachment{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: storage.VolumeAttachmentSpec{
			Attacher: "valid-attacher",
			Source: storage.VolumeAttachmentSource{
				PersistentVolumeName: &name,
			},
			NodeName: "valid-node",
		},
	}
}

func getValidVolumeAttachmentWithInlineSpec(name string) *storage.VolumeAttachment {
	volumeAttachment := getValidVolumeAttachment(name)
	volumeAttachment.Spec.Source.PersistentVolumeName = nil
	volumeAttachment.Spec.Source.InlineVolumeSpec = &api.PersistentVolumeSpec{
		Capacity: api.ResourceList{
			api.ResourceName(api.ResourceStorage): resource.MustParse("10"),
		},
		AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
		PersistentVolumeSource: api.PersistentVolumeSource{
			CSI: &api.CSIPersistentVolumeSource{
				Driver:       "com.test.foo",
				VolumeHandle: name,
			},
		},
		MountOptions: []string{"soft"},
	}
	return volumeAttachment
}

func TestVolumeAttachmentStrategy(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1",
		Resource:   "volumeattachments",
	})
	if Strategy.NamespaceScoped() {
		t.Errorf("VolumeAttachment must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("VolumeAttachment should not allow create on update")
	}

	volumeAttachment := getValidVolumeAttachment("valid-attachment")

	Strategy.PrepareForCreate(ctx, volumeAttachment)

	errs := Strategy.Validate(ctx, volumeAttachment)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	// Create with status should drop status
	statusVolumeAttachment := volumeAttachment.DeepCopy()
	statusVolumeAttachment.Status = storage.VolumeAttachmentStatus{Attached: true}
	Strategy.PrepareForCreate(ctx, statusVolumeAttachment)
	if !apiequality.Semantic.DeepEqual(statusVolumeAttachment, volumeAttachment) {
		t.Errorf("unexpected objects difference after creating with status: %v", diff.ObjectDiff(statusVolumeAttachment, volumeAttachment))
	}

	// Update of spec is disallowed
	newVolumeAttachment := volumeAttachment.DeepCopy()
	newVolumeAttachment.Spec.NodeName = "valid-node-2"

	Strategy.PrepareForUpdate(ctx, newVolumeAttachment, volumeAttachment)

	errs = Strategy.ValidateUpdate(ctx, newVolumeAttachment, volumeAttachment)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}

	// modifying status should be dropped
	statusVolumeAttachment = volumeAttachment.DeepCopy()
	statusVolumeAttachment.Status = storage.VolumeAttachmentStatus{Attached: true}

	Strategy.PrepareForUpdate(ctx, statusVolumeAttachment, volumeAttachment)

	if !apiequality.Semantic.DeepEqual(statusVolumeAttachment, volumeAttachment) {
		t.Errorf("unexpected objects difference after modfying status: %v", diff.ObjectDiff(statusVolumeAttachment, volumeAttachment))
	}
}

func TestVolumeAttachmentStrategySourceInlineSpec(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1",
		Resource:   "volumeattachments",
	})

	volumeAttachment := getValidVolumeAttachmentWithInlineSpec("valid-attachment")
	volumeAttachmentSaved := volumeAttachment.DeepCopy()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, true)()
	Strategy.PrepareForCreate(ctx, volumeAttachment)
	if volumeAttachment.Spec.Source.InlineVolumeSpec == nil {
		t.Errorf("InlineVolumeSpec unexpectedly set to nil during PrepareForCreate")
	}
	if !apiequality.Semantic.DeepEqual(volumeAttachmentSaved, volumeAttachment) {
		t.Errorf("unexpected difference in object after creation: %v", diff.ObjectDiff(volumeAttachment, volumeAttachmentSaved))
	}
	Strategy.PrepareForUpdate(ctx, volumeAttachmentSaved, volumeAttachment)
	if volumeAttachmentSaved.Spec.Source.InlineVolumeSpec == nil {
		t.Errorf("InlineVolumeSpec unexpectedly set to nil during PrepareForUpdate")
	}
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, false)()
	Strategy.PrepareForUpdate(ctx, volumeAttachmentSaved, volumeAttachment)
	if volumeAttachmentSaved.Spec.Source.InlineVolumeSpec == nil {
		t.Errorf("InlineVolumeSpec unexpectedly set to nil during PrepareForUpdate")
	}

	volumeAttachment = getValidVolumeAttachmentWithInlineSpec("valid-attachment")
	volumeAttachmentNew := volumeAttachment.DeepCopy()
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, false)()
	Strategy.PrepareForCreate(ctx, volumeAttachment)
	if volumeAttachment.Spec.Source.InlineVolumeSpec != nil {
		t.Errorf("InlineVolumeSpec unexpectedly not dropped during PrepareForCreate")
	}
	Strategy.PrepareForUpdate(ctx, volumeAttachmentNew, volumeAttachment)
	if volumeAttachmentNew.Spec.Source.InlineVolumeSpec != nil {
		t.Errorf("InlineVolumeSpec unexpectedly not dropped during PrepareForUpdate")
	}
}

func TestVolumeAttachmentStatusStrategy(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
		APIGroup:   "storage.k8s.io",
		APIVersion: "v1",
		Resource:   "volumeattachments",
	})

	volumeAttachment := getValidVolumeAttachment("valid-attachment")

	// modifying status should be allowed
	statusVolumeAttachment := volumeAttachment.DeepCopy()
	statusVolumeAttachment.Status = storage.VolumeAttachmentStatus{Attached: true}

	expectedVolumeAttachment := statusVolumeAttachment.DeepCopy()
	StatusStrategy.PrepareForUpdate(ctx, statusVolumeAttachment, volumeAttachment)
	if !apiequality.Semantic.DeepEqual(statusVolumeAttachment, expectedVolumeAttachment) {
		t.Errorf("unexpected objects differerence after modifying status: %v", diff.ObjectDiff(statusVolumeAttachment, expectedVolumeAttachment))
	}

	// spec and metadata modifications should be dropped
	newVolumeAttachment := volumeAttachment.DeepCopy()
	newVolumeAttachment.Spec.NodeName = "valid-node-2"
	newVolumeAttachment.Labels = map[string]string{"foo": "bar"}
	newVolumeAttachment.Annotations = map[string]string{"foo": "baz"}
	newVolumeAttachment.OwnerReferences = []metav1.OwnerReference{
		{
			APIVersion: "v1",
			Kind:       "Pod",
			Name:       "Foo",
		},
	}

	StatusStrategy.PrepareForUpdate(ctx, newVolumeAttachment, volumeAttachment)
	if !apiequality.Semantic.DeepEqual(newVolumeAttachment, volumeAttachment) {
		t.Errorf("unexpected objects differerence after modifying spec: %v", diff.ObjectDiff(newVolumeAttachment, volumeAttachment))
	}
}

func TestBetaAndV1StatusUpdate(t *testing.T) {
	tests := []struct {
		requestInfo    genericapirequest.RequestInfo
		newStatus      bool
		expectedStatus bool
	}{
		{
			genericapirequest.RequestInfo{
				APIGroup:   "storage.k8s.io",
				APIVersion: "v1",
				Resource:   "volumeattachments",
			},
			true,
			false,
		},
		{
			genericapirequest.RequestInfo{
				APIGroup:   "storage.k8s.io",
				APIVersion: "v1beta1",
				Resource:   "volumeattachments",
			},
			true,
			true,
		},
	}
	for _, test := range tests {
		va := getValidVolumeAttachment("valid-attachment")
		newAttachment := va.DeepCopy()
		newAttachment.Status.Attached = test.newStatus
		context := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &test.requestInfo)
		Strategy.PrepareForUpdate(context, newAttachment, va)
		if newAttachment.Status.Attached != test.expectedStatus {
			t.Errorf("expected status to be %v got %v", test.expectedStatus, newAttachment.Status.Attached)
		}
	}

}

func TestBetaAndV1StatusCreate(t *testing.T) {
	tests := []struct {
		requestInfo    genericapirequest.RequestInfo
		newStatus      bool
		expectedStatus bool
	}{
		{
			genericapirequest.RequestInfo{
				APIGroup:   "storage.k8s.io",
				APIVersion: "v1",
				Resource:   "volumeattachments",
			},
			true,
			false,
		},
		{
			genericapirequest.RequestInfo{
				APIGroup:   "storage.k8s.io",
				APIVersion: "v1beta1",
				Resource:   "volumeattachments",
			},
			true,
			true,
		},
	}
	for _, test := range tests {
		va := getValidVolumeAttachment("valid-attachment")
		va.Status.Attached = test.newStatus
		context := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &test.requestInfo)
		Strategy.PrepareForCreate(context, va)
		if va.Status.Attached != test.expectedStatus {
			t.Errorf("expected status to be %v got %v", test.expectedStatus, va.Status.Attached)
		}
	}
}

func TestVolumeAttachmentValidation(t *testing.T) {
	invalidPVName := "invalid-!@#$%^&*()"
	validPVName := "valid-volume-name"
	tests := []struct {
		name             string
		volumeAttachment *storage.VolumeAttachment
		expectBetaError  bool
		expectV1Error    bool
	}{
		{
			"valid attachment",
			getValidVolumeAttachment("foo"),
			false,
			false,
		},
		{
			"invalid PV name",
			&storage.VolumeAttachment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.VolumeAttachmentSpec{
					Attacher: "valid-attacher",
					Source: storage.VolumeAttachmentSource{
						PersistentVolumeName: &invalidPVName,
					},
					NodeName: "valid-node",
				},
			},
			false,
			true,
		},
		{
			"invalid attacher name",
			&storage.VolumeAttachment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.VolumeAttachmentSpec{
					Attacher: "invalid!@#$%^&*()",
					Source: storage.VolumeAttachmentSource{
						PersistentVolumeName: &validPVName,
					},
					NodeName: "valid-node",
				},
			},
			false,
			true,
		},
		{
			"invalid volume attachment",
			&storage.VolumeAttachment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: storage.VolumeAttachmentSpec{
					Attacher: "invalid!@#$%^&*()",
					Source: storage.VolumeAttachmentSource{
						PersistentVolumeName: nil,
					},
					NodeName: "valid-node",
				},
			},
			true,
			true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			testValidation := func(va *storage.VolumeAttachment, apiVersion string) field.ErrorList {
				ctx := genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{
					APIGroup:   "storage.k8s.io",
					APIVersion: apiVersion,
					Resource:   "volumeattachments",
				})
				return Strategy.Validate(ctx, va)
			}

			v1Err := testValidation(test.volumeAttachment, "v1")
			if len(v1Err) > 0 && !test.expectV1Error {
				t.Errorf("Validation of v1 object failed: %+v", v1Err)
			}
			if len(v1Err) == 0 && test.expectV1Error {
				t.Errorf("Validation of v1 object unexpectedly succeeded")
			}

			betaErr := testValidation(test.volumeAttachment, "v1beta1")
			if len(betaErr) > 0 && !test.expectBetaError {
				t.Errorf("Validation of v1beta1 object failed: %+v", betaErr)
			}
			if len(betaErr) == 0 && test.expectBetaError {
				t.Errorf("Validation of v1beta1 object unexpectedly succeeded")
			}
		})
	}
}
