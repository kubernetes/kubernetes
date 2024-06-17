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
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/storage"
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
		t.Errorf("unexpected objects difference after creating with status: %v", cmp.Diff(statusVolumeAttachment, volumeAttachment))
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
		t.Errorf("unexpected objects difference after modifying status: %v", cmp.Diff(statusVolumeAttachment, volumeAttachment))
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
	Strategy.PrepareForCreate(ctx, volumeAttachment)
	if volumeAttachment.Spec.Source.InlineVolumeSpec == nil {
		t.Errorf("InlineVolumeSpec unexpectedly set to nil during PrepareForCreate")
	}
	if !apiequality.Semantic.DeepEqual(volumeAttachmentSaved, volumeAttachment) {
		t.Errorf("unexpected difference in object after creation: %v", cmp.Diff(volumeAttachment, volumeAttachmentSaved))
	}
	Strategy.PrepareForUpdate(ctx, volumeAttachmentSaved, volumeAttachment)
	if volumeAttachmentSaved.Spec.Source.InlineVolumeSpec == nil {
		t.Errorf("InlineVolumeSpec unexpectedly set to nil during PrepareForUpdate")
	}
	Strategy.PrepareForUpdate(ctx, volumeAttachmentSaved, volumeAttachment)
	if volumeAttachmentSaved.Spec.Source.InlineVolumeSpec == nil {
		t.Errorf("InlineVolumeSpec unexpectedly set to nil during PrepareForUpdate")
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
		t.Errorf("unexpected objects difference after modifying status: %v", cmp.Diff(statusVolumeAttachment, expectedVolumeAttachment))
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
		t.Errorf("unexpected objects difference after modifying spec: %v", cmp.Diff(newVolumeAttachment, volumeAttachment))
	}
}

func TestUpdatePreventsStatusWrite(t *testing.T) {
	va := getValidVolumeAttachment("valid-attachment")
	newAttachment := va.DeepCopy()
	newAttachment.Status.Attached = true
	Strategy.PrepareForUpdate(context.TODO(), newAttachment, va)
	if newAttachment.Status.Attached {
		t.Errorf("expected status to be %v got %v", false, newAttachment.Status.Attached)
	}
}

func TestCreatePreventsStatusWrite(t *testing.T) {
	va := getValidVolumeAttachment("valid-attachment")
	va.Status.Attached = true
	Strategy.PrepareForCreate(context.TODO(), va)
	if va.Status.Attached {
		t.Errorf("expected status to be false got %v", va.Status.Attached)
	}
}

func TestVolumeAttachmentValidation(t *testing.T) {
	invalidPVName := "invalid-!@#$%^&*()"
	validPVName := "valid-volume-name"
	tests := []struct {
		name             string
		volumeAttachment *storage.VolumeAttachment
		expectError      bool
	}{
		{
			"valid attachment",
			getValidVolumeAttachment("foo"),
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
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := Strategy.Validate(context.TODO(), test.volumeAttachment)
			if len(err) > 0 && !test.expectError {
				t.Errorf("Validation of object failed: %+v", err)
			}
			if len(err) == 0 && test.expectError {
				t.Errorf("Validation of object unexpectedly succeeded")
			}
		})
	}
}
