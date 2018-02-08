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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/storage"
)

func TestVolumeAttachmentStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Errorf("VolumeAttachment must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("VolumeAttachment should not allow create on update")
	}

	pvName := "name"
	volumeAttachment := &storage.VolumeAttachment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-attachment",
		},
		Spec: storage.VolumeAttachmentSpec{
			Attacher: "valid-attacher",
			Source: storage.VolumeAttachmentSource{
				PersistentVolumeName: &pvName,
			},
			NodeName: "valid-node",
		},
	}

	Strategy.PrepareForCreate(ctx, volumeAttachment)

	errs := Strategy.Validate(ctx, volumeAttachment)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	newVolumeAttachment := &storage.VolumeAttachment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-attachment-2",
		},
		Spec: storage.VolumeAttachmentSpec{
			Attacher: "valid-attacher-2",
			Source: storage.VolumeAttachmentSource{
				PersistentVolumeName: &pvName,
			},
			NodeName: "valid-node-2",
		},
	}

	Strategy.PrepareForUpdate(ctx, newVolumeAttachment, volumeAttachment)

	errs = Strategy.ValidateUpdate(ctx, newVolumeAttachment, volumeAttachment)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}

}
