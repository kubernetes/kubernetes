/*
Copyright 2023 The Kubernetes Authors.

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

package volumeattributesclass

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/storage"
)

func TestVolumeAttributesClassStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Errorf("VolumeAttributesClassStrategy must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("VolumeAttributesClassStrategy should not allow create on update")
	}

	class := &storage.VolumeAttributesClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-class",
		},
		DriverName: "fake",
		Parameters: map[string]string{
			"foo": "bar",
		},
	}

	Strategy.PrepareForCreate(ctx, class)

	errs := Strategy.Validate(ctx, class)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	newClass := &storage.VolumeAttributesClass{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "valid-class-2",
			ResourceVersion: "4",
		},
		DriverName: "fake",
		Parameters: map[string]string{
			"foo": "bar",
		},
	}

	Strategy.PrepareForUpdate(ctx, newClass, class)

	errs = Strategy.ValidateUpdate(ctx, newClass, class)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}
