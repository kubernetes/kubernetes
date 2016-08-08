/*
Copyright 2015 The Kubernetes Authors.

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

package storageclass

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func TestStorageClassStrategy(t *testing.T) {
	ctx := api.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Errorf("StorageClass must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("StorageClass should not allow create on update")
	}

	storageClass := &extensions.StorageClass{
		ObjectMeta: api.ObjectMeta{
			Name: "valid-class",
		},
		Provisioner: "kubernetes.io/aws-ebs",
		Parameters: map[string]string{
			"foo": "bar",
		},
	}

	Strategy.PrepareForCreate(ctx, storageClass)

	errs := Strategy.Validate(ctx, storageClass)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	newStorageClass := &extensions.StorageClass{
		ObjectMeta: api.ObjectMeta{
			Name:            "valid-class-2",
			ResourceVersion: "4",
		},
		Provisioner: "kubernetes.io/aws-ebs",
		Parameters: map[string]string{
			"foo": "bar",
		},
	}

	Strategy.PrepareForUpdate(ctx, newStorageClass, storageClass)

	errs = Strategy.ValidateUpdate(ctx, newStorageClass, storageClass)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}
