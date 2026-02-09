/*
Copyright 2021 The Kubernetes Authors.

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

package storageversion

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/apiserverinternal"
)

func TestStorageVersionStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Error("StorageVersion strategy must be cluster scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("StorageVersion should not allow create on update")
	}

	storageVersion := validStorageVersion()
	Strategy.PrepareForCreate(ctx, storageVersion)
	errs := Strategy.Validate(ctx, storageVersion)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	storageVersionWithoutName := &apiserverinternal.StorageVersion{
		ObjectMeta: metav1.ObjectMeta{Name: ""},
	}
	Strategy.PrepareForUpdate(ctx, storageVersionWithoutName, storageVersion)
	errs = Strategy.ValidateUpdate(ctx, storageVersionWithoutName, storageVersion)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
}

func validStorageVersion() *apiserverinternal.StorageVersion {
	ssv1 := apiserverinternal.ServerStorageVersion{
		APIServerID:       "1",
		EncodingVersion:   "v1",
		DecodableVersions: []string{"v1", "v2"},
	}
	ssv2 := apiserverinternal.ServerStorageVersion{
		APIServerID:       "2",
		EncodingVersion:   "v1",
		DecodableVersions: []string{"v1", "v2"},
	}
	// ssv3 has a different encoding version
	ssv3 := apiserverinternal.ServerStorageVersion{
		APIServerID:       "3",
		EncodingVersion:   "v2",
		DecodableVersions: []string{"v1", "v2"},
	}
	return &apiserverinternal.StorageVersion{
		ObjectMeta: metav1.ObjectMeta{
			Name: "core.pods",
		},
		Status: apiserverinternal.StorageVersionStatus{
			StorageVersions: []apiserverinternal.ServerStorageVersion{ssv1, ssv2, ssv3},
		},
	}
}
