/*
Copyright 2016 The Kubernetes Authors.

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

package announced

import (
	"testing"

	"k8s.io/kubernetes/pkg/util/sets"
)

func TestFactoryRegistry(t *testing.T) {
	reg := make(APIGroupFactoryRegistry)

	if err := reg.AnnounceGroupVersion(&GroupVersionFactoryArgs{
		GroupName:   "foo",
		VersionName: "v1",
	}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if err := reg.AnnounceGroup(&GroupMetaFactoryArgs{
		GroupName:              "foo",
		VersionPreferenceOrder: []string{"v2", "v1"},
		ImportPrefix:           "pkg/apis/foo",
		RootScopedKinds:        sets.NewString("namespaces"),
	}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if err := reg.AnnounceGroupVersion(&GroupVersionFactoryArgs{
		GroupName:   "foo",
		VersionName: "v2",
	}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if a := len(reg["foo"].VersionArgs); a != 2 {
		t.Errorf("Expected 2 args but got %v", a)
	}
}
