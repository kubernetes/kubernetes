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
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
)

func TestFactoryRegistry(t *testing.T) {
	regA := make(APIGroupFactoryRegistry)
	regB := make(APIGroupFactoryRegistry)

	if err := regA.AnnounceGroup(&GroupMetaFactoryArgs{
		GroupName:              "foo",
		VersionPreferenceOrder: []string{"v2", "v1"},
		ImportPrefix:           "pkg/apis/foo",
		RootScopedKinds:        sets.NewString("namespaces"),
	}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if err := regA.AnnounceGroupVersion(&GroupVersionFactoryArgs{
		GroupName:   "foo",
		VersionName: "v1",
	}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if err := regA.AnnounceGroupVersion(&GroupVersionFactoryArgs{
		GroupName:   "foo",
		VersionName: "v2",
	}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if err := regB.AnnouncePreconstructedFactory(NewGroupMetaFactory(
		&GroupMetaFactoryArgs{
			GroupName:              "foo",
			VersionPreferenceOrder: []string{"v2", "v1"},
			ImportPrefix:           "pkg/apis/foo",
			RootScopedKinds:        sets.NewString("namespaces"),
		},
		VersionToSchemeFunc{"v1": nil, "v2": nil},
	)); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if !reflect.DeepEqual(regA, regB) {
		t.Errorf("Expected both ways of registering to be equivalent, but they were not.\n\n%#v\n\n%#v\n", regA, regB)
	}
}
