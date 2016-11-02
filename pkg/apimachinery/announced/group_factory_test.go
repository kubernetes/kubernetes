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
)

func TestKindResourceConstructor(t *testing.T) {
	table := []struct {
		krc             KindResourceConstructor
		expectedGroup   string
		expectedVersion string
	}{
		{
			krc: &GroupMetaFactoryArgs{
				GroupName: "foo",
			},
			expectedGroup:   "foo",
			expectedVersion: "__internal",
		},
		{
			krc: &GroupVersionFactoryArgs{
				GroupName:   "foo",
				VersionName: "v1",
			},
			expectedGroup:   "foo",
			expectedVersion: "v1",
		},
	}
	for i, tt := range table {
		sgv := tt.krc.SchemeGroupVersion()
		if e, a := tt.expectedGroup, sgv.Group; e != a {
			t.Errorf("%v: expected %v, got %v", i, e, a)
		}
		if e, a := tt.expectedVersion, sgv.Version; e != a {
			t.Errorf("%v: expected %v, got %v", i, e, a)
		}
		gk := tt.krc.Kind("kind")
		if e, a := tt.expectedGroup, gk.Group; e != a {
			t.Errorf("%v: expected %v, got %v", i, e, a)
		}
		gr := tt.krc.Resource("resource")
		if e, a := tt.expectedGroup, gr.Group; e != a {
			t.Errorf("%v: expected %v, got %v", i, e, a)
		}
	}
}
