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

package apimachinery

import (
	"testing"

	"k8s.io/kubernetes/pkg/runtime/schema"
)

func TestAdd(t *testing.T) {
	gm := GroupMeta{
		GroupVersion: schema.GroupVersion{
			Group:   "test",
			Version: "v1",
		},
		GroupVersions: []schema.GroupVersion{{Group: "test", Version: "v1"}},
	}

	gm.AddVersionInterfaces(schema.GroupVersion{Group: "test", Version: "v1"}, nil)
	if e, a := 1, len(gm.InterfacesByVersion); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	// GroupVersions is unchanged
	if e, a := 1, len(gm.GroupVersions); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}
