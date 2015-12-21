/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package json_test

import (
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/json"
)

func TestDecodeJSON(t *testing.T) {
	s := json.NewSerializer(json.DefaultMetaFactory, nil, nil, false)
	obj, gvk, err := s.Decode([]byte("{}"), nil, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if (*gvk != unversioned.GroupVersionKind{}) {
		t.Fatalf("unexpected gvk: %#v", gvk)
	}
	if obj != nil {
		t.Fatalf("unexpected object: %#v", obj)
	}
}

type mockCreater struct {
	apiVersion string
	kind       string
	err        error
}

func (c *mockCreater) New(kind unversioned.GroupVersionKind) (runtime.Object, error) {
	c.apiVersion, c.kind = kind.GroupVersion().String(), kind.Kind
	return nil, c.err
}

func TestDecodeJSONDefaultKind(t *testing.T) {
	creater := &mockCreater{err: fmt.Errorf("fake error")}
	s := json.NewSerializer(json.DefaultMetaFactory, creater, nil, false)
	defaultGVK := unversioned.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}
	obj, gvk, err := s.Decode([]byte("{}"), &defaultGVK, nil)
	if err != creater.err {
		t.Fatal("expected error")
	}
	if creater.apiVersion != "other/blah" || creater.kind != "Test" {
		t.Fatalf("creater should be called with defaults: %#v", creater)
	}
	if gvk == nil || (*gvk != defaultGVK) {
		t.Fatalf("unexpected gvk: %#v", gvk)
	}
	if obj != nil {
		t.Fatalf("unexpected object: %#v", obj)
	}
}
