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

package kubeadm

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	componentconfigtesting "k8s.io/component-base/config/testing"
)

func TestKind(t *testing.T) {
	tests := []struct {
		input string
		out   schema.GroupKind
	}{
		{input: "Pod", out: schema.GroupKind{Kind: "Pod"}},
		{input: ".StatefulSet", out: schema.GroupKind{Group: "StatefulSet"}},
		{input: "StatefulSet.apps", out: schema.GroupKind{Group: "apps", Kind: "StatefulSet"}},
	}
	for i, test := range tests {
		t.Run(test.input, func(t *testing.T) {
			out := Kind(test.input)
			if out != test.out {
				t.Errorf("%d: expected output: %#v, got: %#v", i, test.out, out)
			}
		})
	}
}

func TestGroupResourceParse(t *testing.T) {
	tests := []struct {
		input string
		out   schema.GroupResource
	}{
		{input: "v1", out: schema.GroupResource{Resource: "v1"}},
		{input: ".v1", out: schema.GroupResource{Group: "v1"}},
		{input: "v1.", out: schema.GroupResource{Resource: "v1"}},
		{input: "v1.a", out: schema.GroupResource{Group: "a", Resource: "v1"}},
		{input: "b.v1.a", out: schema.GroupResource{Group: "v1.a", Resource: "b"}},
	}
	for i, test := range tests {
		out := Resource(test.input)
		if out != test.out {
			t.Errorf("%d: unexpected output: %#v", i, out)
		}
	}
}

func TestComponentConfigSetup(t *testing.T) {
	pkginfo := &componentconfigtesting.ComponentConfigPackage{
		ComponentName:      "kubeadm",
		GroupName:          GroupName,
		SchemeGroupVersion: SchemeGroupVersion,
		AddToScheme:        AddToScheme,
	}

	if err := componentconfigtesting.VerifyExternalTypePackage(pkginfo); err != nil {
		t.Fatal(err)
	}
}
