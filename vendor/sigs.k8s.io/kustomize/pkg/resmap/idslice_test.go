/*
Copyright 2018 The Kubernetes Authors.

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

package resmap

import (
	"reflect"
	"sort"
	"testing"

	"sigs.k8s.io/kustomize/pkg/resid"
)

func TestLess(t *testing.T) {
	ids := IdSlice{
		resid.NewResIdKindOnly("ConfigMap", "cm"),
		resid.NewResIdKindOnly("Pod", "pod"),
		resid.NewResIdKindOnly("Namespace", "ns1"),
		resid.NewResIdKindOnly("Namespace", "ns2"),
		resid.NewResIdKindOnly("Role", "ro"),
		resid.NewResIdKindOnly("RoleBinding", "rb"),
		resid.NewResIdKindOnly("CustomResourceDefinition", "crd"),
		resid.NewResIdKindOnly("ServiceAccount", "sa"),
	}
	expected := IdSlice{
		resid.NewResIdKindOnly("Namespace", "ns1"),
		resid.NewResIdKindOnly("Namespace", "ns2"),
		resid.NewResIdKindOnly("CustomResourceDefinition", "crd"),
		resid.NewResIdKindOnly("ServiceAccount", "sa"),
		resid.NewResIdKindOnly("Role", "ro"),
		resid.NewResIdKindOnly("RoleBinding", "rb"),
		resid.NewResIdKindOnly("ConfigMap", "cm"),
		resid.NewResIdKindOnly("Pod", "pod"),
	}
	sort.Sort(ids)
	if !reflect.DeepEqual(ids, expected) {
		t.Fatalf("expected %+v but got %+v", expected, ids)
	}
}
