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

package install

import (
	"encoding/json"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestResourceVersioner(t *testing.T) {
	roleBinding := rbac.RoleBinding{ObjectMeta: api.ObjectMeta{ResourceVersion: "10"}}
	version, err := accessor.ResourceVersion(&roleBinding)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}

	roleBindingList := rbac.RoleBindingList{ListMeta: unversioned.ListMeta{ResourceVersion: "10"}}
	version, err = accessor.ResourceVersion(&roleBindingList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}
}

func TestCodec(t *testing.T) {
	roleBinding := rbac.RoleBinding{}
	// We do want to use package registered rather than testapi here, because we
	// want to test if the package install and package registered work as expected.
	data, err := runtime.Encode(api.Codecs.LegacyCodec(registered.GroupOrDie(rbac.GroupName).GroupVersion), &roleBinding)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	other := rbac.RoleBinding{}
	if err := json.Unmarshal(data, &other); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if other.APIVersion != registered.GroupOrDie(rbac.GroupName).GroupVersion.String() || other.Kind != "RoleBinding" {
		t.Errorf("unexpected unmarshalled object %#v", other)
	}
}

func TestInterfacesFor(t *testing.T) {
	if _, err := registered.GroupOrDie(rbac.GroupName).InterfacesFor(rbac.SchemeGroupVersion); err == nil {
		t.Fatalf("unexpected non-error: %v", err)
	}
	for i, version := range registered.GroupOrDie(rbac.GroupName).GroupVersions {
		if vi, err := registered.GroupOrDie(rbac.GroupName).InterfacesFor(version); err != nil || vi == nil {
			t.Fatalf("%d: unexpected result: %v", i, err)
		}
	}
}

func TestRESTMapper(t *testing.T) {
	gv := v1alpha1.SchemeGroupVersion
	roleBindingGVK := gv.WithKind("RoleBinding")

	if gvk, err := registered.GroupOrDie(rbac.GroupName).RESTMapper.KindFor(gv.WithResource("rolebindings")); err != nil || gvk != roleBindingGVK {
		t.Errorf("unexpected version mapping: %v %v", gvk, err)
	}

	for _, version := range registered.GroupOrDie(rbac.GroupName).GroupVersions {
		mapping, err := registered.GroupOrDie(rbac.GroupName).RESTMapper.RESTMapping(roleBindingGVK.GroupKind(), version.Version)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if mapping.Resource != "rolebindings" {
			t.Errorf("incorrect resource name: %#v", mapping)
		}
		if mapping.GroupVersionKind.GroupVersion() != version {
			t.Errorf("incorrect groupVersion: %v", mapping)
		}

		interfaces, _ := registered.GroupOrDie(rbac.GroupName).InterfacesFor(version)
		if mapping.ObjectConvertor != interfaces.ObjectConvertor {
			t.Errorf("unexpected: %#v, expected: %#v", mapping, interfaces)
		}

		roleBinding := &rbac.RoleBinding{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		name, err := mapping.MetadataAccessor.Name(roleBinding)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if name != "foo" {
			t.Errorf("unable to retrieve object meta with: %v", mapping.MetadataAccessor)
		}
	}
}
