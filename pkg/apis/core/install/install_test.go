/*
Copyright 2014 The Kubernetes Authors.

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
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	internal "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

func TestResourceVersioner(t *testing.T) {
	g, err := legacyscheme.Registry.Group(v1.GroupName)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	intf, err := g.DefaultInterfacesFor(v1.SchemeGroupVersion)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	accessor := intf.MetadataAccessor

	pod := internal.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "10"}}
	version, err := accessor.ResourceVersion(&pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}

	podList := internal.PodList{ListMeta: metav1.ListMeta{ResourceVersion: "10"}}
	version, err = accessor.ResourceVersion(&podList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}
}

func TestCodec(t *testing.T) {
	pod := internal.Pod{}
	// We do want to use package registered rather than testapi here, because we
	// want to test if the package install and package registered work as expected.
	data, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(legacyscheme.Registry.GroupOrDie(internal.GroupName).GroupVersion), &pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	other := internal.Pod{}
	if err := json.Unmarshal(data, &other); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if other.APIVersion != legacyscheme.Registry.GroupOrDie(internal.GroupName).GroupVersion.Version || other.Kind != "Pod" {
		t.Errorf("unexpected unmarshalled object %#v", other)
	}
}

func TestInterfacesFor(t *testing.T) {
	if _, err := legacyscheme.Registry.GroupOrDie(internal.GroupName).InterfacesFor(internal.SchemeGroupVersion); err == nil {
		t.Fatalf("unexpected non-error: %v", err)
	}
	for i, version := range legacyscheme.Registry.GroupOrDie(internal.GroupName).GroupVersions {
		if vi, err := legacyscheme.Registry.GroupOrDie(internal.GroupName).InterfacesFor(version); err != nil || vi == nil {
			t.Fatalf("%d: unexpected result: %v", i, err)
		}
	}
}

func TestRESTMapper(t *testing.T) {
	gv := schema.GroupVersion{Group: "", Version: "v1"}
	rcGVK := gv.WithKind("ReplicationController")
	podTemplateGVK := gv.WithKind("PodTemplate")

	if gvk, err := legacyscheme.Registry.RESTMapper().KindFor(internal.SchemeGroupVersion.WithResource("replicationcontrollers")); err != nil || gvk != rcGVK {
		t.Errorf("unexpected version mapping: %v %v", gvk, err)
	}

	if m, err := legacyscheme.Registry.GroupOrDie(internal.GroupName).RESTMapper.RESTMapping(podTemplateGVK.GroupKind(), ""); err != nil || m.GroupVersionKind != podTemplateGVK || m.Resource != "podtemplates" {
		t.Errorf("unexpected version mapping: %#v %v", m, err)
	}

	for _, version := range legacyscheme.Registry.GroupOrDie(internal.GroupName).GroupVersions {
		mapping, err := legacyscheme.Registry.GroupOrDie(internal.GroupName).RESTMapper.RESTMapping(rcGVK.GroupKind(), version.Version)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if mapping.Resource != "replicationControllers" && mapping.Resource != "replicationcontrollers" {
			t.Errorf("incorrect resource name: %#v", mapping)
		}
		if mapping.GroupVersionKind.GroupVersion() != version {
			t.Errorf("incorrect version: %v", mapping)
		}

		interfaces, _ := legacyscheme.Registry.GroupOrDie(internal.GroupName).InterfacesFor(version)
		if mapping.ObjectConvertor != interfaces.ObjectConvertor {
			t.Errorf("unexpected: %#v, expected: %#v", mapping, interfaces)
		}

		rc := &internal.ReplicationController{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
		name, err := mapping.MetadataAccessor.Name(rc)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if name != "foo" {
			t.Errorf("unable to retrieve object meta with: %v", mapping.MetadataAccessor)
		}
	}
}

func TestUnversioned(t *testing.T) {
	for _, obj := range []runtime.Object{
		&metav1.Status{},
	} {
		if unversioned, ok := legacyscheme.Scheme.IsUnversioned(obj); !unversioned || !ok {
			t.Errorf("%v is expected to be unversioned", reflect.TypeOf(obj))
		}
	}
}
