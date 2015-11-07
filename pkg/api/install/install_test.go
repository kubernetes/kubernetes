/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	internal "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

func TestResourceVersioner(t *testing.T) {
	pod := internal.Pod{ObjectMeta: internal.ObjectMeta{ResourceVersion: "10"}}
	version, err := accessor.ResourceVersion(&pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}

	podList := internal.PodList{ListMeta: unversioned.ListMeta{ResourceVersion: "10"}}
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
	// We do want to use package latest rather than testapi here, because we
	// want to test if the package install and package latest work as expected.
	data, err := latest.GroupOrDie("").Codec.Encode(&pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	other := internal.Pod{}
	if err := json.Unmarshal(data, &other); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if other.APIVersion != latest.GroupOrDie("").Version || other.Kind != "Pod" {
		t.Errorf("unexpected unmarshalled object %#v", other)
	}
}

func TestInterfacesFor(t *testing.T) {
	if _, err := latest.GroupOrDie("").InterfacesFor(""); err == nil {
		t.Fatalf("unexpected non-error: %v", err)
	}
	for i, version := range append([]string{latest.GroupOrDie("").Version}, latest.GroupOrDie("").Versions...) {
		if vi, err := latest.GroupOrDie("").InterfacesFor(version); err != nil || vi == nil {
			t.Fatalf("%d: unexpected result: %v", i, err)
		}
	}
}

func TestRESTMapper(t *testing.T) {
	if v, k, err := latest.GroupOrDie("").RESTMapper.VersionAndKindForResource("replicationcontrollers"); err != nil || v != "v1" || k != "ReplicationController" {
		t.Errorf("unexpected version mapping: %s %s %v", v, k, err)
	}

	if m, err := latest.GroupOrDie("").RESTMapper.RESTMapping("PodTemplate", ""); err != nil || m.APIVersion != "v1" || m.Resource != "podtemplates" {
		t.Errorf("unexpected version mapping: %#v %v", m, err)
	}

	for _, version := range latest.GroupOrDie("").Versions {
		mapping, err := latest.GroupOrDie("").RESTMapper.RESTMapping("ReplicationController", version)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if mapping.Resource != "replicationControllers" && mapping.Resource != "replicationcontrollers" {
			t.Errorf("incorrect resource name: %#v", mapping)
		}
		if mapping.APIVersion != version {
			t.Errorf("incorrect version: %v", mapping)
		}

		interfaces, _ := latest.GroupOrDie("").InterfacesFor(version)
		if mapping.Codec != interfaces.Codec {
			t.Errorf("unexpected codec: %#v, expected: %#v", mapping, interfaces)
		}

		rc := &internal.ReplicationController{ObjectMeta: internal.ObjectMeta{Name: "foo"}}
		name, err := mapping.MetadataAccessor.Name(rc)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if name != "foo" {
			t.Errorf("unable to retrieve object meta with: %v", mapping.MetadataAccessor)
		}
	}
}
