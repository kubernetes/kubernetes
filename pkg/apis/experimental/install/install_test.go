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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/experimental"
)

func TestResourceVersioner(t *testing.T) {
	daemonSet := experimental.DaemonSet{ObjectMeta: api.ObjectMeta{ResourceVersion: "10"}}
	version, err := accessor.ResourceVersion(&daemonSet)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}

	daemonSetList := experimental.DaemonSetList{ListMeta: unversioned.ListMeta{ResourceVersion: "10"}}
	version, err = accessor.ResourceVersion(&daemonSetList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}
}

func TestCodec(t *testing.T) {
	daemonSet := experimental.DaemonSet{}
	// We do want to use package latest rather than testapi here, because we
	// want to test if the package install and package latest work as expected.
	data, err := latest.GroupOrDie("experimental").Codec.Encode(&daemonSet)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	other := experimental.DaemonSet{}
	if err := json.Unmarshal(data, &other); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if other.APIVersion != latest.GroupOrDie("experimental").GroupVersion || other.Kind != "DaemonSet" {
		t.Errorf("unexpected unmarshalled object %#v", other)
	}
}

func TestInterfacesFor(t *testing.T) {
	if _, err := latest.GroupOrDie("experimental").InterfacesFor(""); err == nil {
		t.Fatalf("unexpected non-error: %v", err)
	}
	for i, groupVersion := range append([]string{latest.GroupOrDie("experimental").GroupVersion}, latest.GroupOrDie("experimental").GroupVersions...) {
		if vi, err := latest.GroupOrDie("experimental").InterfacesFor(groupVersion); err != nil || vi == nil {
			t.Fatalf("%d: unexpected result: %v", i, err)
		}
	}
}

func TestRESTMapper(t *testing.T) {
	if v, k, err := latest.GroupOrDie("experimental").RESTMapper.VersionAndKindForResource("horizontalpodautoscalers"); err != nil || v != "experimental/v1alpha1" || k != "HorizontalPodAutoscaler" {
		t.Errorf("unexpected version mapping: %s %s %v", v, k, err)
	}

	if m, err := latest.GroupOrDie("experimental").RESTMapper.RESTMapping("DaemonSet", ""); err != nil || m.APIVersion != "experimental/v1alpha1" || m.Resource != "daemonsets" {
		t.Errorf("unexpected version mapping: %#v %v", m, err)
	}

	for _, groupVersion := range latest.GroupOrDie("experimental").GroupVersions {
		mapping, err := latest.GroupOrDie("experimental").RESTMapper.RESTMapping("HorizontalPodAutoscaler", groupVersion)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if mapping.Resource != "horizontalpodautoscalers" {
			t.Errorf("incorrect resource name: %#v", mapping)
		}
		if mapping.APIVersion != groupVersion {
			t.Errorf("incorrect groupVersion: %v", mapping)
		}

		interfaces, _ := latest.GroupOrDie("experimental").InterfacesFor(groupVersion)
		if mapping.Codec != interfaces.Codec {
			t.Errorf("unexpected codec: %#v, expected: %#v", mapping, interfaces)
		}

		rc := &experimental.HorizontalPodAutoscaler{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		name, err := mapping.MetadataAccessor.Name(rc)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if name != "foo" {
			t.Errorf("unable to retrieve object meta with: %v", mapping.MetadataAccessor)
		}
	}
}
