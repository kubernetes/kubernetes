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

	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

func TestCodec(t *testing.T) {
	daemonSet := componentconfig.KubeProxyConfiguration{}
	// We do want to use package latest rather than testapi here, because we
	// want to test if the package install and package latest work as expected.
	data, err := latest.GroupOrDie("componentconfig").Codec.Encode(&daemonSet)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	other := componentconfig.KubeProxyConfiguration{}
	if err := json.Unmarshal(data, &other); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if other.APIVersion != latest.GroupOrDie("componentconfig").GroupVersion || other.Kind != "KubeProxyConfiguration" {
		t.Errorf("unexpected unmarshalled object %#v", other)
	}
}

func TestInterfacesFor(t *testing.T) {
	if _, err := latest.GroupOrDie("componentconfig").InterfacesFor(""); err == nil {
		t.Fatalf("unexpected non-error: %v", err)
	}
	for i, groupVersion := range append([]string{latest.GroupOrDie("componentconfig").GroupVersion}, latest.GroupOrDie("componentconfig").GroupVersions...) {
		if vi, err := latest.GroupOrDie("componentconfig").InterfacesFor(groupVersion); err != nil || vi == nil {
			t.Fatalf("%d: unexpected result: %v", i, err)
		}
	}
}

func TestRESTMapper(t *testing.T) {
	expectedGroupVersion := unversioned.GroupVersion{Group: "componentconfig", Version: "v1alpha1"}

	if v, k, err := latest.GroupOrDie("componentconfig").RESTMapper.VersionAndKindForResource("kubeproxyconfiguration"); err != nil || v != expectedGroupVersion.String() || k != "KubeProxyConfiguration" {
		t.Errorf("unexpected version mapping: %q %q %v", v, k, err)
	}

	if m, err := latest.GroupOrDie("componentconfig").RESTMapper.RESTMapping("KubeProxyConfiguration", ""); err != nil || m.GroupVersionKind.GroupVersion() != expectedGroupVersion || m.Resource != "kubeproxyconfigurations" {
		t.Errorf("unexpected version mapping: %#v %v", m, err)
	}

	for _, groupVersionString := range latest.GroupOrDie("componentconfig").GroupVersions {
		gv, err := unversioned.ParseGroupVersion(groupVersionString)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		mapping, err := latest.GroupOrDie("componentconfig").RESTMapper.RESTMapping("KubeProxyConfiguration", gv.String())
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}

		if mapping.Resource != "kubeproxyconfigurations" {
			t.Errorf("incorrect resource name: %#v", mapping)
		}
		if mapping.GroupVersionKind.GroupVersion() != gv {
			t.Errorf("incorrect groupVersion: %v", mapping)
		}

		interfaces, _ := latest.GroupOrDie("componentconfig").InterfacesFor(gv.String())
		if mapping.Codec != interfaces.Codec {
			t.Errorf("unexpected codec: %#v, expected: %#v", mapping, interfaces)
		}
	}
}
