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
	data, err := latest.GroupOrDie(componentconfig.GroupName).Codec.Encode(&daemonSet)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	other := componentconfig.KubeProxyConfiguration{}
	if err := json.Unmarshal(data, &other); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if other.APIVersion != latest.GroupOrDie(componentconfig.GroupName).GroupVersion.String() || other.Kind != "KubeProxyConfiguration" {
		t.Errorf("unexpected unmarshalled object %#v", other)
	}
}

func TestInterfacesFor(t *testing.T) {
	if _, err := latest.GroupOrDie(componentconfig.GroupName).InterfacesFor(componentconfig.SchemeGroupVersion); err == nil {
		t.Fatalf("unexpected non-error: %v", err)
	}
	for i, version := range latest.GroupOrDie(componentconfig.GroupName).GroupVersions {
		if vi, err := latest.GroupOrDie(componentconfig.GroupName).InterfacesFor(version); err != nil || vi == nil {
			t.Fatalf("%d: unexpected result: %v", i, err)
		}
	}
}

func TestRESTMapper(t *testing.T) {
	gv := unversioned.GroupVersion{Group: componentconfig.GroupName, Version: "v1alpha1"}
	proxyGVK := gv.WithKind("KubeProxyConfiguration")

	if gvk, err := latest.GroupOrDie(componentconfig.GroupName).RESTMapper.KindFor(gv.WithResource("kubeproxyconfiguration")); err != nil || gvk != proxyGVK {
		t.Errorf("unexpected version mapping: %v %v", gvk, err)
	}

	if m, err := latest.GroupOrDie(componentconfig.GroupName).RESTMapper.RESTMapping(proxyGVK.GroupKind(), ""); err != nil || m.GroupVersionKind != proxyGVK || m.Resource != "kubeproxyconfigurations" {
		t.Errorf("unexpected version mapping: %#v %v", m, err)
	}

	for _, version := range latest.GroupOrDie(componentconfig.GroupName).GroupVersions {
		mapping, err := latest.GroupOrDie(componentconfig.GroupName).RESTMapper.RESTMapping(proxyGVK.GroupKind(), version.Version)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}

		if mapping.Resource != "kubeproxyconfigurations" {
			t.Errorf("incorrect resource name: %#v", mapping)
		}
		if mapping.GroupVersionKind.GroupVersion() != version {
			t.Errorf("incorrect groupVersion: %v", mapping)
		}

		interfaces, _ := latest.GroupOrDie(componentconfig.GroupName).InterfacesFor(version)
		if mapping.Codec != interfaces.Codec {
			t.Errorf("unexpected codec: %#v, expected: %#v", mapping, interfaces)
		}
	}
}
