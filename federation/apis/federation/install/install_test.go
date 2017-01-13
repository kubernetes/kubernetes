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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/pkg/api"
)

func TestResourceVersioner(t *testing.T) {
	cluster := federation.Cluster{ObjectMeta: api.ObjectMeta{ResourceVersion: "10"}}
	version, err := accessor.ResourceVersion(&cluster)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}

	clusterList := federation.ClusterList{ListMeta: metav1.ListMeta{ResourceVersion: "10"}}
	version, err = accessor.ResourceVersion(&clusterList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}
}

func TestCodec(t *testing.T) {
	cluster := federation.Cluster{}
	// We do want to use package registered rather than testapi here, because we
	// want to test if the package install and package registered work as expected.
	data, err := runtime.Encode(api.Codecs.LegacyCodec(api.Registry.GroupOrDie(federation.GroupName).GroupVersion), &cluster)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	other := federation.Cluster{}
	if err := json.Unmarshal(data, &other); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if other.APIVersion != api.Registry.GroupOrDie(federation.GroupName).GroupVersion.String() || other.Kind != "Cluster" {
		t.Errorf("unexpected unmarshalled object %#v", other)
	}
}

func TestInterfacesFor(t *testing.T) {
	if _, err := api.Registry.GroupOrDie(federation.GroupName).InterfacesFor(federation.SchemeGroupVersion); err == nil {
		t.Fatalf("unexpected non-error: %v", err)
	}
	for i, version := range api.Registry.GroupOrDie(federation.GroupName).GroupVersions {
		if vi, err := api.Registry.GroupOrDie(federation.GroupName).InterfacesFor(version); err != nil || vi == nil {
			t.Fatalf("%d: unexpected result: %v", i, err)
		}
	}
}

func TestRESTMapper(t *testing.T) {
	gv := v1beta1.SchemeGroupVersion
	clusterGVK := gv.WithKind("Cluster")

	if gvk, err := api.Registry.GroupOrDie(federation.GroupName).RESTMapper.KindFor(gv.WithResource("clusters")); err != nil || gvk != clusterGVK {
		t.Errorf("unexpected version mapping: %v %v", gvk, err)
	}

	if m, err := api.Registry.GroupOrDie(federation.GroupName).RESTMapper.RESTMapping(clusterGVK.GroupKind(), ""); err != nil || m.GroupVersionKind != clusterGVK || m.Resource != "clusters" {
		t.Errorf("unexpected version mapping: %#v %v", m, err)
	}

	for _, version := range api.Registry.GroupOrDie(federation.GroupName).GroupVersions {
		mapping, err := api.Registry.GroupOrDie(federation.GroupName).RESTMapper.RESTMapping(clusterGVK.GroupKind(), version.Version)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if mapping.Resource != "clusters" {
			t.Errorf("incorrect resource name: %#v", mapping)
		}
		if mapping.GroupVersionKind.GroupVersion() != version {
			t.Errorf("incorrect groupVersion: %v", mapping)
		}

		interfaces, _ := api.Registry.GroupOrDie(federation.GroupName).InterfacesFor(version)
		if mapping.ObjectConvertor != interfaces.ObjectConvertor {
			t.Errorf("unexpected: %#v, expected: %#v", mapping, interfaces)
		}

		rc := &federation.Cluster{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		name, err := mapping.MetadataAccessor.Name(rc)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if name != "foo" {
			t.Errorf("unable to retrieve object meta with: %v", mapping.MetadataAccessor)
		}
	}
}
