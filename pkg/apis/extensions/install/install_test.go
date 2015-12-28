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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

func TestResourceVersioner(t *testing.T) {
	daemonSet := extensions.DaemonSet{ObjectMeta: api.ObjectMeta{ResourceVersion: "10"}}
	version, err := accessor.ResourceVersion(&daemonSet)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}

	daemonSetList := extensions.DaemonSetList{ListMeta: unversioned.ListMeta{ResourceVersion: "10"}}
	version, err = accessor.ResourceVersion(&daemonSetList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}
}

func TestCodec(t *testing.T) {
	daemonSet := extensions.DaemonSet{}
	// We do want to use package latest rather than testapi here, because we
	// want to test if the package install and package latest work as expected.
	data, err := latest.GroupOrDie(extensions.GroupName).Codec.Encode(&daemonSet)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	other := extensions.DaemonSet{}
	if err := json.Unmarshal(data, &other); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if other.APIVersion != latest.GroupOrDie(extensions.GroupName).GroupVersion.String() || other.Kind != "DaemonSet" {
		t.Errorf("unexpected unmarshalled object %#v", other)
	}
}

func TestInterfacesFor(t *testing.T) {
	if _, err := latest.GroupOrDie(extensions.GroupName).InterfacesFor(extensions.SchemeGroupVersion); err == nil {
		t.Fatalf("unexpected non-error: %v", err)
	}
	for i, version := range latest.GroupOrDie(extensions.GroupName).GroupVersions {
		if vi, err := latest.GroupOrDie(extensions.GroupName).InterfacesFor(version); err != nil || vi == nil {
			t.Fatalf("%d: unexpected result: %v", i, err)
		}
	}
}

func TestRESTMapper(t *testing.T) {
	gv := v1beta1.SchemeGroupVersion
	hpaGVK := gv.WithKind("HorizontalPodAutoscaler")
	daemonSetGVK := gv.WithKind("DaemonSet")

	if gvk, err := latest.GroupOrDie(extensions.GroupName).RESTMapper.KindFor(gv.WithResource("horizontalpodautoscalers")); err != nil || gvk != hpaGVK {
		t.Errorf("unexpected version mapping: %v %v", gvk, err)
	}

	if m, err := latest.GroupOrDie(extensions.GroupName).RESTMapper.RESTMapping(daemonSetGVK.GroupKind(), ""); err != nil || m.GroupVersionKind != daemonSetGVK || m.Resource != "daemonsets" {
		t.Errorf("unexpected version mapping: %#v %v", m, err)
	}

	for _, version := range latest.GroupOrDie(extensions.GroupName).GroupVersions {
		mapping, err := latest.GroupOrDie(extensions.GroupName).RESTMapper.RESTMapping(hpaGVK.GroupKind(), version.Version)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if mapping.Resource != "horizontalpodautoscalers" {
			t.Errorf("incorrect resource name: %#v", mapping)
		}
		if mapping.GroupVersionKind.GroupVersion() != version {
			t.Errorf("incorrect groupVersion: %v", mapping)
		}

		interfaces, _ := latest.GroupOrDie(extensions.GroupName).InterfacesFor(version)
		if mapping.Codec != interfaces.Codec {
			t.Errorf("unexpected codec: %#v, expected: %#v", mapping, interfaces)
		}

		rc := &extensions.HorizontalPodAutoscaler{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		name, err := mapping.MetadataAccessor.Name(rc)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if name != "foo" {
			t.Errorf("unable to retrieve object meta with: %v", mapping.MetadataAccessor)
		}
	}
}
