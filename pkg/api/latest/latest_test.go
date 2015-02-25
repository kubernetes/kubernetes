/*
Copyright 2014 Google Inc. All rights reserved.

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

package latest

import (
	"encoding/json"
	"math/rand"
	"testing"

	internal "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apitesting "github.com/GoogleCloudPlatform/kubernetes/pkg/api/testing"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestInternalRoundTrip(t *testing.T) {
	latest := "v1beta2"

	seed := rand.Int63()
	apiObjectFuzzer := apitesting.FuzzerFor(t, "", rand.NewSource(seed))
	for k := range internal.Scheme.KnownTypes("") {
		obj, err := internal.Scheme.New("", k)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", k, err)
			continue
		}
		apiObjectFuzzer.Fuzz(obj)

		newer, err := internal.Scheme.New(latest, k)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", k, err)
			continue
		}

		if err := internal.Scheme.Convert(obj, newer); err != nil {
			t.Errorf("unable to convert %#v to %#v: %v", obj, newer, err)
			continue
		}

		actual, err := internal.Scheme.New("", k)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", k, err)
			continue
		}

		if err := internal.Scheme.Convert(newer, actual); err != nil {
			t.Errorf("unable to convert %#v to %#v: %v", newer, actual, err)
			continue
		}

		if !internal.Semantic.DeepEqual(obj, actual) {
			t.Errorf("%s: diff %s", k, util.ObjectDiff(obj, actual))
		}
	}
}

func TestResourceVersioner(t *testing.T) {
	pod := internal.Pod{ObjectMeta: internal.ObjectMeta{ResourceVersion: "10"}}
	version, err := ResourceVersioner.ResourceVersion(&pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}

	podList := internal.PodList{ListMeta: internal.ListMeta{ResourceVersion: "10"}}
	version, err = ResourceVersioner.ResourceVersion(&podList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}
}

func TestCodec(t *testing.T) {
	pod := internal.Pod{}
	data, err := Codec.Encode(&pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	other := internal.Pod{}
	if err := json.Unmarshal(data, &other); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if other.APIVersion != Version || other.Kind != "Pod" {
		t.Errorf("unexpected unmarshalled object %#v", other)
	}
}

func TestInterfacesFor(t *testing.T) {
	if _, err := InterfacesFor(""); err == nil {
		t.Fatalf("unexpected non-error: %v", err)
	}
	for i, version := range append([]string{Version, OldestVersion}, Versions...) {
		if vi, err := InterfacesFor(version); err != nil || vi == nil {
			t.Fatalf("%d: unexpected result: %v", i, err)
		}
	}
}

func TestRESTMapper(t *testing.T) {
	if v, k, err := RESTMapper.VersionAndKindForResource("replicationControllers"); err != nil || v != Version || k != "ReplicationController" {
		t.Errorf("unexpected version mapping: %s %s %v", v, k, err)
	}
	if v, k, err := RESTMapper.VersionAndKindForResource("replicationcontrollers"); err != nil || v != Version || k != "ReplicationController" {
		t.Errorf("unexpected version mapping: %s %s %v", v, k, err)
	}

	for _, version := range Versions {
		mapping, err := RESTMapper.RESTMapping("ReplicationController", version)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if mapping.Resource != "replicationControllers" && mapping.Resource != "replicationcontrollers" {
			t.Errorf("incorrect resource name: %#v", mapping)
		}
		if mapping.APIVersion != version {
			t.Errorf("incorrect version: %v", mapping)
		}

		interfaces, _ := InterfacesFor(version)
		if mapping.Codec != interfaces.Codec {
			t.Errorf("unexpected codec: %#v", mapping)
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
