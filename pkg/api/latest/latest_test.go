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
	"strconv"
	"testing"

	internal "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	docker "github.com/fsouza/go-dockerclient"
	fuzz "github.com/google/gofuzz"
)

// apiObjectFuzzer can randomly populate api objects.
var apiObjectFuzzer = fuzz.New().NilChance(.5).NumElements(1, 1).Funcs(
	func(j *internal.TypeMeta, c fuzz.Continue) {
		// We have to customize the randomization of TypeMetas because their
		// APIVersion and Kind must remain blank in memory.
		j.APIVersion = ""
		j.Kind = ""
	},
	func(j *internal.ObjectMeta, c fuzz.Continue) {
		j.Name = c.RandString()
		j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
		j.SelfLink = c.RandString()

		var sec, nsec int64
		c.Fuzz(&sec)
		c.Fuzz(&nsec)
		j.CreationTimestamp = util.Unix(sec, nsec).Rfc3339Copy()
	},
	func(j *internal.ListMeta, c fuzz.Continue) {
		j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
		j.SelfLink = c.RandString()
	},
	func(j *internal.ObjectReference, c fuzz.Continue) {
		// We have to customize the randomization of TypeMetas because their
		// APIVersion and Kind must remain blank in memory.
		j.APIVersion = c.RandString()
		j.Kind = c.RandString()
		j.Namespace = c.RandString()
		j.Name = c.RandString()
		j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
		j.FieldPath = c.RandString()
	},
	func(j *internal.PodPhase, c fuzz.Continue) {
		statuses := []internal.PodPhase{internal.PodPending, internal.PodRunning, internal.PodFailed, internal.PodUnknown}
		*j = statuses[c.Rand.Intn(len(statuses))]
	},
	func(j *internal.ReplicationControllerSpec, c fuzz.Continue) {
		// TemplateRef must be nil for round trip
		c.Fuzz(&j.Template)
		if j.Template == nil {
			// TODO: v1beta1/2 can't round trip a nil template correctly, fix by having v1beta1/2
			// conversion compare converted object to nil via DeepEqual
			j.Template = &internal.PodTemplateSpec{}
		}
		j.Template.ObjectMeta = internal.ObjectMeta{Labels: j.Template.ObjectMeta.Labels}
		j.Template.Spec.NodeSelector = nil
		c.Fuzz(&j.Selector)
		j.Replicas = int(c.RandUint64())
	},
	func(j *internal.ReplicationControllerStatus, c fuzz.Continue) {
		// only replicas round trips
		j.Replicas = int(c.RandUint64())
	},
	func(j *internal.List, c fuzz.Continue) {
		c.Fuzz(&j.ListMeta)
		c.Fuzz(&j.Items)
		if j.Items == nil {
			j.Items = []runtime.Object{}
		}
	},
	func(j *runtime.Object, c fuzz.Continue) {
		if c.RandBool() {
			*j = &runtime.Unknown{
				TypeMeta: runtime.TypeMeta{Kind: "Something", APIVersion: "unknown"},
				RawJSON:  []byte(`{"apiVersion":"unknown","kind":"Something","someKey":"someValue"}`),
			}
		} else {
			types := []runtime.Object{&internal.Pod{}, &internal.ReplicationController{}}
			t := types[c.Rand.Intn(len(types))]
			c.Fuzz(t)
			*j = t
		}
	},
	func(intstr *util.IntOrString, c fuzz.Continue) {
		// util.IntOrString will panic if its kind is set wrong.
		if c.RandBool() {
			intstr.Kind = util.IntstrInt
			intstr.IntVal = int(c.RandUint64())
			intstr.StrVal = ""
		} else {
			intstr.Kind = util.IntstrString
			intstr.IntVal = 0
			intstr.StrVal = c.RandString()
		}
	},
	func(pb map[docker.Port][]docker.PortBinding, c fuzz.Continue) {
		// This is necessary because keys with nil values get omitted.
		// TODO: Is this a bug?
		pb[docker.Port(c.RandString())] = []docker.PortBinding{
			{c.RandString(), c.RandString()},
			{c.RandString(), c.RandString()},
		}
	},
	func(pm map[string]docker.PortMapping, c fuzz.Continue) {
		// This is necessary because keys with nil values get omitted.
		// TODO: Is this a bug?
		pm[c.RandString()] = docker.PortMapping{
			c.RandString(): c.RandString(),
		}
	},
	func(p *internal.PullPolicy, c fuzz.Continue) {
		policies := []internal.PullPolicy{internal.PullAlways, internal.PullNever, internal.PullIfNotPresent}
		*p = policies[c.Rand.Intn(len(policies))]
	},
)

func TestInternalRoundTrip(t *testing.T) {
	latest := "v1beta2"

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
