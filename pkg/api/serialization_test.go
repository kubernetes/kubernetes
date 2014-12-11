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

package api_test

import (
	"encoding/json"
	"flag"
	"math/rand"
	"reflect"
	"strconv"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	docker "github.com/fsouza/go-dockerclient"
	fuzz "github.com/google/gofuzz"
)

var fuzzIters = flag.Int("fuzz_iters", 40, "How many fuzzing iterations to do.")

// apiObjectFuzzer can randomly populate api objects.
var apiObjectFuzzer = fuzz.New().NilChance(.5).NumElements(1, 1).Funcs(
	func(j *runtime.PluginBase, c fuzz.Continue) {
		// Do nothing; this struct has only a Kind field and it must stay blank in memory.
	},
	func(j *runtime.TypeMeta, c fuzz.Continue) {
		// We have to customize the randomization of TypeMetas because their
		// APIVersion and Kind must remain blank in memory.
		j.APIVersion = ""
		j.Kind = ""
	},
	func(j *api.TypeMeta, c fuzz.Continue) {
		// We have to customize the randomization of TypeMetas because their
		// APIVersion and Kind must remain blank in memory.
		j.APIVersion = ""
		j.Kind = ""
	},
	func(j *api.ObjectMeta, c fuzz.Continue) {
		j.Name = c.RandString()
		j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
		j.SelfLink = c.RandString()

		var sec, nsec int64
		c.Fuzz(&sec)
		c.Fuzz(&nsec)
		j.CreationTimestamp = util.Unix(sec, nsec).Rfc3339Copy()
	},
	func(j *api.ListMeta, c fuzz.Continue) {
		j.ResourceVersion = strconv.FormatUint(c.RandUint64(), 10)
		j.SelfLink = c.RandString()
	},
	func(j *api.PodPhase, c fuzz.Continue) {
		statuses := []api.PodPhase{api.PodPending, api.PodRunning, api.PodFailed}
		*j = statuses[c.Rand.Intn(len(statuses))]
	},
	func(j *api.ReplicationControllerSpec, c fuzz.Continue) {
		// TemplateRef must be nil for round trip
		c.Fuzz(&j.Template)
		if j.Template == nil {
			// TODO: v1beta1/2 can't round trip a nil template correctly, fix by having v1beta1/2
			// conversion compare converted object to nil via DeepEqual
			j.Template = &api.PodTemplateSpec{}
		}
		j.Template.ObjectMeta = api.ObjectMeta{Labels: j.Template.ObjectMeta.Labels}
		j.Template.Spec.NodeSelector = nil
		c.Fuzz(&j.Selector)
		j.Replicas = int(c.RandUint64())
	},
	func(j *api.ReplicationControllerStatus, c fuzz.Continue) {
		// only replicas round trips
		j.Replicas = int(c.RandUint64())
	},
	func(j *api.List, c fuzz.Continue) {
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
			types := []runtime.Object{&api.Pod{}, &api.ReplicationController{}}
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
)

func runTest(t *testing.T, codec runtime.Codec, source runtime.Object) {
	name := reflect.TypeOf(source).Elem().Name()
	apiObjectFuzzer.Fuzz(source)
	j, err := meta.Accessor(source)
	if err != nil {
		t.Fatalf("Unexpected error %v for %#v", err, source)
	}
	j.SetKind("")
	j.SetAPIVersion("")

	data, err := codec.Encode(source)
	if err != nil {
		t.Errorf("%v: %v (%#v)", name, err, source)
		return
	}

	obj2, err := codec.Decode(data)
	if err != nil {
		t.Errorf("0: %v: %v\nCodec: %v\nData: %s\nSource: %#v", name, err, codec, string(data), source)
		return
	}
	if !reflect.DeepEqual(source, obj2) {
		t.Errorf("1: %v: diff: %v\nCodec: %v\nData: %s\nSource: %#v", name, util.ObjectGoPrintDiff(source, obj2), codec, string(data), source)
		return
	}

	obj3 := reflect.New(reflect.TypeOf(source).Elem()).Interface().(runtime.Object)
	err = codec.DecodeInto(data, obj3)
	if err != nil {
		t.Errorf("2: %v: %v", name, err)
		return
	}
	if !reflect.DeepEqual(source, obj3) {
		t.Errorf("3: %v: diff: %v\nCodec: %v", name, util.ObjectDiff(source, obj3), codec)
		return
	}
}

// For debugging problems
func TestSpecificKind(t *testing.T) {
	api.Scheme.Log(t)
	kind := "PodList"
	item, err := api.Scheme.New("", kind)
	if err != nil {
		t.Errorf("Couldn't make a %v? %v", kind, err)
		return
	}
	runTest(t, v1beta1.Codec, item)
	runTest(t, v1beta2.Codec, item)
	api.Scheme.Log(nil)
}

func TestList(t *testing.T) {
	api.Scheme.Log(t)
	kind := "List"
	item, err := api.Scheme.New("", kind)
	if err != nil {
		t.Errorf("Couldn't make a %v? %v", kind, err)
		return
	}
	runTest(t, v1beta1.Codec, item)
	runTest(t, v1beta2.Codec, item)
	api.Scheme.Log(nil)
}

var nonRoundTrippableTypes = util.NewStringSet("ContainerManifest")
var nonInternalRoundTrippableTypes = util.NewStringSet("List")

func TestRoundTripTypes(t *testing.T) {
	for kind := range api.Scheme.KnownTypes("") {
		if nonRoundTrippableTypes.Has(kind) {
			continue
		}
		// Try a few times, since runTest uses random values.
		for i := 0; i < *fuzzIters; i++ {
			item, err := api.Scheme.New("", kind)
			if err != nil {
				t.Fatalf("Couldn't make a %v? %v", kind, err)
			}
			if _, err := meta.Accessor(item); err != nil {
				t.Fatalf("%q is not a TypeMeta and cannot be tested - add it to nonRoundTrippableTypes: %v", kind, err)
			}
			runTest(t, v1beta1.Codec, item)
			runTest(t, v1beta2.Codec, item)
			if !nonInternalRoundTrippableTypes.Has(kind) {
				runTest(t, api.Codec, item)
			}
		}
	}
}

func TestEncode_Ptr(t *testing.T) {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels: map[string]string{"name": "foo"},
		},
	}
	obj := runtime.Object(pod)
	data, err := latest.Codec.Encode(obj)
	obj2, err2 := latest.Codec.Decode(data)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'", err, err2)
	}
	if _, ok := obj2.(*api.Pod); !ok {
		t.Fatalf("Got wrong type")
	}
	if !reflect.DeepEqual(obj2, pod) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", &pod, obj2)
	}
}

func TestBadJSONRejection(t *testing.T) {
	badJSONMissingKind := []byte(`{ }`)
	if _, err := latest.Codec.Decode(badJSONMissingKind); err == nil {
		t.Errorf("Did not reject despite lack of kind field: %s", badJSONMissingKind)
	}
	badJSONUnknownType := []byte(`{"kind": "bar"}`)
	if _, err1 := latest.Codec.Decode(badJSONUnknownType); err1 == nil {
		t.Errorf("Did not reject despite use of unknown type: %s", badJSONUnknownType)
	}
	/*badJSONKindMismatch := []byte(`{"kind": "Pod"}`)
	if err2 := DecodeInto(badJSONKindMismatch, &Minion{}); err2 == nil {
		t.Errorf("Kind is set but doesn't match the object type: %s", badJSONKindMismatch)
	}*/
}

const benchmarkSeed = 100

func BenchmarkEncode(b *testing.B) {
	pod := api.Pod{}
	apiObjectFuzzer.RandSource(rand.NewSource(benchmarkSeed))
	apiObjectFuzzer.Fuzz(&pod)
	for i := 0; i < b.N; i++ {
		latest.Codec.Encode(&pod)
	}
}

// BenchmarkEncodeJSON provides a baseline for regular JSON encode performance
func BenchmarkEncodeJSON(b *testing.B) {
	pod := api.Pod{}
	apiObjectFuzzer.RandSource(rand.NewSource(benchmarkSeed))
	apiObjectFuzzer.Fuzz(&pod)
	for i := 0; i < b.N; i++ {
		json.Marshal(&pod)
	}
}

func BenchmarkDecode(b *testing.B) {
	pod := api.Pod{}
	apiObjectFuzzer.RandSource(rand.NewSource(benchmarkSeed))
	apiObjectFuzzer.Fuzz(&pod)
	data, _ := latest.Codec.Encode(&pod)
	for i := 0; i < b.N; i++ {
		latest.Codec.Decode(data)
	}
}

func BenchmarkDecodeInto(b *testing.B) {
	pod := api.Pod{}
	apiObjectFuzzer.RandSource(rand.NewSource(benchmarkSeed))
	apiObjectFuzzer.Fuzz(&pod)
	data, _ := latest.Codec.Encode(&pod)
	for i := 0; i < b.N; i++ {
		obj := api.Pod{}
		latest.Codec.DecodeInto(data, &obj)
	}
}

// BenchmarkDecodeJSON provides a baseline for regular JSON decode performance
func BenchmarkDecodeJSON(b *testing.B) {
	pod := api.Pod{}
	apiObjectFuzzer.RandSource(rand.NewSource(benchmarkSeed))
	apiObjectFuzzer.Fuzz(&pod)
	data, _ := latest.Codec.Encode(&pod)
	for i := 0; i < b.N; i++ {
		obj := api.Pod{}
		json.Unmarshal(data, &obj)
	}
}
