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
	"fmt"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/fsouza/go-dockerclient"
	"github.com/google/gofuzz"
)

var fuzzIters = flag.Int("fuzz_iters", 50, "How many fuzzing iterations to do.")

// apiObjectFuzzer can randomly populate api objects.
var apiObjectFuzzer = fuzz.New().NilChance(.5).NumElements(1, 1).Funcs(
	func(j *api.JSONBase, c fuzz.Continue) {
		// We have to customize the randomization of JSONBases because their
		// APIVersion and Kind must remain blank in memory.
		j.APIVersion = ""
		j.Kind = ""
		j.ID = c.RandString()
		// TODO: Fix JSON/YAML packages and/or write custom encoding
		// for uint64's. Somehow the LS *byte* of this is lost, but
		// only when all 8 bytes are set.
		j.ResourceVersion = c.RandUint64() >> 8
		j.SelfLink = c.RandString()

		var sec, nsec int64
		c.Fuzz(&sec)
		c.Fuzz(&nsec)
		j.CreationTimestamp = util.Unix(sec, nsec).Rfc3339Copy()
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
	func(u64 *uint64, c fuzz.Continue) {
		// TODO: uint64's are NOT handled right.
		*u64 = c.RandUint64() >> 8
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

func objDiff(a, b interface{}) string {
	ab, err := json.Marshal(a)
	if err != nil {
		panic("a")
	}
	bb, err := json.Marshal(b)
	if err != nil {
		panic("b")
	}
	return util.StringDiff(string(ab), string(bb))

	// An alternate diff attempt, in case json isn't showing you
	// the difference. (reflect.DeepEqual makes a distinction between
	// nil and empty slices, for example.)
	return util.StringDiff(
		fmt.Sprintf("%#v", a),
		fmt.Sprintf("%#v", b),
	)
}

func runTest(t *testing.T, source interface{}) {
	name := reflect.TypeOf(source).Elem().Name()
	apiObjectFuzzer.Fuzz(source)
	j, err := api.FindJSONBase(source)
	if err != nil {
		t.Fatalf("Unexpected error %v for %#v", err, source)
	}
	j.SetKind("")
	j.SetAPIVersion("")

	data, err := api.Encode(source)
	if err != nil {
		t.Errorf("%v: %v (%#v)", name, err, source)
		return
	}

	obj2, err := api.Decode(data)
	if err != nil {
		t.Errorf("%v: %v", name, err)
		return
	} else {
		if !reflect.DeepEqual(source, obj2) {
			t.Errorf("1: %v: diff: %v", name, objDiff(source, obj2))
			return
		}
	}
	obj3 := reflect.New(reflect.TypeOf(source).Elem()).Interface()
	err = api.DecodeInto(data, obj3)
	if err != nil {
		t.Errorf("2: %v: %v", name, err)
		return
	} else {
		if !reflect.DeepEqual(source, obj3) {
			t.Errorf("3: %v: diff: %v", name, objDiff(source, obj3))
			return
		}
	}
}

func TestTypes(t *testing.T) {
	table := []interface{}{
		&api.PodList{},
		&api.Pod{},
		&api.ServiceList{},
		&api.Service{},
		&api.ReplicationControllerList{},
		&api.ReplicationController{},
		&api.MinionList{},
		&api.Minion{},
		&api.Status{},
		&api.ServerOpList{},
		&api.ServerOp{},
		&api.ContainerManifestList{},
		&api.Endpoints{},
		&api.Binding{},
	}
	for _, item := range table {
		// Try a few times, since runTest uses random values.
		for i := 0; i < *fuzzIters; i++ {
			runTest(t, item)
		}
	}
}

func TestEncode_NonPtr(t *testing.T) {
	pod := api.Pod{
		Labels: map[string]string{"name": "foo"},
	}
	obj := interface{}(pod)
	data, err := api.Encode(obj)
	obj2, err2 := api.Decode(data)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'", err, err2)
	}
	if _, ok := obj2.(*api.Pod); !ok {
		t.Fatalf("Got wrong type")
	}
	if !reflect.DeepEqual(obj2, &pod) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", &pod, obj2)
	}
}

func TestEncode_Ptr(t *testing.T) {
	pod := &api.Pod{
		Labels: map[string]string{"name": "foo"},
	}
	obj := interface{}(pod)
	data, err := api.Encode(obj)
	obj2, err2 := api.Decode(data)
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
	if _, err := api.Decode(badJSONMissingKind); err == nil {
		t.Errorf("Did not reject despite lack of kind field: %s", badJSONMissingKind)
	}
	badJSONUnknownType := []byte(`{"kind": "bar"}`)
	if _, err1 := api.Decode(badJSONUnknownType); err1 == nil {
		t.Errorf("Did not reject despite use of unknown type: %s", badJSONUnknownType)
	}
	/*badJSONKindMismatch := []byte(`{"kind": "Pod"}`)
	if err2 := DecodeInto(badJSONKindMismatch, &Minion{}); err2 == nil {
		t.Errorf("Kind is set but doesn't match the object type: %s", badJSONKindMismatch)
	}*/
}
