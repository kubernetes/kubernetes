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

package api

import (
	"encoding/json"
	"flag"
	"fmt"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

var fuzzIters = flag.Int("fuzz_iters", 3, "How many fuzzing iterations to do.")

// apiObjectFuzzer can randomly populate api objects.
var apiObjectFuzzer = util.NewFuzzer(
	func(j *JSONBase) {
		// We have to customize the randomization of JSONBases because their
		// APIVersion and Kind must remain blank in memory.
		j.APIVersion = ""
		j.Kind = ""
		j.ID = util.RandString()
		// TODO: Fix JSON/YAML packages and/or write custom encoding
		// for uint64's. Somehow the LS *byte* of this is lost, but
		// only when all 8 bytes are set.
		j.ResourceVersion = util.RandUint64() >> 8
		j.SelfLink = util.RandString()
		j.CreationTimestamp = util.RandString()
	},
	func(intstr *util.IntOrString) {
		// util.IntOrString will panic if its kind is set wrong.
		if util.RandBool() {
			intstr.Kind = util.IntstrInt
			intstr.IntVal = int(util.RandUint64())
			intstr.StrVal = ""
		} else {
			intstr.Kind = util.IntstrString
			intstr.IntVal = 0
			intstr.StrVal = util.RandString()
		}
	},
	func(p *PodInfo) {
		// The docker container type doesn't survive fuzzing.
		// TODO: fix this.
		*p = nil
	},
)

func objDiff(a, b interface{}) string {

	// An alternate diff attempt, in case json isn't showing you
	// the difference. (reflect.DeepEqual makes a distinction between
	// nil and empty slices, for example.)
	return util.StringDiff(
		fmt.Sprintf("%#v", a),
		fmt.Sprintf("%#v", b),
	)

	ab, err := json.Marshal(a)
	if err != nil {
		panic("a")
	}
	bb, err := json.Marshal(b)
	if err != nil {
		panic("b")
	}
	return util.StringDiff(string(ab), string(bb))
}

func runTest(t *testing.T, source interface{}) {
	name := reflect.TypeOf(source).Elem().Name()
	apiObjectFuzzer.Fuzz(source)
	j, err := FindJSONBase(source)
	if err != nil {
		t.Fatalf("Unexpected error %v for %#v", err, source)
	}
	j.SetKind("")
	j.SetAPIVersion("")

	data, err := Encode(source)
	if err != nil {
		t.Errorf("%v: %v (%#v)", name, err, source)
		return
	}
	obj2, err := Decode(data)
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
	err = DecodeInto(data, obj3)
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
	// TODO: auto-fill all fields.
	table := []interface{}{
		&PodList{},
		&Pod{},
		&ServiceList{},
		&Service{},
		&ReplicationControllerList{},
		&ReplicationController{},
		&MinionList{},
		&Minion{},
		&Status{},
		&ServerOpList{},
		&ServerOp{},
		&ContainerManifestList{},
		&Endpoints{},
	}
	for _, item := range table {
		// Try a few times, since runTest uses random values.
		for i := 0; i < *fuzzIters; i++ {
			runTest(t, item)
		}
	}
}

func TestEncode_NonPtr(t *testing.T) {
	pod := Pod{
		Labels: map[string]string{"name": "foo"},
	}
	obj := interface{}(pod)
	data, err := Encode(obj)
	obj2, err2 := Decode(data)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'", err, err2)
	}
	if _, ok := obj2.(*Pod); !ok {
		t.Fatalf("Got wrong type")
	}
	if !reflect.DeepEqual(obj2, &pod) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", &pod, obj2)
	}
}

func TestEncode_Ptr(t *testing.T) {
	pod := &Pod{
		Labels: map[string]string{"name": "foo"},
	}
	obj := interface{}(pod)
	data, err := Encode(obj)
	obj2, err2 := Decode(data)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'", err, err2)
	}
	if _, ok := obj2.(*Pod); !ok {
		t.Fatalf("Got wrong type")
	}
	if !reflect.DeepEqual(obj2, pod) {
		t.Errorf("Expected:\n %#v,\n Got:\n %#v", &pod, obj2)
	}
}

func TestBadJSONRejection(t *testing.T) {
	badJSONMissingKind := []byte(`{ }`)
	if _, err := Decode(badJSONMissingKind); err == nil {
		t.Errorf("Did not reject despite lack of kind field: %s", badJSONMissingKind)
	}
	badJSONUnknownType := []byte(`{"kind": "bar"}`)
	if _, err1 := Decode(badJSONUnknownType); err1 == nil {
		t.Errorf("Did not reject despite use of unknown type: %s", badJSONUnknownType)
	}
	/*badJSONKindMismatch := []byte(`{"kind": "Pod"}`)
	if err2 := DecodeInto(badJSONKindMismatch, &Minion{}); err2 == nil {
		t.Errorf("Kind is set but doesn't match the object type: %s", badJSONKindMismatch)
	}*/
}
