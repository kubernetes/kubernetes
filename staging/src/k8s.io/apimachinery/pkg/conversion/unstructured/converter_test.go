/*
Copyright 2017 The Kubernetes Authors.

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

package unstructured

import (
	"reflect"
	"testing"

	"time"

	"github.com/json-iterator/go"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func Test_deepCopyJSON(t *testing.T) {
	type args struct {
		x interface{}
	}
	tests := []string{
		`{}`,
		`{"x":42}`,
		`{"x":3.14159265}`,
		`{"x":"foo"}`,
		`{"x":[1,2,3]}`,
		`{"x":{"foo":"bar"}}`,
		`{"x":null}`,
	}
	for _, tt := range tests {
		t.Run(tt, func(t *testing.T) {
			js := jsonDecodeOrDie(t, tt)
			clone := DeepCopyJSONValue(js)
			bs, err := jsoniter.Marshal(clone)
			if err != nil {
				t.Fatalf("Failed to marshal %#v: %v", clone, err)
			}
			if tt != string(bs) {
				t.Errorf("deepCopyJSON() = %v, want %v", string(bs), tt)
			}
		})
	}
}

func jsonDecodeOrDie(t *testing.T, x string) interface{} {
	var result interface{}
	if err := jsoniter.Unmarshal([]byte(x), &result); err != nil {
		t.Fatalf("Failed to decode %q via jsoniter: %v", x, err)
	}
	return result
}

func Test_converterImpl_ToUnstructured(t *testing.T) {
	fourtyTwo := 42
	tests := []struct {
		name string
		obj  interface{}
	}{
		{"struct", &Obj{42, 3.14159265, "foo", []int64{1, 2, 3}, map[string]int64{"foo": 42}, &fourtyTwo, nil, metav1.NewTime(time.Now()), nil}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &converterImpl{}
			u, err := c.ToUnstructured(tt.obj)
			if err != nil {
				t.Errorf("converterImpl.ToUnstructured() error: %v", err)
				return
			}

			clone := DeepCopyJSON(u)
			if !reflect.DeepEqual(u, clone) {
				t.Errorf("DeepCopyJSON = %#v, want %#v", clone, u)
			}
		})
	}
}

type Obj struct {
	Int     int64            `json:"int"`
	Float64 float64          `json:"float64"`
	String  string           `json:"string"`
	Slice   []int64          `json:"slice"`
	Map     map[string]int64 `json:"map"`
	Ptr     *int             `json:"ptr,omitempty"`
	Nil     *int             `json:"nil,omitempty"`
	Time    metav1.Time      `json:"time,omitempty"`
	TimePtr *metav1.Time     `json:"timePtr,omitempty"`
}
