/*
Copyright 2015 The Kubernetes Authors.

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

package json_test

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	runtimetesting "k8s.io/apimachinery/pkg/runtime/testing"
	"k8s.io/apimachinery/pkg/util/diff"

	"github.com/google/go-cmp/cmp"
)

type testDecodable struct {
	metav1.TypeMeta `json:",inline"`

	Other     string
	Value     int           `json:"value"`
	Spec      DecodableSpec `json:"spec"`
	Interface interface{}   `json:"interface"`
}

// DecodableSpec has 15 fields.
type DecodableSpec struct {
	A int `json:"A"`
	B int `json:"B"`
	C int `json:"C"`
	D int `json:"D"`
	E int `json:"E"`
	F int `json:"F"`
	G int `json:"G"`
	H int `json:"h"`
	I int `json:"i"`
	J int `json:"j"`
	K int `json:"k"`
	L int `json:"l"`
	M int `json:"m"`
	N int `json:"n"`
	O int `json:"o"`
}

func (d *testDecodable) DeepCopyObject() runtime.Object {
	if d == nil {
		return nil
	}
	out := new(testDecodable)
	d.DeepCopyInto(out)
	return out
}
func (d *testDecodable) DeepCopyInto(out *testDecodable) {
	*out = *d
	out.Other = d.Other
	out.Value = d.Value
	out.Spec = d.Spec
	out.Interface = d.Interface
	return
}

type testDecodeCoercion struct {
	metav1.TypeMeta `json:",inline"`

	Bool bool `json:"bool"`

	Int   int `json:"int"`
	Int32 int `json:"int32"`
	Int64 int `json:"int64"`

	Float32 float32 `json:"float32"`
	Float64 float64 `json:"float64"`

	String string `json:"string"`

	Struct testDecodable `json:"struct"`

	Array []string          `json:"array"`
	Map   map[string]string `json:"map"`
}

func (d *testDecodeCoercion) DeepCopyObject() runtime.Object {
	if d == nil {
		return nil
	}
	out := new(testDecodeCoercion)
	d.DeepCopyInto(out)
	return out
}
func (d *testDecodeCoercion) DeepCopyInto(out *testDecodeCoercion) {
	*out = *d
	return
}

func TestDecode(t *testing.T) {
	type testCase struct {
		creater runtime.ObjectCreater
		typer   runtime.ObjectTyper
		yaml    bool
		pretty  bool
		strict  bool

		data       []byte
		defaultGVK *schema.GroupVersionKind
		into       runtime.Object

		errFn          func(error) bool
		expectedObject runtime.Object
		expectedGVK    *schema.GroupVersionKind
	}

	testCases := []testCase{
		// missing metadata without into, typed creater
		{
			data: []byte("{}"),

			expectedGVK: &schema.GroupVersionKind{},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
		},
		{
			data: []byte("{}"),

			expectedGVK: &schema.GroupVersionKind{},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
			strict:      true,
		},

		{
			data: []byte(`{"kind":"Foo"}`),

			expectedGVK: &schema.GroupVersionKind{Kind: "Foo"},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'apiVersion' is missing in") },
		},
		{
			data: []byte(`{"kind":"Foo"}`),

			expectedGVK: &schema.GroupVersionKind{Kind: "Foo"},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'apiVersion' is missing in") },
			strict:      true,
		},

		{
			data: []byte(`{"apiVersion":"foo/v1"}`),

			expectedGVK: &schema.GroupVersionKind{Group: "foo", Version: "v1"},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
		},
		{
			data: []byte(`{"apiVersion":"foo/v1"}`),

			expectedGVK: &schema.GroupVersionKind{Group: "foo", Version: "v1"},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
			strict:      true,
		},

		{
			data:    []byte(`{"apiVersion":"/v1","kind":"Foo"}`),
			typer:   &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			creater: &mockCreater{obj: &testDecodable{}},

			expectedObject: &testDecodable{TypeMeta: metav1.TypeMeta{APIVersion: "/v1", Kind: "Foo"}},
			expectedGVK:    &schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Foo"},
		},
		{
			data:    []byte(`{"apiVersion":"/v1","kind":"Foo"}`),
			typer:   &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			creater: &mockCreater{obj: &testDecodable{}},

			expectedObject: &testDecodable{TypeMeta: metav1.TypeMeta{APIVersion: "/v1", Kind: "Foo"}},
			expectedGVK:    &schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Foo"},
			strict:         true,
		},

		// missing metadata with unstructured into
		{
			data:  []byte("{}"),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{},

			expectedGVK: &schema.GroupVersionKind{},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
		},
		{
			data:  []byte("{}"),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{},

			expectedGVK: &schema.GroupVersionKind{},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
			strict:      true,
		},

		{
			data:  []byte(`{"kind":"Foo"}`),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{},

			expectedGVK:    &schema.GroupVersionKind{Kind: "Foo"},
			expectedObject: &unstructured.Unstructured{Object: map[string]interface{}{"kind": "Foo"}},
			// TODO(109023): expect this to error; unstructured decoding currently only requires kind to be set, not apiVersion
		},
		{
			data:  []byte(`{"kind":"Foo"}`),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{},

			expectedGVK:    &schema.GroupVersionKind{Kind: "Foo"},
			expectedObject: &unstructured.Unstructured{Object: map[string]interface{}{"kind": "Foo"}},
			strict:         true,
			// TODO(109023): expect this to error; unstructured decoding currently only requires kind to be set, not apiVersion
		},

		{
			data:  []byte(`{"apiVersion":"foo/v1"}`),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{},

			expectedGVK: &schema.GroupVersionKind{Group: "foo", Version: "v1"},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
		},
		{
			data:  []byte(`{"apiVersion":"foo/v1"}`),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{},

			expectedGVK: &schema.GroupVersionKind{Group: "foo", Version: "v1"},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
			strict:      true,
		},

		{
			data:  []byte(`{"apiVersion":"/v1","kind":"Foo"}`),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{},

			expectedGVK:    &schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Foo"},
			expectedObject: &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "/v1", "kind": "Foo"}},
		},
		{
			data:  []byte(`{"apiVersion":"/v1","kind":"Foo"}`),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{},

			expectedGVK:    &schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Foo"},
			expectedObject: &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "/v1", "kind": "Foo"}},
			strict:         true,
		},

		// missing metadata with unstructured into providing metadata
		{
			data:  []byte("{}"),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "into/v1", "kind": "Into"}},

			expectedGVK: &schema.GroupVersionKind{},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
		},
		{
			data:  []byte("{}"),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "into/v1", "kind": "Into"}},

			expectedGVK: &schema.GroupVersionKind{},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
			strict:      true,
		},

		{
			data:  []byte(`{"kind":"Foo"}`),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "into/v1", "kind": "Into"}},

			expectedGVK:    &schema.GroupVersionKind{Kind: "Foo"},
			expectedObject: &unstructured.Unstructured{Object: map[string]interface{}{"kind": "Foo"}},
			// TODO(109023): expect this to error; unstructured decoding currently only requires kind to be set, not apiVersion
		},
		{
			data:  []byte(`{"kind":"Foo"}`),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "into/v1", "kind": "Into"}},

			expectedGVK:    &schema.GroupVersionKind{Kind: "Foo"},
			expectedObject: &unstructured.Unstructured{Object: map[string]interface{}{"kind": "Foo"}},
			strict:         true,
			// TODO(109023): expect this to error; unstructured decoding currently only requires kind to be set, not apiVersion
		},

		{
			data:  []byte(`{"apiVersion":"foo/v1"}`),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "into/v1", "kind": "Into"}},

			expectedGVK: &schema.GroupVersionKind{Group: "foo", Version: "v1"},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
		},
		{
			data:  []byte(`{"apiVersion":"foo/v1"}`),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "into/v1", "kind": "Into"}},

			expectedGVK: &schema.GroupVersionKind{Group: "foo", Version: "v1"},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
			strict:      true,
		},

		{
			data:  []byte(`{"apiVersion":"/v1","kind":"Foo"}`),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "into/v1", "kind": "Into"}},

			expectedGVK:    &schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Foo"},
			expectedObject: &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "/v1", "kind": "Foo"}},
		},
		{
			data:  []byte(`{"apiVersion":"/v1","kind":"Foo"}`),
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			into:  &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "into/v1", "kind": "Into"}},

			expectedGVK:    &schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Foo"},
			expectedObject: &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "/v1", "kind": "Foo"}},
			strict:         true,
		},

		// missing metadata without into, unstructured creater
		{
			data:    []byte("{}"),
			typer:   &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			creater: &mockCreater{obj: &unstructured.Unstructured{}},

			expectedGVK: &schema.GroupVersionKind{},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
		},
		{
			data:    []byte("{}"),
			typer:   &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			creater: &mockCreater{obj: &unstructured.Unstructured{}},

			expectedGVK: &schema.GroupVersionKind{},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
			strict:      true,
		},

		{
			data:    []byte(`{"kind":"Foo"}`),
			typer:   &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			creater: &mockCreater{obj: &unstructured.Unstructured{}},

			expectedGVK: &schema.GroupVersionKind{Kind: "Foo"},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'apiVersion' is missing in") },
		},
		{
			data:    []byte(`{"kind":"Foo"}`),
			typer:   &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			creater: &mockCreater{obj: &unstructured.Unstructured{}},

			expectedGVK: &schema.GroupVersionKind{Kind: "Foo"},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'apiVersion' is missing in") },
			strict:      true,
		},

		{
			data:    []byte(`{"apiVersion":"foo/v1"}`),
			typer:   &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			creater: &mockCreater{obj: &unstructured.Unstructured{}},

			expectedGVK: &schema.GroupVersionKind{Group: "foo", Version: "v1"},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
		},
		{
			data:    []byte(`{"apiVersion":"foo/v1"}`),
			typer:   &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			creater: &mockCreater{obj: &unstructured.Unstructured{}},

			expectedGVK: &schema.GroupVersionKind{Group: "foo", Version: "v1"},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
			strict:      true,
		},

		{
			data:    []byte(`{"apiVersion":"/v1","kind":"Foo"}`),
			typer:   &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			creater: &mockCreater{obj: &unstructured.Unstructured{}},

			expectedGVK:    &schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Foo"},
			expectedObject: &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "/v1", "kind": "Foo"}},
		},
		{
			data:    []byte(`{"apiVersion":"/v1","kind":"Foo"}`),
			typer:   &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			creater: &mockCreater{obj: &unstructured.Unstructured{}},

			expectedGVK:    &schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Foo"},
			expectedObject: &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "/v1", "kind": "Foo"}},
			strict:         true,
		},

		// creator errors
		{
			data:       []byte("{}"),
			defaultGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			creater:    &mockCreater{err: fmt.Errorf("fake error")},

			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			errFn:       func(err error) bool { return err.Error() == "fake error" },
		},
		{
			data:       []byte("{}"),
			defaultGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			creater:    &mockCreater{err: fmt.Errorf("fake error")},

			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			errFn:       func(err error) bool { return err.Error() == "fake error" },
		},
		// creator typed
		{
			data:           []byte("{}"),
			defaultGVK:     &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
		},
		{
			data:           []byte("{}"),
			defaultGVK:     &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			strict:         true,
		},

		// version without group is not defaulted
		{
			data:           []byte(`{"apiVersion":"blah"}`),
			defaultGVK:     &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{TypeMeta: metav1.TypeMeta{APIVersion: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "", Version: "blah"},
		},
		// group without version is defaulted
		{
			data:           []byte(`{"apiVersion":"other/"}`),
			defaultGVK:     &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{TypeMeta: metav1.TypeMeta{APIVersion: "other/"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
		},
		// group version, kind is defaulted
		{
			data:           []byte(`{"apiVersion":"other1/blah1"}`),
			defaultGVK:     &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{TypeMeta: metav1.TypeMeta{APIVersion: "other1/blah1"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other1", Version: "blah1"},
		},
		// gvk all provided then not defaulted at all
		{
			data:           []byte(`{"kind":"Test","apiVersion":"other/blah"}`),
			defaultGVK:     &schema.GroupVersionKind{Kind: "Test1", Group: "other1", Version: "blah1"},
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{TypeMeta: metav1.TypeMeta{APIVersion: "other/blah", Kind: "Test"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
		},
		//gvk defaulting if kind not provided in data and defaultGVK use into's kind
		{
			data:           []byte(`{"apiVersion":"b1/c1"}`),
			into:           &testDecodable{TypeMeta: metav1.TypeMeta{Kind: "a3", APIVersion: "b1/c1"}},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "a3", Group: "b1", Version: "c1"}},
			defaultGVK:     nil,
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{TypeMeta: metav1.TypeMeta{Kind: "a3", APIVersion: "b1/c1"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "a3", Group: "b1", Version: "c1"},
		},

		// accept runtime.Unknown as into and bypass creator
		{
			data: []byte(`{}`),
			into: &runtime.Unknown{},

			expectedGVK: &schema.GroupVersionKind{},
			expectedObject: &runtime.Unknown{
				Raw:         []byte(`{}`),
				ContentType: runtime.ContentTypeJSON,
			},
		},
		{
			data: []byte(`{"test":"object"}`),
			into: &runtime.Unknown{},

			expectedGVK: &schema.GroupVersionKind{},
			expectedObject: &runtime.Unknown{
				Raw:         []byte(`{"test":"object"}`),
				ContentType: runtime.ContentTypeJSON,
			},
		},
		{
			data:        []byte(`{"test":"object"}`),
			into:        &runtime.Unknown{},
			defaultGVK:  &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &runtime.Unknown{
				TypeMeta:    runtime.TypeMeta{APIVersion: "other/blah", Kind: "Test"},
				Raw:         []byte(`{"test":"object"}`),
				ContentType: runtime.ContentTypeJSON,
			},
		},

		// unregistered objects can be decoded into directly
		{
			data:        []byte(`{"kind":"Test","apiVersion":"other/blah","value":1,"Other":"test"}`),
			into:        &testDecodable{},
			typer:       &mockTyper{err: runtime.NewNotRegisteredErrForKind("mock", schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"})},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodable{
				TypeMeta: metav1.TypeMeta{APIVersion: "other/blah", Kind: "Test"},
				Other:    "test",
				Value:    1,
			},
		},
		// registered types get defaulted by the into object kind
		{
			data:        []byte(`{"value":1,"Other":"test"}`),
			into:        &testDecodable{},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodable{
				Other: "test",
				Value: 1,
			},
		},
		// registered types get defaulted by the into object kind even without version, but return an error
		{
			data:        []byte(`{"value":1,"Other":"test"}`),
			into:        &testDecodable{},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: ""}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: ""},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'apiVersion' is missing in") },
			expectedObject: &testDecodable{
				Other: "test",
				Value: 1,
			},
		},
		// Error on invalid number
		{
			data:        []byte(`{"kind":"Test","apiVersion":"other/blah","interface":1e1000}`),
			creater:     &mockCreater{obj: &testDecodable{}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), `json: cannot unmarshal number 1e1000 into Go struct field testDecodable.interface of type float64`)
			},
		},
		// Unmarshalling is case-sensitive
		{
			// "VaLue" should have been "value"
			data:        []byte(`{"kind":"Test","apiVersion":"other/blah","VaLue":1,"Other":"test"}`),
			into:        &testDecodable{},
			typer:       &mockTyper{err: runtime.NewNotRegisteredErrForKind("mock", schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"})},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodable{
				TypeMeta: metav1.TypeMeta{APIVersion: "other/blah", Kind: "Test"},
				Other:    "test",
			},
		},
		// Unmarshalling is case-sensitive for big struct.
		{
			// "b" should have been "B", "I" should have been "i"
			data:        []byte(`{"kind":"Test","apiVersion":"other/blah","spec": {"A": 1, "b": 2, "h": 3, "I": 4}}`),
			into:        &testDecodable{},
			typer:       &mockTyper{err: runtime.NewNotRegisteredErrForKind("mock", schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"})},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodable{
				TypeMeta: metav1.TypeMeta{APIVersion: "other/blah", Kind: "Test"},
				Spec:     DecodableSpec{A: 1, H: 3},
			},
		},
		// Unknown fields should return an error from the strict JSON deserializer.
		{
			data:        []byte(`{"unknown": 1}`),
			into:        &testDecodable{},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), `unknown field "unknown"`)
			},
			strict: true,
		},
		// Unknown fields should return an error from the strict YAML deserializer.
		{
			data:        []byte("unknown: 1\n"),
			into:        &testDecodable{},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), `unknown field "unknown"`)
			},
			yaml:   true,
			strict: true,
		},
		// Duplicate fields should return an error from the strict JSON deserializer.
		{
			data:        []byte(`{"value":1,"value":1}`),
			into:        &testDecodable{},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), `duplicate field "value"`)
			},
			strict: true,
		},
		// Duplicate fields should return an error from the strict YAML deserializer.
		{
			data: []byte("value: 1\n" +
				"value: 1\n"),
			into:        &testDecodable{},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), `"value" already set in map`)
			},
			yaml:   true,
			strict: true,
		},
		// Duplicate fields should return an error from the strict JSON deserializer for unstructured.
		{
			data:        []byte(`{"kind":"Custom","value":1,"value":1}`),
			into:        &unstructured.Unstructured{},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Custom"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), `duplicate field "value"`)
			},
			strict: true,
		},
		// Duplicate fields should return an error from the strict YAML deserializer for unstructured.
		{
			data: []byte("kind: Custom\n" +
				"value: 1\n" +
				"value: 1\n"),
			into:        &unstructured.Unstructured{},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Custom"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), `"value" already set in map`)
			},
			yaml:   true,
			strict: true,
		},
		// Strict JSON decode into unregistered objects directly.
		{
			data:        []byte(`{"kind":"Test","apiVersion":"other/blah","value":1,"Other":"test"}`),
			into:        &testDecodable{},
			typer:       &mockTyper{err: runtime.NewNotRegisteredErrForKind("mock", schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"})},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodable{
				TypeMeta: metav1.TypeMeta{APIVersion: "other/blah", Kind: "Test"},
				Other:    "test",
				Value:    1,
			},
			strict: true,
		},
		// Strict YAML decode into unregistered objects directly.
		{
			data: []byte("kind: Test\n" +
				"apiVersion: other/blah\n" +
				"value: 1\n" +
				"Other: test\n"),
			into:        &testDecodable{},
			typer:       &mockTyper{err: runtime.NewNotRegisteredErrForKind("mock", schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"})},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodable{
				TypeMeta: metav1.TypeMeta{APIVersion: "other/blah", Kind: "Test"},
				Other:    "test",
				Value:    1,
			},
			yaml:   true,
			strict: true,
		},
		// Valid strict JSON decode without GVK.
		{
			data:        []byte(`{"value":1234}`),
			into:        &testDecodable{},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodable{
				Value: 1234,
			},
			strict: true,
		},
		// Valid strict YAML decode without GVK.
		{
			data:        []byte("value: 1234\n"),
			into:        &testDecodable{},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodable{
				Value: 1234,
			},
			yaml:   true,
			strict: true,
		},
		// Invalid strict JSON, results in json parse error:
		{
			data:  []byte("foo"),
			into:  &unstructured.Unstructured{},
			typer: &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), `json parse error: invalid character 'o'`)
			},
			strict: true,
		},
		// empty JSON strict, results in missing kind error
		{
			data:        []byte("{}"),
			into:        &unstructured.Unstructured{},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK: &schema.GroupVersionKind{},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), `Object 'Kind' is missing`)
			},
			strict: true,
		},
		// coerce from null
		{
			data:           []byte(`{"bool":null,"int":null,"int32":null,"int64":null,"float32":null,"float64":null,"string":null,"array":null,"map":null,"struct":null}`),
			into:           &testDecodeCoercion{},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodeCoercion{},
			strict:         true,
		},
		{
			data:           []byte(`{"bool":null,"int":null,"int32":null,"int64":null,"float32":null,"float64":null,"string":null,"array":null,"map":null,"struct":null}`),
			into:           &testDecodeCoercion{},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodeCoercion{},
			yaml:           true,
			strict:         true,
		},
		// coerce from string
		{
			data:           []byte(`{"string":""}`),
			into:           &testDecodeCoercion{},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodeCoercion{},
			strict:         true,
		},
		{
			data:           []byte(`{"string":""}`),
			into:           &testDecodeCoercion{},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodeCoercion{},
			yaml:           true,
			strict:         true,
		},
		// coerce from array
		{
			data:           []byte(`{"array":[]}`),
			into:           &testDecodeCoercion{},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodeCoercion{Array: []string{}},
			strict:         true,
		},
		{
			data:           []byte(`{"array":[]}`),
			into:           &testDecodeCoercion{},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodeCoercion{Array: []string{}},
			yaml:           true,
			strict:         true,
		},
		// coerce from map
		{
			data:           []byte(`{"map":{},"struct":{}}`),
			into:           &testDecodeCoercion{},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodeCoercion{Map: map[string]string{}},
			strict:         true,
		},
		{
			data:           []byte(`{"map":{},"struct":{}}`),
			into:           &testDecodeCoercion{},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodeCoercion{Map: map[string]string{}},
			yaml:           true,
			strict:         true,
		},
		// coerce from int
		{
			data:           []byte(`{"int":1,"int32":1,"int64":1,"float32":1,"float64":1}`),
			into:           &testDecodeCoercion{},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodeCoercion{Int: 1, Int32: 1, Int64: 1, Float32: 1, Float64: 1},
			strict:         true,
		},
		{
			data:           []byte(`{"int":1,"int32":1,"int64":1,"float32":1,"float64":1}`),
			into:           &testDecodeCoercion{},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodeCoercion{Int: 1, Int32: 1, Int64: 1, Float32: 1, Float64: 1},
			yaml:           true,
			strict:         true,
		},
		// coerce from float
		{
			data:           []byte(`{"float32":1.0,"float64":1.0}`),
			into:           &testDecodeCoercion{},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodeCoercion{Float32: 1, Float64: 1},
			strict:         true,
		},
		{
			data:           []byte(`{"int":1.0,"int32":1.0,"int64":1.0,"float32":1.0,"float64":1.0}`), // floating point gets dropped in yaml -> json step
			into:           &testDecodeCoercion{},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &testDecodeCoercion{Int: 1, Int32: 1, Int64: 1, Float32: 1, Float64: 1},
			yaml:           true,
			strict:         true,
		},
	}

	logTestCase := func(t *testing.T, tc testCase) {
		t.Logf("data=%s\n\tinto=%T, yaml=%v, strict=%v", string(tc.data), tc.into, tc.yaml, tc.strict)
	}

	for i, test := range testCases {
		var s runtime.Serializer
		if test.yaml {
			s = json.NewSerializerWithOptions(json.DefaultMetaFactory, test.creater, test.typer, json.SerializerOptions{Yaml: test.yaml, Pretty: false, Strict: test.strict})
		} else {
			s = json.NewSerializerWithOptions(json.DefaultMetaFactory, test.creater, test.typer, json.SerializerOptions{Yaml: test.yaml, Pretty: test.pretty, Strict: test.strict})
		}
		obj, gvk, err := s.Decode([]byte(test.data), test.defaultGVK, test.into)

		if !reflect.DeepEqual(test.expectedGVK, gvk) {
			logTestCase(t, test)
			t.Errorf("%d: unexpected GVK: %v", i, gvk)
		}

		switch {
		case err == nil && test.errFn != nil:
			logTestCase(t, test)
			t.Errorf("%d: failed: not getting the expected error", i)
			continue
		case err != nil && test.errFn == nil:
			logTestCase(t, test)
			t.Errorf("%d: failed: %v", i, err)
			continue
		case err != nil:
			if !test.errFn(err) {
				logTestCase(t, test)
				t.Errorf("%d: failed: %v", i, err)
			}
			if !runtime.IsStrictDecodingError(err) && obj != nil {
				logTestCase(t, test)
				t.Errorf("%d: should have returned nil object", i)
			}
			continue
		}

		if test.into != nil && test.into != obj {
			logTestCase(t, test)
			t.Errorf("%d: expected into to be returned: %v", i, obj)
			continue
		}

		if !reflect.DeepEqual(test.expectedObject, obj) {
			logTestCase(t, test)
			t.Errorf("%d: unexpected object:\n%s", i, diff.ObjectGoPrintSideBySide(test.expectedObject, obj))
		}
	}
}

func TestCacheableObject(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "group", Version: "version", Kind: "MockCacheableObject"}
	creater := &mockCreater{obj: &runtimetesting.MockCacheableObject{}}
	typer := &mockTyper{gvk: &gvk}
	serializer := json.NewSerializer(json.DefaultMetaFactory, creater, typer, false)

	runtimetesting.CacheableObjectTest(t, serializer)
}

type mockCreater struct {
	apiVersion string
	kind       string
	err        error
	obj        runtime.Object
}

func (c *mockCreater) New(kind schema.GroupVersionKind) (runtime.Object, error) {
	c.apiVersion, c.kind = kind.GroupVersion().String(), kind.Kind
	return c.obj, c.err
}

type mockTyper struct {
	gvk *schema.GroupVersionKind
	err error
}

func (t *mockTyper) ObjectKinds(obj runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	if t.gvk == nil {
		return nil, false, t.err
	}
	return []schema.GroupVersionKind{*t.gvk}, false, t.err
}

func (t *mockTyper) Recognizes(_ schema.GroupVersionKind) bool {
	return false
}

type testEncodableDuplicateTag struct {
	metav1.TypeMeta `json:",inline"`

	A1 int `json:"a"`
	A2 int `json:"a"` //nolint:govet // This is intentional to test that the encoder will not encode two map entries with the same key.
}

func (testEncodableDuplicateTag) DeepCopyObject() runtime.Object {
	panic("unimplemented")
}

type testEncodableTagMatchesUntaggedName struct {
	metav1.TypeMeta `json:",inline"`

	A       int
	TaggedA int `json:"A"`
}

func (testEncodableTagMatchesUntaggedName) DeepCopyObject() runtime.Object {
	panic("unimplemented")
}

type staticTextMarshaler int

func (staticTextMarshaler) MarshalText() ([]byte, error) {
	return []byte("static"), nil
}

type testEncodableMap[K comparable] map[K]interface{}

func (testEncodableMap[K]) GetObjectKind() schema.ObjectKind {
	panic("unimplemented")
}

func (testEncodableMap[K]) DeepCopyObject() runtime.Object {
	panic("unimplemented")
}

func TestEncode(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   runtime.Object
		want []byte
	}{
		// The Go visibility rules for struct fields are amended for JSON when deciding
		// which field to marshal or unmarshal. If there are multiple fields at the same
		// level, and that level is the least nested (and would therefore be the nesting
		// level selected by the usual Go rules), the following extra rules apply:

		// 1) Of those fields, if any are JSON-tagged, only tagged fields are considered,
		//    even if there are multiple untagged fields that would otherwise conflict.
		{
			name: "only tagged field is considered if any are tagged",
			in: &testEncodableTagMatchesUntaggedName{
				A:       1,
				TaggedA: 2,
			},
			want: []byte("{\"A\":2}\n"),
		},
		// 2) If there is exactly one field (tagged or not according to the first rule),
		//    that is selected.
		// 3) Otherwise there are multiple fields, and all are ignored; no error occurs.
		{
			name: "all duplicate fields are ignored",
			in:   &testEncodableDuplicateTag{},
			want: []byte("{}\n"),
		},
		{
			name: "text marshaler keys can compare inequal but serialize to duplicates",
			in: testEncodableMap[staticTextMarshaler]{
				staticTextMarshaler(1): nil,
				staticTextMarshaler(2): nil,
			},
			want: []byte("{\"static\":null,\"static\":null}\n"),
		},
		{
			name: "time.Time keys can compare inequal but serialize to duplicates because time.Time implements TextMarshaler",
			in: testEncodableMap[time.Time]{
				time.Date(2222, 11, 30, 23, 59, 58, 57, time.UTC):              nil,
				time.Date(2222, 11, 30, 23, 59, 58, 57, time.FixedZone("", 0)): nil,
			},
			want: []byte("{\"2222-11-30T23:59:58.000000057Z\":null,\"2222-11-30T23:59:58.000000057Z\":null}\n"),
		},
		{
			name: "metav1.Time keys can compare inequal but serialize to duplicates because metav1.Time embeds time.Time which implements TextMarshaler",
			in: testEncodableMap[metav1.Time]{
				metav1.Date(2222, 11, 30, 23, 59, 58, 57, time.UTC):              nil,
				metav1.Date(2222, 11, 30, 23, 59, 58, 57, time.FixedZone("", 0)): nil,
			},
			want: []byte("{\"2222-11-30T23:59:58.000000057Z\":null,\"2222-11-30T23:59:58.000000057Z\":null}\n"),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var dst bytes.Buffer
			s := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, json.SerializerOptions{})
			if err := s.Encode(tc.in, &dst); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if diff := cmp.Diff(tc.want, dst.Bytes()); diff != "" {
				t.Errorf("unexpected output:\n%s", diff)
			}
		})
	}
}
