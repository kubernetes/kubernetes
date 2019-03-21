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
	"fmt"
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/util/diff"
)

type testDecodable struct {
	Other     string
	Value     int           `json:"value"`
	Spec      DecodableSpec `json:"spec"`
	Interface interface{}   `json:"interface"`
	gvk       schema.GroupVersionKind
}

// DecodableSpec has 15 fields. json-iterator treats struct with more than 10
// fields differently from struct that has less than 10 fields.
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

func (d *testDecodable) GetObjectKind() schema.ObjectKind                { return d }
func (d *testDecodable) SetGroupVersionKind(gvk schema.GroupVersionKind) { d.gvk = gvk }
func (d *testDecodable) GroupVersionKind() schema.GroupVersionKind       { return d.gvk }
func (d *testDecodable) DeepCopyObject() runtime.Object {
	panic("testDecodable does not support DeepCopy")
}

func TestDecode(t *testing.T) {
	testCases := []struct {
		creater runtime.ObjectCreater
		typer   runtime.ObjectTyper
		yaml    bool
		pretty  bool

		data       []byte
		defaultGVK *schema.GroupVersionKind
		into       runtime.Object

		errFn          func(error) bool
		expectedObject runtime.Object
		expectedGVK    *schema.GroupVersionKind
	}{
		{
			data: []byte("{}"),

			expectedGVK: &schema.GroupVersionKind{},
			errFn:       func(err error) bool { return strings.Contains(err.Error(), "Object 'Kind' is missing in") },
		},
		{
			data:       []byte("{}"),
			defaultGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			creater:    &mockCreater{err: fmt.Errorf("fake error")},

			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			errFn:       func(err error) bool { return err.Error() == "fake error" },
		},
		{
			data:           []byte("{}"),
			defaultGVK:     &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
		},

		// version without group is not defaulted
		{
			data:           []byte(`{"apiVersion":"blah"}`),
			defaultGVK:     &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "", Version: "blah"},
		},
		// group without version is defaulted
		{
			data:           []byte(`{"apiVersion":"other/"}`),
			defaultGVK:     &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
		},
		// group version, kind is defaulted
		{
			data:           []byte(`{"apiVersion":"other1/blah1"}`),
			defaultGVK:     &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other1", Version: "blah1"},
		},
		// gvk all provided then not defaulted at all
		{
			data:           []byte(`{"kind":"Test","apiVersion":"other/blah"}`),
			defaultGVK:     &schema.GroupVersionKind{Kind: "Test1", Group: "other1", Version: "blah1"},
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{},
			expectedGVK:    &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
		},
		//gvk defaulting if kind not provided in data and defaultGVK use into's kind
		{
			data:           []byte(`{"apiVersion":"b1/c1"}`),
			into:           &testDecodable{gvk: schema.GroupVersionKind{Kind: "a3", Group: "b1", Version: "c1"}},
			typer:          &mockTyper{gvk: &schema.GroupVersionKind{Kind: "a3", Group: "b1", Version: "c1"}},
			defaultGVK:     nil,
			creater:        &mockCreater{obj: &testDecodable{}},
			expectedObject: &testDecodable{gvk: schema.GroupVersionKind{Kind: "a3", Group: "b1", Version: "c1"}},
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
				Other: "test",
				Value: 1,
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

		// runtime.VersionedObjects are decoded
		{
			data:        []byte(`{"value":1,"Other":"test"}`),
			into:        &runtime.VersionedObjects{Objects: []runtime.Object{}},
			creater:     &mockCreater{obj: &testDecodable{}},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			defaultGVK:  &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &runtime.VersionedObjects{
				Objects: []runtime.Object{
					&testDecodable{
						Other: "test",
						Value: 1,
					},
				},
			},
		},
		// runtime.VersionedObjects with an object are decoded into
		{
			data:        []byte(`{"Other":"test"}`),
			into:        &runtime.VersionedObjects{Objects: []runtime.Object{&testDecodable{Value: 2}}},
			typer:       &mockTyper{gvk: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			expectedObject: &runtime.VersionedObjects{
				Objects: []runtime.Object{
					&testDecodable{
						Other: "test",
						Value: 2,
					},
				},
			},
		},
		// Error on invalid number
		{
			data:        []byte(`{"kind":"Test","apiVersion":"other/blah","interface":1e1000}`),
			creater:     &mockCreater{obj: &testDecodable{}},
			expectedGVK: &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"},
			errFn: func(err error) bool {
				return strings.Contains(err.Error(), `json_test.testDecodable.Interface: DecodeNumber: strconv.ParseFloat: parsing "1e1000": value out of range`)
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
				Other: "test",
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
				Spec: DecodableSpec{A: 1, H: 3},
			},
		},
	}

	for i, test := range testCases {
		var s runtime.Serializer
		if test.yaml {
			s = json.NewYAMLSerializer(json.DefaultMetaFactory, test.creater, test.typer)
		} else {
			s = json.NewSerializer(json.DefaultMetaFactory, test.creater, test.typer, test.pretty)
		}
		obj, gvk, err := s.Decode([]byte(test.data), test.defaultGVK, test.into)

		if !reflect.DeepEqual(test.expectedGVK, gvk) {
			t.Errorf("%d: unexpected GVK: %v", i, gvk)
		}

		switch {
		case err == nil && test.errFn != nil:
			t.Errorf("%d: failed: %v", i, err)
			continue
		case err != nil && test.errFn == nil:
			t.Errorf("%d: failed: %v", i, err)
			continue
		case err != nil:
			if !test.errFn(err) {
				t.Errorf("%d: failed: %v", i, err)
			}
			if obj != nil {
				t.Errorf("%d: should have returned nil object", i)
			}
			continue
		}

		if test.into != nil && test.into != obj {
			t.Errorf("%d: expected into to be returned: %v", i, obj)
			continue
		}

		if !reflect.DeepEqual(test.expectedObject, obj) {
			t.Errorf("%d: unexpected object:\n%s", i, diff.ObjectGoPrintSideBySide(test.expectedObject, obj))
		}
	}
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
