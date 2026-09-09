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

package versioning

import (
	"fmt"
	"io"
	"io/ioutil"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimetesting "k8s.io/apimachinery/pkg/runtime/testing"
	"k8s.io/apimachinery/pkg/util/diff"
)

type testDecodable struct {
	Other string
	Value int `json:"value"`
	gvk   schema.GroupVersionKind
}

func (d *testDecodable) GetObjectKind() schema.ObjectKind                { return d }
func (d *testDecodable) SetGroupVersionKind(gvk schema.GroupVersionKind) { d.gvk = gvk }
func (d *testDecodable) GroupVersionKind() schema.GroupVersionKind       { return d.gvk }
func (d *testDecodable) DeepCopyObject() runtime.Object {
	// no real deepcopy because these tests check for pointer equality
	return d
}

type testNestedDecodable struct {
	Other string
	Value int `json:"value"`

	gvk          schema.GroupVersionKind
	nestedCalled bool
	nestedErr    error
}

func (d *testNestedDecodable) GetObjectKind() schema.ObjectKind                { return d }
func (d *testNestedDecodable) SetGroupVersionKind(gvk schema.GroupVersionKind) { d.gvk = gvk }
func (d *testNestedDecodable) GroupVersionKind() schema.GroupVersionKind       { return d.gvk }
func (d *testNestedDecodable) DeepCopyObject() runtime.Object {
	// no real deepcopy because these tests check for pointer equality
	return d
}

func (d *testNestedDecodable) EncodeNestedObjects(e runtime.Encoder) error {
	d.nestedCalled = true
	return d.nestedErr
}

func (d *testNestedDecodable) DecodeNestedObjects(_ runtime.Decoder) error {
	d.nestedCalled = true
	return d.nestedErr
}

func TestNestedDecode(t *testing.T) {
	n := &testNestedDecodable{nestedErr: fmt.Errorf("unable to decode")}
	decoder := &mockSerializer{obj: n}
	codec := NewCodec(nil, decoder, nil, nil, nil, nil, nil, nil, "TestNestedDecode")
	if _, _, err := codec.Decode([]byte(`{}`), nil, n); err != n.nestedErr {
		t.Errorf("unexpected error: %v", err)
	}
	if !n.nestedCalled {
		t.Errorf("did not invoke nested decoder")
	}
}

func TestNestedDecodeStrictDecodingError(t *testing.T) {
	strictErr := runtime.NewStrictDecodingError([]error{fmt.Errorf("duplicate field")})
	n := &testNestedDecodable{nestedErr: strictErr}
	decoder := &mockSerializer{obj: n}
	codec := NewCodec(nil, decoder, nil, nil, nil, nil, nil, nil, "TestNestedDecode")
	o, _, err := codec.Decode([]byte(`{}`), nil, n)
	if strictErr, ok := runtime.AsStrictDecodingError(err); !ok || err != strictErr {
		t.Errorf("unexpected error: %v", err)
	}
	if o != n {
		t.Errorf("did not successfully decode with strict decoding error: %v", o)
	}
	if !n.nestedCalled {
		t.Errorf("did not invoke nested decoder")
	}
}

func TestNestedEncode(t *testing.T) {
	n := &testNestedDecodable{nestedErr: fmt.Errorf("unable to decode")}
	n2 := &testNestedDecodable{nestedErr: fmt.Errorf("unable to decode 2")}
	encoder := &mockSerializer{obj: n}
	codec := NewCodec(
		encoder, nil,
		&checkConvertor{obj: n2, groupVersion: schema.GroupVersion{Group: "other"}},
		nil,
		&mockTyper{gvks: []schema.GroupVersionKind{{Kind: "test"}}},
		nil,
		schema.GroupVersion{Group: "other"}, nil,
		"TestNestedEncode",
	)
	if err := codec.Encode(n, ioutil.Discard); err != n2.nestedErr {
		t.Errorf("unexpected error: %v", err)
	}
	if n.nestedCalled || !n2.nestedCalled {
		t.Errorf("did not invoke correct nested decoder")
	}
}

func TestNestedEncodeError(t *testing.T) {
	n := &testNestedDecodable{nestedErr: fmt.Errorf("unable to encode")}
	gvk1 := schema.GroupVersionKind{Kind: "test", Group: "other", Version: "v1"}
	gvk2 := schema.GroupVersionKind{Kind: "test", Group: "other", Version: "v2"}
	n.SetGroupVersionKind(gvk1)
	encoder := &mockSerializer{obj: n}
	codec := NewCodec(
		encoder, nil,
		&mockConvertor{},
		nil,
		&mockTyper{gvks: []schema.GroupVersionKind{gvk1, gvk2}},
		nil,
		schema.GroupVersion{Group: "other", Version: "v2"}, nil,
		"TestNestedEncodeError",
	)
	if err := codec.Encode(n, ioutil.Discard); err != n.nestedErr {
		t.Errorf("unexpected error: %v", err)
	}
	if n.GroupVersionKind() != gvk1 {
		t.Errorf("unexpected gvk of input object: %v", n.GroupVersionKind())
	}
}

func TestDecode(t *testing.T) {
	gvk1 := &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}
	decodable1 := &testDecodable{}
	decodable2 := &testDecodable{}
	decodable3 := &testDecodable{}

	testCases := []struct {
		serializer runtime.Serializer
		convertor  runtime.ObjectConvertor
		creater    runtime.ObjectCreater
		typer      runtime.ObjectTyper
		defaulter  runtime.ObjectDefaulter
		yaml       bool
		pretty     bool

		encodes, decodes runtime.GroupVersioner

		defaultGVK *schema.GroupVersionKind
		into       runtime.Object

		errFn          func(error) bool
		expectedObject runtime.Object
		sameObject     runtime.Object
		expectedGVK    *schema.GroupVersionKind
	}{
		{
			serializer:  &mockSerializer{actual: gvk1},
			convertor:   &checkConvertor{groupVersion: schema.GroupVersion{Group: "other", Version: runtime.APIVersionInternal}},
			expectedGVK: gvk1,
			decodes:     schema.GroupVersion{Group: "other", Version: runtime.APIVersionInternal},
		},
		{
			serializer:  &mockSerializer{actual: gvk1, obj: decodable1},
			convertor:   &checkConvertor{in: decodable1, obj: decodable2, groupVersion: schema.GroupVersion{Group: "other", Version: runtime.APIVersionInternal}},
			expectedGVK: gvk1,
			sameObject:  decodable2,
			decodes:     schema.GroupVersion{Group: "other", Version: runtime.APIVersionInternal},
		},
		// defaultGVK.Group is allowed to force a conversion to the destination group
		{
			serializer:  &mockSerializer{actual: gvk1, obj: decodable1},
			defaultGVK:  &schema.GroupVersionKind{Group: "force"},
			convertor:   &checkConvertor{in: decodable1, obj: decodable2, groupVersion: schema.GroupVersion{Group: "force", Version: runtime.APIVersionInternal}},
			expectedGVK: gvk1,
			sameObject:  decodable2,
			decodes:     schema.GroupVersion{Group: "force", Version: runtime.APIVersionInternal},
		},
		// uses direct conversion for into when objects differ
		{
			into:        decodable3,
			serializer:  &mockSerializer{actual: gvk1, obj: decodable1},
			convertor:   &checkConvertor{in: decodable1, obj: decodable3, directConvert: true},
			expectedGVK: gvk1,
			sameObject:  decodable3,
		},
		// decode into the same version as the serialized object
		{
			decodes: schema.GroupVersions{gvk1.GroupVersion()},

			serializer:     &mockSerializer{actual: gvk1, obj: decodable1},
			convertor:      &checkConvertor{in: decodable1, obj: decodable1, groupVersion: schema.GroupVersions{{Group: "other", Version: "blah"}}},
			expectedGVK:    gvk1,
			expectedObject: decodable1,
		},
	}

	for i, test := range testCases {
		t.Logf("%d", i)
		s := NewCodec(test.serializer, test.serializer, test.convertor, test.creater, test.typer, test.defaulter, test.encodes, test.decodes, fmt.Sprintf("mock-%d", i))
		obj, gvk, err := s.Decode([]byte(`{}`), test.defaultGVK, test.into)

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

		switch {
		case test.expectedObject != nil:
			if !reflect.DeepEqual(test.expectedObject, obj) {
				t.Errorf("%d: unexpected object:\n%s", i, diff.ObjectGoPrintSideBySide(test.expectedObject, obj))
			}
		case test.sameObject != nil:
			if test.sameObject != obj {
				t.Errorf("%d: unexpected object:\n%s", i, diff.ObjectGoPrintSideBySide(test.sameObject, obj))
			}
		case obj != nil:
			t.Errorf("%d: unexpected object: %#v", i, obj)
		}
	}
}

type checkConvertor struct {
	err           error
	in, obj       runtime.Object
	groupVersion  runtime.GroupVersioner
	directConvert bool
}

func (c *checkConvertor) Convert(in, out, context interface{}) error {
	if !c.directConvert {
		return fmt.Errorf("unexpected call to Convert")
	}
	if c.in != nil && c.in != in {
		return fmt.Errorf("unexpected in: %s", in)
	}
	if c.obj != nil && c.obj != out {
		return fmt.Errorf("unexpected out: %s", out)
	}
	return c.err
}
func (c *checkConvertor) ConvertToVersion(in runtime.Object, outVersion runtime.GroupVersioner) (out runtime.Object, err error) {
	if c.directConvert {
		return nil, fmt.Errorf("unexpected call to ConvertToVersion")
	}
	if c.in != nil && c.in != in {
		return nil, fmt.Errorf("unexpected in: %s", in)
	}
	if !reflect.DeepEqual(c.groupVersion, outVersion) {
		return nil, fmt.Errorf("unexpected outversion: %s (%s)", outVersion, c.groupVersion)
	}
	return c.obj, c.err
}
func (c *checkConvertor) ConvertFieldLabel(gvk schema.GroupVersionKind, label, value string) (string, string, error) {
	return "", "", fmt.Errorf("unexpected call to ConvertFieldLabel")
}

type mockConvertor struct {
}

func (c *mockConvertor) Convert(in, out, context interface{}) error {
	return fmt.Errorf("unexpect call to Convert")
}

func (c *mockConvertor) ConvertToVersion(in runtime.Object, outVersion runtime.GroupVersioner) (out runtime.Object, err error) {
	objectKind := in.GetObjectKind()
	inGVK := objectKind.GroupVersionKind()
	if out, ok := outVersion.KindForGroupVersionKinds([]schema.GroupVersionKind{inGVK}); ok {
		objectKind.SetGroupVersionKind(out)
	} else {
		return nil, fmt.Errorf("unexpected conversion")
	}
	return in, nil
}

func (c *mockConvertor) ConvertFieldLabel(gvk schema.GroupVersionKind, label, value string) (string, string, error) {
	return "", "", fmt.Errorf("unexpected call to ConvertFieldLabel")
}

type mockSerializer struct {
	err            error
	obj            runtime.Object
	encodingObjGVK schema.GroupVersionKind

	defaults, actual *schema.GroupVersionKind
	into             runtime.Object
}

func (s *mockSerializer) Decode(data []byte, defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	s.defaults = defaults
	s.into = into
	return s.obj, s.actual, s.err
}

func (s *mockSerializer) Encode(obj runtime.Object, w io.Writer) error {
	s.obj = obj
	s.encodingObjGVK = obj.GetObjectKind().GroupVersionKind()
	return s.err
}

func (s *mockSerializer) Identifier() runtime.Identifier {
	return runtime.Identifier("mock")
}

type mockTyper struct {
	gvks        []schema.GroupVersionKind
	unversioned bool
	err         error
}

func (t *mockTyper) ObjectKinds(obj runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	return t.gvks, t.unversioned, t.err
}

func (t *mockTyper) Recognizes(_ schema.GroupVersionKind) bool {
	return true
}

func TestDirectCodecEncode(t *testing.T) {
	serializer := mockSerializer{}
	typer := mockTyper{
		gvks: []schema.GroupVersionKind{
			{
				Group: "wrong_group",
				Kind:  "some_kind",
			},
			{
				Group: "expected_group",
				Kind:  "some_kind",
			},
		},
	}

	c := runtime.WithVersionEncoder{
		Version:     schema.GroupVersion{Group: "expected_group"},
		Encoder:     &serializer,
		ObjectTyper: &typer,
	}
	c.Encode(&testDecodable{}, ioutil.Discard)
	if e, a := "expected_group", serializer.encodingObjGVK.Group; e != a {
		t.Errorf("expected group to be %v, got %v", e, a)
	}
}

func TestCacheableObject(t *testing.T) {
	gvk1 := schema.GroupVersionKind{Group: "group", Version: "version1", Kind: "MockCacheableObject"}
	gvk2 := schema.GroupVersionKind{Group: "group", Version: "version2", Kind: "MockCacheableObject"}

	encoder := NewCodec(
		&mockSerializer{}, &mockSerializer{},
		&mockConvertor{}, nil,
		&mockTyper{gvks: []schema.GroupVersionKind{gvk1, gvk2}}, nil,
		gvk1.GroupVersion(), gvk2.GroupVersion(),
		"TestCacheableObject")

	runtimetesting.CacheableObjectTest(t, encoder)
}

func BenchmarkIdentifier(b *testing.B) {
	encoder := &mockSerializer{}
	gv := schema.GroupVersion{Group: "group", Version: "version"}

	for i := 0; i < b.N; i++ {
		id := identifier(gv, encoder)
		// Avoid optimizing by compiler.
		if id[0] != '{' {
			b.Errorf("unexpected identifier: %s", id)
		}
	}
}
