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
	codec := NewCodec(nil, decoder, nil, nil, nil, nil, nil, nil)
	if _, _, err := codec.Decode([]byte(`{}`), nil, n); err != n.nestedErr {
		t.Errorf("unexpected error: %v", err)
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
	)
	if err := codec.Encode(n, ioutil.Discard); err != n2.nestedErr {
		t.Errorf("unexpected error: %v", err)
	}
	if n.nestedCalled || !n2.nestedCalled {
		t.Errorf("did not invoke correct nested decoder")
	}
}

func TestDecode(t *testing.T) {
	gvk1 := &schema.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}
	decodable1 := &testDecodable{}
	decodable2 := &testDecodable{}
	decodable3 := &testDecodable{}
	versionedDecodable1 := &runtime.VersionedObjects{Objects: []runtime.Object{decodable1}}

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
			convertor:   &checkConvertor{groupVersion: schema.GroupVersion{Group: "other", Version: "__internal"}},
			expectedGVK: gvk1,
			decodes:     schema.GroupVersion{Group: "other", Version: "__internal"},
		},
		{
			serializer:  &mockSerializer{actual: gvk1, obj: decodable1},
			convertor:   &checkConvertor{in: decodable1, obj: decodable2, groupVersion: schema.GroupVersion{Group: "other", Version: "__internal"}},
			expectedGVK: gvk1,
			sameObject:  decodable2,
			decodes:     schema.GroupVersion{Group: "other", Version: "__internal"},
		},
		// defaultGVK.Group is allowed to force a conversion to the destination group
		{
			serializer:  &mockSerializer{actual: gvk1, obj: decodable1},
			defaultGVK:  &schema.GroupVersionKind{Group: "force"},
			convertor:   &checkConvertor{in: decodable1, obj: decodable2, groupVersion: schema.GroupVersion{Group: "force", Version: "__internal"}},
			expectedGVK: gvk1,
			sameObject:  decodable2,
			decodes:     schema.GroupVersion{Group: "force", Version: "__internal"},
		},
		// uses direct conversion for into when objects differ
		{
			into:        decodable3,
			serializer:  &mockSerializer{actual: gvk1, obj: decodable1},
			convertor:   &checkConvertor{in: decodable1, obj: decodable3, directConvert: true},
			expectedGVK: gvk1,
			sameObject:  decodable3,
		},
		{
			into:        versionedDecodable1,
			serializer:  &mockSerializer{actual: gvk1, obj: decodable3},
			convertor:   &checkConvertor{in: decodable3, obj: decodable1, directConvert: true},
			expectedGVK: gvk1,
			sameObject:  versionedDecodable1,
		},
		// returns directly when serializer returns into
		{
			into:        decodable3,
			serializer:  &mockSerializer{actual: gvk1, obj: decodable3},
			expectedGVK: gvk1,
			sameObject:  decodable3,
		},
		// returns directly when serializer returns into
		{
			into:        versionedDecodable1,
			serializer:  &mockSerializer{actual: gvk1, obj: decodable1},
			expectedGVK: gvk1,
			sameObject:  versionedDecodable1,
		},

		// runtime.VersionedObjects are decoded
		{
			into: &runtime.VersionedObjects{Objects: []runtime.Object{}},

			serializer:     &mockSerializer{actual: gvk1, obj: decodable1},
			convertor:      &checkConvertor{in: decodable1, obj: decodable2, groupVersion: schema.GroupVersion{Group: "other", Version: "__internal"}},
			expectedGVK:    gvk1,
			expectedObject: &runtime.VersionedObjects{Objects: []runtime.Object{decodable1, decodable2}},
			decodes:        schema.GroupVersion{Group: "other", Version: "__internal"},
		},

		// decode into the same version as the serialized object
		{
			decodes: schema.GroupVersions{gvk1.GroupVersion()},

			serializer:     &mockSerializer{actual: gvk1, obj: decodable1},
			convertor:      &checkConvertor{in: decodable1, obj: decodable1, groupVersion: schema.GroupVersions{{Group: "other", Version: "blah"}}},
			expectedGVK:    gvk1,
			expectedObject: decodable1,
		},
		{
			into:    &runtime.VersionedObjects{Objects: []runtime.Object{}},
			decodes: schema.GroupVersions{gvk1.GroupVersion()},

			serializer:     &mockSerializer{actual: gvk1, obj: decodable1},
			convertor:      &checkConvertor{in: decodable1, obj: decodable1, groupVersion: schema.GroupVersions{{Group: "other", Version: "blah"}}},
			expectedGVK:    gvk1,
			expectedObject: &runtime.VersionedObjects{Objects: []runtime.Object{decodable1}},
		},

		// codec with non matching version skips conversion altogether
		{
			decodes: schema.GroupVersions{{Group: "something", Version: "else"}},

			serializer:     &mockSerializer{actual: gvk1, obj: decodable1},
			convertor:      &checkConvertor{in: decodable1, obj: decodable1, groupVersion: schema.GroupVersions{{Group: "something", Version: "else"}}},
			expectedGVK:    gvk1,
			expectedObject: decodable1,
		},
		{
			into:    &runtime.VersionedObjects{Objects: []runtime.Object{}},
			decodes: schema.GroupVersions{{Group: "something", Version: "else"}},

			serializer:     &mockSerializer{actual: gvk1, obj: decodable1},
			convertor:      &checkConvertor{in: decodable1, obj: decodable1, groupVersion: schema.GroupVersions{{Group: "something", Version: "else"}}},
			expectedGVK:    gvk1,
			expectedObject: &runtime.VersionedObjects{Objects: []runtime.Object{decodable1}},
		},
	}

	for i, test := range testCases {
		t.Logf("%d", i)
		s := NewCodec(test.serializer, test.serializer, test.convertor, test.creater, test.typer, test.defaulter, test.encodes, test.decodes)
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
func (c *checkConvertor) ConvertFieldLabel(version, kind, label, value string) (string, string, error) {
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

type mockCreater struct {
	err error
	obj runtime.Object
}

func (c *mockCreater) New(kind schema.GroupVersionKind) (runtime.Object, error) {
	return c.obj, c.err
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

	c := DirectEncoder{
		Version:     schema.GroupVersion{Group: "expected_group"},
		Encoder:     &serializer,
		ObjectTyper: &typer,
	}
	c.Encode(&testDecodable{}, ioutil.Discard)
	if e, a := "expected_group", serializer.encodingObjGVK.Group; e != a {
		t.Errorf("expected group to be %v, got %v", e, a)
	}
}
