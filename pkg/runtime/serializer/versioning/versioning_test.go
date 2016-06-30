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
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/diff"
)

type testDecodable struct {
	Other string
	Value int `json:"value"`
	gvk   unversioned.GroupVersionKind
}

func (d *testDecodable) GetObjectKind() unversioned.ObjectKind                { return d }
func (d *testDecodable) SetGroupVersionKind(gvk unversioned.GroupVersionKind) { d.gvk = gvk }
func (d *testDecodable) GroupVersionKind() unversioned.GroupVersionKind       { return d.gvk }

func TestDecode(t *testing.T) {
	gvk1 := &unversioned.GroupVersionKind{Kind: "Test", Group: "other", Version: "blah"}
	decodable1 := &testDecodable{}
	decodable2 := &testDecodable{}
	decodable3 := &testDecodable{}
	versionedDecodable1 := &runtime.VersionedObjects{Objects: []runtime.Object{decodable1}}

	testCases := []struct {
		serializer runtime.Serializer
		convertor  runtime.ObjectConvertor
		creater    runtime.ObjectCreater
		copier     runtime.ObjectCopier
		typer      runtime.ObjectTyper
		yaml       bool
		pretty     bool

		encodes, decodes []unversioned.GroupVersion

		defaultGVK *unversioned.GroupVersionKind
		into       runtime.Object

		errFn          func(error) bool
		expectedObject runtime.Object
		sameObject     runtime.Object
		expectedGVK    *unversioned.GroupVersionKind
	}{
		{
			serializer:  &mockSerializer{actual: gvk1},
			convertor:   &checkConvertor{groupVersion: unversioned.GroupVersion{Group: "other", Version: "__internal"}},
			expectedGVK: gvk1,
		},
		{
			serializer:  &mockSerializer{actual: gvk1, obj: decodable1},
			convertor:   &checkConvertor{in: decodable1, obj: decodable2, groupVersion: unversioned.GroupVersion{Group: "other", Version: "__internal"}},
			expectedGVK: gvk1,
			sameObject:  decodable2,
		},
		// defaultGVK.Group is allowed to force a conversion to the destination group
		{
			serializer:  &mockSerializer{actual: gvk1, obj: decodable1},
			defaultGVK:  &unversioned.GroupVersionKind{Group: "force"},
			convertor:   &checkConvertor{in: decodable1, obj: decodable2, groupVersion: unversioned.GroupVersion{Group: "force", Version: "__internal"}},
			expectedGVK: gvk1,
			sameObject:  decodable2,
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
			copier:         &checkCopy{in: decodable1, obj: decodable1},
			convertor:      &checkConvertor{in: decodable1, obj: decodable2, groupVersion: unversioned.GroupVersion{Group: "other", Version: "__internal"}},
			expectedGVK:    gvk1,
			expectedObject: &runtime.VersionedObjects{Objects: []runtime.Object{decodable1, decodable2}},
		},
		{
			into: &runtime.VersionedObjects{Objects: []runtime.Object{}},

			serializer:     &mockSerializer{actual: gvk1, obj: decodable1},
			copier:         &checkCopy{in: decodable1, obj: nil, err: fmt.Errorf("error on copy")},
			convertor:      &checkConvertor{in: decodable1, obj: decodable2, groupVersion: unversioned.GroupVersion{Group: "other", Version: "__internal"}},
			expectedGVK:    gvk1,
			expectedObject: &runtime.VersionedObjects{Objects: []runtime.Object{decodable1, decodable2}},
		},

		// decode into the same version as the serialized object
		{
			decodes: []unversioned.GroupVersion{gvk1.GroupVersion()},

			serializer:     &mockSerializer{actual: gvk1, obj: decodable1},
			expectedGVK:    gvk1,
			expectedObject: decodable1,
		},
		{
			into:    &runtime.VersionedObjects{Objects: []runtime.Object{}},
			decodes: []unversioned.GroupVersion{gvk1.GroupVersion()},

			serializer:     &mockSerializer{actual: gvk1, obj: decodable1},
			expectedGVK:    gvk1,
			expectedObject: &runtime.VersionedObjects{Objects: []runtime.Object{decodable1}},
		},

		// codec with non matching version skips conversion altogether
		{
			decodes: []unversioned.GroupVersion{{Group: "something", Version: "else"}},

			serializer:     &mockSerializer{actual: gvk1, obj: decodable1},
			expectedGVK:    gvk1,
			expectedObject: decodable1,
		},
		{
			into:    &runtime.VersionedObjects{Objects: []runtime.Object{}},
			decodes: []unversioned.GroupVersion{{Group: "something", Version: "else"}},

			serializer:     &mockSerializer{actual: gvk1, obj: decodable1},
			expectedGVK:    gvk1,
			expectedObject: &runtime.VersionedObjects{Objects: []runtime.Object{decodable1}},
		},
	}

	for i, test := range testCases {
		t.Logf("%d", i)
		s := NewCodec(test.serializer, test.serializer, test.convertor, test.creater, test.copier, test.typer, test.encodes, test.decodes)
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

type checkCopy struct {
	in, obj runtime.Object
	err     error
}

func (c *checkCopy) Copy(obj runtime.Object) (runtime.Object, error) {
	if c.in != nil && c.in != obj {
		return nil, fmt.Errorf("unexpected input to copy: %#v", obj)
	}
	return c.obj, c.err
}

type checkConvertor struct {
	err           error
	in, obj       runtime.Object
	groupVersion  unversioned.GroupVersion
	directConvert bool
}

func (c *checkConvertor) Convert(in, out interface{}) error {
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
func (c *checkConvertor) ConvertToVersion(in runtime.Object, outVersion unversioned.GroupVersion) (out runtime.Object, err error) {
	if c.directConvert {
		return nil, fmt.Errorf("unexpected call to ConvertToVersion")
	}
	if c.in != nil && c.in != in {
		return nil, fmt.Errorf("unexpected in: %s", in)
	}
	if c.groupVersion != outVersion {
		return nil, fmt.Errorf("unexpected outversion: %s", outVersion)
	}
	return c.obj, c.err
}
func (c *checkConvertor) ConvertFieldLabel(version, kind, label, value string) (string, string, error) {
	return "", "", fmt.Errorf("unexpected call to ConvertFieldLabel")
}

type mockSerializer struct {
	err error
	obj runtime.Object

	defaults, actual *unversioned.GroupVersionKind
	into             runtime.Object
}

func (s *mockSerializer) Decode(data []byte, defaults *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	s.defaults = defaults
	s.into = into
	return s.obj, s.actual, s.err
}

func (s *mockSerializer) Encode(obj runtime.Object, w io.Writer) error {
	s.obj = obj
	return s.err
}

type mockCreater struct {
	err error
	obj runtime.Object
}

func (c *mockCreater) New(kind unversioned.GroupVersionKind) (runtime.Object, error) {
	return c.obj, c.err
}

type mockTyper struct {
	gvk *unversioned.GroupVersionKind
	err error
}

func (t *mockTyper) ObjectKind(obj runtime.Object) (*unversioned.GroupVersionKind, bool, error) {
	return t.gvk, false, t.err
}
