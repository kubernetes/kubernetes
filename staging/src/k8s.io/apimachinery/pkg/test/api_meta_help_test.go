/*
Copyright 2014 The Kubernetes Authors.

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

package test

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	fuzz "github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/api/meta"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/testapigroup"
	v1 "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

func TestIsList(t *testing.T) {
	tests := []struct {
		obj    runtime.Object
		isList bool
	}{
		{&testapigroup.CarpList{}, true},
		{&testapigroup.Carp{}, false},
	}
	for _, item := range tests {
		if e, a := item.isList, meta.IsListType(item.obj); e != a {
			t.Errorf("%v: Expected %v, got %v", reflect.TypeOf(item.obj), e, a)
		}
	}
}

func TestExtractList(t *testing.T) {
	list1 := []runtime.Object{
		&testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "1"}},
		&testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
	}
	list2 := &ListV1{
		Items: []runtime.RawExtension{
			{Raw: []byte("foo")},
			{Raw: []byte("bar")},
			{Object: &v1.Carp{ObjectMeta: metav1.ObjectMeta{Name: "other"}}},
		},
	}
	list3 := &fakePtrValueList{
		Items: []*testapigroup.Carp{
			{ObjectMeta: metav1.ObjectMeta{Name: "1"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
		},
	}
	list4 := &testapigroup.CarpList{
		Items: []testapigroup.Carp{
			{ObjectMeta: metav1.ObjectMeta{Name: "1"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "3"}},
		},
	}

	testCases := []struct {
		in    runtime.Object
		out   []interface{}
		equal bool
	}{
		{
			in:  &List{},
			out: []interface{}{},
		},
		{
			in:  &ListV1{},
			out: []interface{}{},
		},
		{
			in:  &List{Items: list1},
			out: []interface{}{list1[0], list1[1]},
		},
		{
			in:    list2,
			out:   []interface{}{&runtime.Unknown{Raw: list2.Items[0].Raw}, &runtime.Unknown{Raw: list2.Items[1].Raw}, list2.Items[2].Object},
			equal: true,
		},
		{
			in:  list3,
			out: []interface{}{list3.Items[0], list3.Items[1]},
		},
		{
			in:  list4,
			out: []interface{}{&list4.Items[0], &list4.Items[1], &list4.Items[2]},
		},
	}
	for i, test := range testCases {
		list, err := meta.ExtractList(test.in)
		if err != nil {
			t.Fatalf("%d: extract: Unexpected error %v", i, err)
		}
		if e, a := len(test.out), len(list); e != a {
			t.Fatalf("%d: extract: Expected %v, got %v", i, e, a)
		}
		for j, e := range test.out {
			if e != list[j] {
				if !test.equal {
					t.Fatalf("%d: extract: Expected list[%d] to be %#v, but found %#v", i, j, e, list[j])
				}
				if !reflect.DeepEqual(e, list[j]) {
					t.Fatalf("%d: extract: Expected list[%d] to be %#v, but found %#v", i, j, e, list[j])
				}
			}
		}
	}
}

func TestEachListItem(t *testing.T) {
	list1 := []runtime.Object{
		&testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "1"}},
		&testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
	}
	list2 := &ListV1{
		Items: []runtime.RawExtension{
			{Raw: []byte("foo")},
			{Raw: []byte("bar")},
			{Object: &v1.Carp{ObjectMeta: metav1.ObjectMeta{Name: "other"}}},
		},
	}
	list3 := &fakePtrValueList{
		Items: []*testapigroup.Carp{
			{ObjectMeta: metav1.ObjectMeta{Name: "1"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
		},
	}
	list4 := &testapigroup.CarpList{
		Items: []testapigroup.Carp{
			{ObjectMeta: metav1.ObjectMeta{Name: "1"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "3"}},
		},
	}

	testCases := []struct {
		in  runtime.Object
		out []interface{}
	}{
		{
			in:  &List{},
			out: []interface{}{},
		},
		{
			in:  &ListV1{},
			out: []interface{}{},
		},
		{
			in:  &List{Items: list1},
			out: []interface{}{list1[0], list1[1]},
		},
		{
			in:  list2,
			out: []interface{}{nil, nil, list2.Items[2].Object},
		},
		{
			in:  list3,
			out: []interface{}{list3.Items[0], list3.Items[1]},
		},
		{
			in:  list4,
			out: []interface{}{&list4.Items[0], &list4.Items[1], &list4.Items[2]},
		},
	}
	for i, test := range testCases {
		list := []runtime.Object{}
		err := meta.EachListItem(test.in, func(obj runtime.Object) error {
			list = append(list, obj)
			return nil
		})
		if err != nil {
			t.Fatalf("%d: each: Unexpected error %v", i, err)
		}
		if e, a := len(test.out), len(list); e != a {
			t.Fatalf("%d: each: Expected %v, got %v", i, e, a)
		}
		for j, e := range test.out {
			if e != list[j] {
				t.Fatalf("%d: each: Expected list[%d] to be %#v, but found %#v", i, j, e, list[j])
			}
		}
	}
}

type fakePtrInterfaceList struct {
	Items *[]runtime.Object
}

func (obj fakePtrInterfaceList) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
}
func (obj fakePtrInterfaceList) DeepCopyObject() runtime.Object {
	panic("fakePtrInterfaceList does not support DeepCopy")
}

func TestExtractListOfInterfacePtrs(t *testing.T) {
	pl := &fakePtrInterfaceList{
		Items: &[]runtime.Object{},
	}
	list, err := meta.ExtractList(pl)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if len(list) > 0 {
		t.Fatalf("Expected empty list, got %#v", list)
	}
}

type fakePtrValueList struct {
	Items []*testapigroup.Carp
}

func (obj fakePtrValueList) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
}
func (obj *fakePtrValueList) DeepCopyObject() runtime.Object {
	if obj == nil {
		return nil
	}
	clone := fakePtrValueList{
		Items: make([]*testapigroup.Carp, len(obj.Items)),
	}
	for i, carp := range obj.Items {
		clone.Items[i] = carp.DeepCopy()
	}
	return &clone
}

func TestSetList(t *testing.T) {
	pl := &testapigroup.CarpList{}
	list := []runtime.Object{
		&testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "1"}},
		&testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
		&testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "3"}},
	}
	err := meta.SetList(pl, list)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := len(list), len(pl.Items); e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	for i := range list {
		if e, a := list[i].(*testapigroup.Carp).Name, pl.Items[i].Name; e != a {
			t.Fatalf("Expected %v, got %v", e, a)
		}
	}
}

func TestSetListToRuntimeObjectArray(t *testing.T) {
	pl := &List{}
	list := []runtime.Object{
		&testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "1"}},
		&testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
		&testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "3"}},
	}
	err := meta.SetList(pl, list)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := len(list), len(pl.Items); e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	for i := range list {
		if e, a := list[i], pl.Items[i]; e != a {
			t.Fatalf("%d: unmatched: %s", i, cmp.Diff(e, a))
		}
	}
}

func TestSetListToMatchingType(t *testing.T) {
	pl := &unstructured.UnstructuredList{}
	list := []runtime.Object{
		&unstructured.Unstructured{Object: map[string]interface{}{"foo": 1}},
		&unstructured.Unstructured{Object: map[string]interface{}{"foo": 2}},
		&unstructured.Unstructured{Object: map[string]interface{}{"foo": 3}},
	}
	err := meta.SetList(pl, list)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := len(list), len(pl.Items); e != a {
		t.Fatalf("Expected %v, got %v", e, a)
	}
	for i := range list {
		if e, a := list[i], &pl.Items[i]; !reflect.DeepEqual(e, a) {
			t.Fatalf("%d: unmatched: %s", i, cmp.Diff(e, a))
		}
	}
}

func TestSetExtractListRoundTrip(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	fuzzer := fuzz.New().NilChance(0).NumElements(1, 5).Funcs(metafuzzer.Funcs(codecs)...).MaxDepth(10)
	for i := 0; i < 5; i++ {
		start := &testapigroup.CarpList{}
		fuzzer.Fuzz(&start.Items)

		list, err := meta.ExtractList(start)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
			continue
		}
		got := &testapigroup.CarpList{}
		err = meta.SetList(got, list)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
			continue
		}
		if e, a := start, got; !reflect.DeepEqual(e, a) {
			t.Fatalf("Expected %#v, got %#v", e, a)
		}
	}
}
