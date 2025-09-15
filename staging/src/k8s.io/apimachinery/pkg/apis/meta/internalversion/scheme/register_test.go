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

package scheme

import (
	"net/url"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestListOptions(t *testing.T) {
	// verify round trip conversion
	ten := int64(10)
	in := &metav1.ListOptions{
		LabelSelector:   "a=1",
		FieldSelector:   "b=1",
		ResourceVersion: "10",
		TimeoutSeconds:  &ten,
		Watch:           true,
	}
	out := &metainternalversion.ListOptions{}
	if err := Scheme.Convert(in, out, nil); err != nil {
		t.Fatal(err)
	}
	actual := &metav1.ListOptions{}
	if err := Scheme.Convert(out, actual, nil); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, actual) {
		t.Errorf("unexpected: %s", cmp.Diff(in, actual))
	}

	// verify failing conversion
	for i, failingObject := range []*metav1.ListOptions{
		{LabelSelector: "a!!!"},
		{FieldSelector: "a!!!"},
	} {
		out = &metainternalversion.ListOptions{}
		if err := Scheme.Convert(failingObject, out, nil); err == nil {
			t.Errorf("%d: unexpected conversion: %#v", i, out)
		}
	}

	// verify kind registration
	if gvks, unversioned, err := Scheme.ObjectKinds(in); err != nil || unversioned || gvks[0] != metav1.SchemeGroupVersion.WithKind("ListOptions") {
		t.Errorf("unexpected: %v %v %v", gvks[0], unversioned, err)
	}
	if gvks, unversioned, err := Scheme.ObjectKinds(out); err != nil || unversioned || gvks[0] != metainternalversion.SchemeGroupVersion.WithKind("ListOptions") {
		t.Errorf("unexpected: %v %v %v", gvks[0], unversioned, err)
	}

	actual = &metav1.ListOptions{}
	if err := ParameterCodec.DecodeParameters(url.Values{"watch": []string{"1"}}, metav1.SchemeGroupVersion, actual); err != nil {
		t.Fatal(err)
	}
	if !actual.Watch {
		t.Errorf("unexpected watch decode: %#v", actual)
	}

	// check ParameterCodec
	query, err := ParameterCodec.EncodeParameters(in, metav1.SchemeGroupVersion)
	if err != nil {
		t.Fatal(err)
	}
	actual = &metav1.ListOptions{}
	if err := ParameterCodec.DecodeParameters(query, metav1.SchemeGroupVersion, actual); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, actual) {
		t.Errorf("unexpected: %s", cmp.Diff(in, actual))
	}
}
