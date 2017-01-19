package internalversion

import (
	"net/url"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
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
	out := &ListOptions{}
	if err := scheme.Convert(in, out, nil); err != nil {
		t.Fatal(err)
	}
	actual := &metav1.ListOptions{}
	if err := scheme.Convert(out, actual, nil); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, actual) {
		t.Errorf("unexpected: %s", diff.ObjectReflectDiff(in, actual))
	}

	// verify failing conversion
	for i, failingObject := range []*metav1.ListOptions{
		&metav1.ListOptions{LabelSelector: "a!!!"},
		&metav1.ListOptions{FieldSelector: "a!!!"},
	} {
		out = &ListOptions{}
		if err := scheme.Convert(failingObject, out, nil); err == nil {
			t.Errorf("%d: unexpected conversion: %#v", i, out)
		}
	}

	// verify kind registration
	if gvk, unversioned, err := scheme.ObjectKind(in); err != nil || unversioned || gvk != metav1.SchemeGroupVersion.WithKind("ListOptions") {
		t.Errorf("unexpected: %v %v %v", gvk, unversioned, err)
	}
	if gvk, unversioned, err := scheme.ObjectKind(out); err != nil || unversioned || gvk != SchemeGroupVersion.WithKind("ListOptions") {
		t.Errorf("unexpected: %v %v %v", gvk, unversioned, err)
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
		t.Errorf("unexpected: %s", diff.ObjectReflectDiff(in, actual))
	}
}
