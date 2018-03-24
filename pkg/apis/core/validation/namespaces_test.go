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

package validation

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	_ "k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestValidateNamespace(t *testing.T) {
	validLabels := map[string]string{"a": "b"}
	invalidLabels := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	successCases := []core.Namespace{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "abc", Labels: validLabels},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "abc-123"},
			Spec: core.NamespaceSpec{
				Finalizers: []core.FinalizerName{"example.com/something", "example.com/other"},
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateNamespace(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]struct {
		R core.Namespace
		D string
	}{
		"zero-length name": {
			core.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ""}},
			"",
		},
		"defined-namespace": {
			core.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "abc-123", Namespace: "makesnosense"}},
			"",
		},
		"invalid-labels": {
			core.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "abc", Labels: invalidLabels}},
			"",
		},
	}
	for k, v := range errorCases {
		errs := ValidateNamespace(&v.R)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateNamespaceFinalizeUpdate(t *testing.T) {
	tests := []struct {
		oldNamespace core.Namespace
		namespace    core.Namespace
		valid        bool
	}{
		{core.Namespace{}, core.Namespace{}, true},
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo"}},
			core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo"},
				Spec: core.NamespaceSpec{
					Finalizers: []core.FinalizerName{"Foo"},
				},
			}, false},
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo"},
			Spec: core.NamespaceSpec{
				Finalizers: []core.FinalizerName{"foo.com/bar"},
			},
		},
			core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo"},
				Spec: core.NamespaceSpec{
					Finalizers: []core.FinalizerName{"foo.com/bar", "what.com/bar"},
				},
			}, true},
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: "fooemptyfinalizer"},
			Spec: core.NamespaceSpec{
				Finalizers: []core.FinalizerName{"foo.com/bar"},
			},
		},
			core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "fooemptyfinalizer"},
				Spec: core.NamespaceSpec{
					Finalizers: []core.FinalizerName{"", "foo.com/bar", "what.com/bar"},
				},
			}, false},
	}
	for i, test := range tests {
		test.namespace.ObjectMeta.ResourceVersion = "1"
		test.oldNamespace.ObjectMeta.ResourceVersion = "1"
		errs := ValidateNamespaceFinalizeUpdate(&test.namespace, &test.oldNamespace)
		if test.valid && len(errs) > 0 {
			t.Errorf("%d: Unexpected error: %v", i, errs)
			t.Logf("%#v vs %#v", test.oldNamespace, test.namespace)
		}
		if !test.valid && len(errs) == 0 {
			t.Errorf("%d: Unexpected non-error", i)
		}
	}
}

func TestValidateNamespaceStatusUpdate(t *testing.T) {
	now := metav1.Now()

	tests := []struct {
		oldNamespace core.Namespace
		namespace    core.Namespace
		valid        bool
	}{
		{core.Namespace{}, core.Namespace{
			Status: core.NamespaceStatus{
				Phase: core.NamespaceActive,
			},
		}, true},
		// Cannot set deletionTimestamp via status update
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo"}},
			core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "foo",
					DeletionTimestamp: &now},
				Status: core.NamespaceStatus{
					Phase: core.NamespaceTerminating,
				},
			}, false},
		// Can update phase via status update
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:              "foo",
				DeletionTimestamp: &now}},
			core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "foo",
					DeletionTimestamp: &now},
				Status: core.NamespaceStatus{
					Phase: core.NamespaceTerminating,
				},
			}, true},
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo"}},
			core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo"},
				Status: core.NamespaceStatus{
					Phase: core.NamespaceTerminating,
				},
			}, false},
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo"}},
			core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar"},
				Status: core.NamespaceStatus{
					Phase: core.NamespaceTerminating,
				},
			}, false},
	}
	for i, test := range tests {
		test.namespace.ObjectMeta.ResourceVersion = "1"
		test.oldNamespace.ObjectMeta.ResourceVersion = "1"
		errs := ValidateNamespaceStatusUpdate(&test.namespace, &test.oldNamespace)
		if test.valid && len(errs) > 0 {
			t.Errorf("%d: Unexpected error: %v", i, errs)
			t.Logf("%#v vs %#v", test.oldNamespace.ObjectMeta, test.namespace.ObjectMeta)
		}
		if !test.valid && len(errs) == 0 {
			t.Errorf("%d: Unexpected non-error", i)
		}
	}
}

func TestValidateNamespaceUpdate(t *testing.T) {
	tests := []struct {
		oldNamespace core.Namespace
		namespace    core.Namespace
		valid        bool
	}{
		{core.Namespace{}, core.Namespace{}, true},
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo1"}},
			core.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar1"},
			}, false},
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo2",
				Labels: map[string]string{"foo": "bar"},
			},
		}, core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo2",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo3",
			},
		}, core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo3",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo4",
				Labels: map[string]string{"bar": "foo"},
			},
		}, core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo4",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo5",
				Labels: map[string]string{"foo": "baz"},
			},
		}, core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo5",
				Labels: map[string]string{"Foo": "baz"},
			},
		}, true},
		{core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo6",
				Labels: map[string]string{"foo": "baz"},
			},
		}, core.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo6",
				Labels: map[string]string{"Foo": "baz"},
			},
			Spec: core.NamespaceSpec{
				Finalizers: []core.FinalizerName{"kubernetes"},
			},
			Status: core.NamespaceStatus{
				Phase: core.NamespaceTerminating,
			},
		}, true},
	}
	for i, test := range tests {
		test.namespace.ObjectMeta.ResourceVersion = "1"
		test.oldNamespace.ObjectMeta.ResourceVersion = "1"
		errs := ValidateNamespaceUpdate(&test.namespace, &test.oldNamespace)
		if test.valid && len(errs) > 0 {
			t.Errorf("%d: Unexpected error: %v", i, errs)
			t.Logf("%#v vs %#v", test.oldNamespace.ObjectMeta, test.namespace.ObjectMeta)
		}
		if !test.valid && len(errs) == 0 {
			t.Errorf("%d: Unexpected non-error", i)
		}
	}
}
