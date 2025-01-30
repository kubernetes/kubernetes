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

package testing

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

var _ metav1.Object = &metav1.ObjectMeta{}

func TestAccessorImplementations(t *testing.T) {
	for _, gv := range legacyscheme.Scheme.PrioritizedVersionsAllGroups() {
		internalGV := schema.GroupVersion{Group: gv.Group, Version: runtime.APIVersionInternal}
		for _, gv := range []schema.GroupVersion{gv, internalGV} {
			for kind, knownType := range legacyscheme.Scheme.KnownTypes(gv) {
				value := reflect.New(knownType)
				obj := value.Interface()
				if _, ok := obj.(runtime.Object); !ok {
					t.Errorf("%v (%v) does not implement runtime.Object", gv.WithKind(kind), knownType)
				}
				lm, isLM := obj.(meta.ListMetaAccessor)
				om, isOM := obj.(metav1.ObjectMetaAccessor)
				switch {
				case isLM && isOM:
					t.Errorf("%v (%v) implements ListMetaAccessor and ObjectMetaAccessor", gv.WithKind(kind), knownType)
					continue
				case isLM:
					m := lm.GetListMeta()
					if m == nil {
						t.Errorf("%v (%v) returns nil ListMeta", gv.WithKind(kind), knownType)
						continue
					}
					m.SetResourceVersion("102030")
					if m.GetResourceVersion() != "102030" {
						t.Errorf("%v (%v) did not preserve resource version", gv.WithKind(kind), knownType)
						continue
					}
				case isOM:
					m := om.GetObjectMeta()
					if m == nil {
						t.Errorf("%v (%v) returns nil ObjectMeta", gv.WithKind(kind), knownType)
						continue
					}
					m.SetResourceVersion("102030")
					if m.GetResourceVersion() != "102030" {
						t.Errorf("%v (%v) did not preserve resource version", gv.WithKind(kind), knownType)
						continue
					}
					labels := map[string]string{"a": "b"}
					m.SetLabels(labels)
					if !reflect.DeepEqual(m.GetLabels(), labels) {
						t.Errorf("%v (%v) did not preserve labels", gv.WithKind(kind), knownType)
						continue
					}
				default:
					if _, ok := obj.(metav1.ListMetaAccessor); ok {
						continue
					}
					if _, ok := value.Elem().Type().FieldByName("ObjectMeta"); ok {
						t.Errorf("%v (%v) has ObjectMeta but does not implement ObjectMetaAccessor", gv.WithKind(kind), knownType)
						continue
					}
					if _, ok := value.Elem().Type().FieldByName("ListMeta"); ok {
						t.Errorf("%v (%v) has ListMeta but does not implement ListMetaAccessor", gv.WithKind(kind), knownType)
						continue
					}
					t.Logf("%v (%v) does not implement ListMetaAccessor or ObjectMetaAccessor", gv.WithKind(kind), knownType)
				}
			}
		}
	}
}
