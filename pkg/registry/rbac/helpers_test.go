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

package rbac

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	kapi "k8s.io/kubernetes/pkg/apis/core"
	kapihelper "k8s.io/kubernetes/pkg/apis/core/helper"

	fuzz "github.com/google/gofuzz"
)

func newPod() *kapi.Pod {
	return &kapi.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Annotations:     map[string]string{},
			Name:            "foo",
			OwnerReferences: []metav1.OwnerReference{},
		},
	}

}

func TestIsOnlyMutatingGCFields(t *testing.T) {
	tests := []struct {
		name     string
		obj      func() runtime.Object
		old      func() runtime.Object
		expected bool
	}{
		{
			name: "same",
			obj: func() runtime.Object {
				return newPod()
			},
			old: func() runtime.Object {
				return newPod()
			},
			expected: true,
		},
		{
			name: "different managedFields",
			obj: func() runtime.Object {
				return newPod()
			},
			old: func() runtime.Object {
				obj := newPod()
				obj.ManagedFields = []metav1.ManagedFieldsEntry{
					{
						Manager: "manager",
					},
				}
				return obj
			},
			expected: true,
		},
		{
			name: "only annotations",
			obj: func() runtime.Object {
				obj := newPod()
				obj.Annotations["foo"] = "bar"
				return obj
			},
			old: func() runtime.Object {
				return newPod()
			},
			expected: false,
		},
		{
			name: "only other",
			obj: func() runtime.Object {
				obj := newPod()
				obj.Spec.RestartPolicy = kapi.RestartPolicyAlways
				return obj
			},
			old: func() runtime.Object {
				return newPod()
			},
			expected: false,
		},
		{
			name: "only ownerRef",
			obj: func() runtime.Object {
				obj := newPod()
				obj.OwnerReferences = append(obj.OwnerReferences, metav1.OwnerReference{Name: "foo"})
				return obj
			},
			old: func() runtime.Object {
				return newPod()
			},
			expected: true,
		},
		{
			name: "ownerRef and finalizer",
			obj: func() runtime.Object {
				obj := newPod()
				obj.OwnerReferences = append(obj.OwnerReferences, metav1.OwnerReference{Name: "foo"})
				obj.Finalizers = []string{"final"}
				return obj
			},
			old: func() runtime.Object {
				return newPod()
			},
			expected: true,
		},
		{
			name: "and annotations",
			obj: func() runtime.Object {
				obj := newPod()
				obj.OwnerReferences = append(obj.OwnerReferences, metav1.OwnerReference{Name: "foo"})
				obj.Annotations["foo"] = "bar"
				return obj
			},
			old: func() runtime.Object {
				return newPod()
			},
			expected: false,
		},
		{
			name: "and other",
			obj: func() runtime.Object {
				obj := newPod()
				obj.OwnerReferences = append(obj.OwnerReferences, metav1.OwnerReference{Name: "foo"})
				obj.Spec.RestartPolicy = kapi.RestartPolicyAlways
				return obj
			},
			old: func() runtime.Object {
				return newPod()
			},
			expected: false,
		},
		{
			name: "and nil",
			obj: func() runtime.Object {
				obj := newPod()
				obj.OwnerReferences = append(obj.OwnerReferences, metav1.OwnerReference{Name: "foo"})
				obj.Spec.RestartPolicy = kapi.RestartPolicyAlways
				return obj
			},
			old: func() runtime.Object {
				return (*kapi.Pod)(nil)
			},
			expected: false,
		},
	}

	for _, tc := range tests {
		actual := IsOnlyMutatingGCFields(tc.obj(), tc.old(), kapihelper.Semantic)
		if tc.expected != actual {
			t.Errorf("%s: expected %v, got %v", tc.name, tc.expected, actual)
		}
	}
}

func TestNewMetadataFields(t *testing.T) {
	f := fuzz.New().NilChance(0.0).NumElements(1, 1)
	for i := 0; i < 100; i++ {
		objMeta := metav1.ObjectMeta{}
		f.Fuzz(&objMeta)
		objMeta.Name = ""
		objMeta.GenerateName = ""
		objMeta.Namespace = ""
		objMeta.SelfLink = ""
		objMeta.UID = types.UID("")
		objMeta.ResourceVersion = ""
		objMeta.Generation = 0
		objMeta.CreationTimestamp = metav1.Time{}
		objMeta.DeletionTimestamp = nil
		objMeta.DeletionGracePeriodSeconds = nil
		objMeta.Labels = nil
		objMeta.Annotations = nil
		objMeta.OwnerReferences = nil
		objMeta.Finalizers = nil
		objMeta.ManagedFields = nil

		if !reflect.DeepEqual(metav1.ObjectMeta{}, objMeta) {
			t.Fatalf(`A new field was introduced in ObjectMeta, add the field to
IsOnlyMutatingGCFields if necessary, and update this test:
%#v`, objMeta)
		}
	}
}
