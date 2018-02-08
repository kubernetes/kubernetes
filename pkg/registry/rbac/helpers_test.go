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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kapi "k8s.io/kubernetes/pkg/apis/core"
	kapihelper "k8s.io/kubernetes/pkg/apis/core/helper"
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
