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

package container

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
)

func TestFieldPath(t *testing.T) {
	pod := &v1.Pod{Spec: v1.PodSpec{Containers: []v1.Container{
		{Name: "foo"},
		{Name: "bar"},
		{Name: ""},
		{Name: "baz"},
	}}}
	table := map[string]struct {
		pod       *v1.Pod
		container *v1.Container
		path      string
		success   bool
	}{
		"basic":            {pod, &v1.Container{Name: "foo"}, "spec.containers{foo}", true},
		"basic2":           {pod, &v1.Container{Name: "baz"}, "spec.containers{baz}", true},
		"emptyName":        {pod, &v1.Container{Name: ""}, "spec.containers[2]", true},
		"basicSamePointer": {pod, &pod.Spec.Containers[0], "spec.containers{foo}", true},
		"missing":          {pod, &v1.Container{Name: "qux"}, "", false},
	}

	for name, item := range table {
		res, err := fieldPath(item.pod, item.container)
		if item.success == false {
			if err == nil {
				t.Errorf("%v: unexpected non-error", name)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v: unexpected error: %v", name, err)
			continue
		}
		if e, a := item.path, res; e != a {
			t.Errorf("%v: wanted %v, got %v", name, e, a)
		}
	}
}

func TestGenerateContainerRef(t *testing.T) {
	var (
		okPod = v1.Pod{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Pod",
				APIVersion: api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String(),
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:            "ok",
				Namespace:       "test-ns",
				UID:             "bar",
				ResourceVersion: "42",
				SelfLink:        "/api/" + api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String() + "/pods/foo",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "by-name",
					},
					{},
				},
			},
		}
		noSelfLinkPod        = okPod
		defaultedSelfLinkPod = okPod
	)
	noSelfLinkPod.Kind = ""
	noSelfLinkPod.APIVersion = ""
	noSelfLinkPod.ObjectMeta.SelfLink = ""
	defaultedSelfLinkPod.ObjectMeta.SelfLink = "/api/" + api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String() + "/pods/ok"

	cases := []struct {
		name      string
		pod       *v1.Pod
		container *v1.Container
		expected  *v1.ObjectReference
		success   bool
	}{
		{
			name: "by-name",
			pod:  &okPod,
			container: &v1.Container{
				Name: "by-name",
			},
			expected: &v1.ObjectReference{
				Kind:            "Pod",
				APIVersion:      api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String(),
				Name:            "ok",
				Namespace:       "test-ns",
				UID:             "bar",
				ResourceVersion: "42",
				FieldPath:       ".spec.containers{by-name}",
			},
			success: true,
		},
		{
			name:      "no-name",
			pod:       &okPod,
			container: &v1.Container{},
			expected: &v1.ObjectReference{
				Kind:            "Pod",
				APIVersion:      api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String(),
				Name:            "ok",
				Namespace:       "test-ns",
				UID:             "bar",
				ResourceVersion: "42",
				FieldPath:       ".spec.containers[1]",
			},
			success: true,
		},
		{
			name:      "no-selflink",
			pod:       &noSelfLinkPod,
			container: &v1.Container{},
			expected:  nil,
			success:   false,
		},
		{
			name: "defaulted-selflink",
			pod:  &defaultedSelfLinkPod,
			container: &v1.Container{
				Name: "by-name",
			},
			expected: &v1.ObjectReference{
				Kind:            "Pod",
				APIVersion:      api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String(),
				Name:            "ok",
				Namespace:       "test-ns",
				UID:             "bar",
				ResourceVersion: "42",
				FieldPath:       ".spec.containers{by-name}",
			},
			success: true,
		},
		{
			name: "implicitly-required",
			pod:  &okPod,
			container: &v1.Container{
				Name: "net",
			},
			expected: &v1.ObjectReference{
				Kind:            "Pod",
				APIVersion:      api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String(),
				Name:            "ok",
				Namespace:       "test-ns",
				UID:             "bar",
				ResourceVersion: "42",
				FieldPath:       "implicitly required container net",
			},
			success: true,
		},
	}

	for _, tc := range cases {
		actual, err := GenerateContainerRef(tc.pod, tc.container)
		if err != nil {
			if tc.success {
				t.Errorf("%v: unexpected error: %v", tc.name, err)
			}

			continue
		}

		if !tc.success {
			t.Errorf("%v: unexpected success", tc.name)
			continue
		}

		if e, a := tc.expected.Kind, actual.Kind; e != a {
			t.Errorf("%v: kind: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.expected.APIVersion, actual.APIVersion; e != a {
			t.Errorf("%v: apiVersion: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.expected.Name, actual.Name; e != a {
			t.Errorf("%v: name: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.expected.Namespace, actual.Namespace; e != a {
			t.Errorf("%v: namespace: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.expected.UID, actual.UID; e != a {
			t.Errorf("%v: uid: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.expected.ResourceVersion, actual.ResourceVersion; e != a {
			t.Errorf("%v: kind: expected %v, got %v", tc.name, e, a)
		}
	}
}
