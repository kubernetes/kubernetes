/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
)

func TestFieldPath(t *testing.T) {
	pod := &api.Pod{Spec: api.PodSpec{Containers: []api.Container{
		{Name: "foo"},
		{Name: "bar"},
		{Name: ""},
		{Name: "baz"},
	}}}
	table := map[string]struct {
		pod       *api.Pod
		container *api.Container
		path      string
		success   bool
	}{
		"basic":            {pod, &api.Container{Name: "foo"}, "spec.containers{foo}", true},
		"basic2":           {pod, &api.Container{Name: "baz"}, "spec.containers{baz}", true},
		"emptyName":        {pod, &api.Container{Name: ""}, "spec.containers[2]", true},
		"basicSamePointer": {pod, &pod.Spec.Containers[0], "spec.containers{foo}", true},
		"missing":          {pod, &api.Container{Name: "qux"}, "", false},
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
		okPod = api.Pod{
			TypeMeta: api.TypeMeta{
				Kind:       "Pod",
				APIVersion: testapi.Version(),
			},
			ObjectMeta: api.ObjectMeta{
				Name:            "ok",
				Namespace:       "test-ns",
				UID:             "bar",
				ResourceVersion: "42",
				SelfLink:        "/api/" + testapi.Version() + "/pods/foo",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
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
	defaultedSelfLinkPod.ObjectMeta.SelfLink = "/api/" + testapi.Version() + "/pods/ok"

	cases := []struct {
		name      string
		pod       *api.Pod
		container *api.Container
		expected  *api.ObjectReference
		success   bool
	}{
		{
			name: "by-name",
			pod:  &okPod,
			container: &api.Container{
				Name: "by-name",
			},
			expected: &api.ObjectReference{
				Kind:            "Pod",
				APIVersion:      testapi.Version(),
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
			container: &api.Container{},
			expected: &api.ObjectReference{
				Kind:            "Pod",
				APIVersion:      testapi.Version(),
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
			container: &api.Container{},
			expected:  nil,
			success:   false,
		},
		{
			name: "defaulted-selflink",
			pod:  &defaultedSelfLinkPod,
			container: &api.Container{
				Name: "by-name",
			},
			expected: &api.ObjectReference{
				Kind:            "Pod",
				APIVersion:      testapi.Version(),
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
			container: &api.Container{
				Name: "net",
			},
			expected: &api.ObjectReference{
				Kind:            "Pod",
				APIVersion:      testapi.Version(),
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
