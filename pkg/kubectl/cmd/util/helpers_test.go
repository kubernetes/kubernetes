/*
Copyright 2014 Google Inc. All rights reserved.

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

package util

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func TestMerge(t *testing.T) {
	tests := []struct {
		obj       runtime.Object
		fragment  string
		expected  runtime.Object
		expectErr bool
		kind      string
	}{
		{
			kind: "Pod",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: `{ "apiVersion": "v1beta1" }`,
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicy{
						Always: &api.RestartPolicyAlways{},
					},
					DNSPolicy: api.DNSClusterFirst,
				},
			},
		},
		{
			kind: "Pod",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: `{ "apiVersion": "v1beta1", "id": "baz", "desiredState": { "host": "bar" } }`,
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "baz",
				},
				Spec: api.PodSpec{
					Host: "bar",
					RestartPolicy: api.RestartPolicy{
						Always: &api.RestartPolicyAlways{},
					},
					DNSPolicy: api.DNSClusterFirst,
				},
			},
		},
		{
			kind: "Pod",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: `{ "apiVersion": "v1beta3", "spec": { "volumes": [ {"name": "v1"}, {"name": "v2"} ] } }`,
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PodSpec{
					Volumes: []api.Volume{
						{
							Name:         "v1",
							VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}},
						},
						{
							Name:         "v2",
							VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}},
						},
					},
					RestartPolicy: api.RestartPolicy{
						Always: &api.RestartPolicyAlways{},
					},
					DNSPolicy: api.DNSClusterFirst,
				},
			},
		},
		{
			kind:      "Pod",
			obj:       &api.Pod{},
			fragment:  "invalid json",
			expected:  &api.Pod{},
			expectErr: true,
		},
		{
			kind: "Pod",
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: `{ "apiVersion": "v1beta1", "id": null}`,
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "",
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicy{
						Always: &api.RestartPolicyAlways{},
					},
					DNSPolicy: api.DNSClusterFirst,
				},
			},
		},
		{
			kind:      "Service",
			obj:       &api.Service{},
			fragment:  `{ "apiVersion": "badVersion" }`,
			expectErr: true,
		},
		{
			kind: "Service",
			obj: &api.Service{
				Spec: api.ServiceSpec{
					Port: 10,
				},
			},
			fragment: `{ "apiVersion": "v1beta1", "port": 0 }`,
			expected: &api.Service{
				Spec: api.ServiceSpec{
					Port:            0,
					Protocol:        "TCP",
					SessionAffinity: "None",
				},
			},
		},
		{
			kind: "Service",
			obj: &api.Service{
				Spec: api.ServiceSpec{
					Selector: map[string]string{
						"version": "v1",
					},
				},
			},
			fragment: `{ "apiVersion": "v1beta1", "selector": { "version": "v2" } }`,
			expected: &api.Service{
				Spec: api.ServiceSpec{
					Protocol:        "TCP",
					SessionAffinity: "None",
					Selector: map[string]string{
						"version": "v2",
					},
				},
			},
		},
	}

	for i, test := range tests {
		out, err := Merge(test.obj, test.fragment, test.kind)
		if !test.expectErr {
			if err != nil {
				t.Errorf("testcase[%d], unexpected error: %v", i, err)
			} else if !reflect.DeepEqual(out, test.expected) {
				t.Errorf("\n\ntestcase[%d]\nexpected:\n%+v\nsaw:\n%+v", i, test.expected, out)
			}
		}
		if test.expectErr && err == nil {
			t.Errorf("testcase[%d], unexpected non-error", i)
		}
	}

}
