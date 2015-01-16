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

package cmd

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
	}{
		{
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: "{ \"apiVersion\": \"v1beta1\" }",
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
		},
		{
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: "{ \"apiVersion\": \"v1beta1\", \"id\": \"baz\", \"desiredState\": { \"host\": \"bar\" } }",
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "baz",
				},
				Spec: api.PodSpec{
					Host: "bar",
				},
			},
		},
		{
			obj: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fragment: "{ \"apiVersion\": \"v1beta3\", \"spec\": { \"volumes\": [ {\"name\": \"v1\"}, {\"name\": \"v2\"} ] } }",
			expected: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				Spec: api.PodSpec{
					Volumes: []api.Volume{
						{
							Name: "v1",
						},
						{
							Name: "v2",
						},
					},
				},
			},
		},
		{
			obj:       &api.Pod{},
			fragment:  "invalid json",
			expected:  &api.Pod{},
			expectErr: true,
		},
	}

	for _, test := range tests {
		err := Merge(test.obj, test.fragment, "Pod")
		if !test.expectErr {
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			} else if !reflect.DeepEqual(test.obj, test.expected) {
				t.Errorf("\nexpected:\n%v\nsaw:\n%v", test.expected, test.obj)
			}
		}
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error")
		}
	}

}
