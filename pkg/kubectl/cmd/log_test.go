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
	"bytes"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestSelectContainer(t *testing.T) {
	tests := []struct {
		input             string
		pod               api.Pod
		expectedContainer string
	}{
		{
			input: "1\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
		{
			input: "foo\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
		{
			input: "foo\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "bar",
						},
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
		{
			input: "2\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "bar",
						},
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
		{
			input: "-1\n2\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "bar",
						},
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
		{
			input: "3\n2\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "bar",
						},
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
		{
			input: "baz\n2\n",
			pod: api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "bar",
						},
						{
							Name: "foo",
						},
					},
				},
			},
			expectedContainer: "foo",
		},
	}

	for _, test := range tests {
		var buff bytes.Buffer
		container := selectContainer(&test.pod, bytes.NewBufferString(test.input), &buff)
		if container != test.expectedContainer {
			t.Errorf("unexpected output: %s for input: %s", container, test.input)
		}
	}
}
