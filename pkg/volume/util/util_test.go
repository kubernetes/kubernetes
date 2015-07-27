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

package util

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestContainerHasVolumeMountForName(t *testing.T) {
	cases := []struct {
		name       string
		input      *api.Container
		volumeName string
		expected   bool
	}{
		{
			name: "positive",
			input: &api.Container{
				VolumeMounts: []api.VolumeMount{
					{
						Name: "volName",
					},
				},
			},
			volumeName: "volName",
			expected:   true,
		},
		{
			name: "negative",
			input: &api.Container{
				VolumeMounts: []api.VolumeMount{
					{
						Name: "volName2",
					},
				},
			},
			volumeName: "volName",
			expected:   false,
		},
	}

	for _, tc := range cases {
		actual := ContainerHasVolumeMountForName(tc.input, tc.volumeName)

		if e, a := tc.expected, actual; e != a {
			t.Errorf("%v: expected: %v, got: %v", tc.name, e, a)
		}
	}
}
