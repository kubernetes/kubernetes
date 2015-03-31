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

package dockertools

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/fsouza/go-dockerclient"
)

func TestSetEntrypointAndCommand(t *testing.T) {
	cases := []struct {
		name      string
		container *api.Container
		expected  *docker.CreateContainerOptions
	}{
		{
			name:      "none",
			container: &api.Container{},
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{},
			},
		},
		{
			name: "command",
			container: &api.Container{
				Command: []string{"foo", "bar"},
			},
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{
					Entrypoint: []string{"foo", "bar"},
				},
			},
		},
		{
			name: "args",
			container: &api.Container{
				Args: []string{"foo", "bar"},
			},
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{
					Cmd: []string{"foo", "bar"},
				},
			},
		},
		{
			name: "both",
			container: &api.Container{
				Command: []string{"foo"},
				Args:    []string{"bar", "baz"},
			},
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{
					Entrypoint: []string{"foo"},
					Cmd:        []string{"bar", "baz"},
				},
			},
		},
	}

	for _, tc := range cases {
		actualOpts := &docker.CreateContainerOptions{
			Config: &docker.Config{},
		}
		setEntrypointAndCommand(tc.container, actualOpts)

		if e, a := tc.expected.Config.Entrypoint, actualOpts.Config.Entrypoint; !api.Semantic.DeepEqual(e, a) {
			t.Errorf("%v: unexpected entrypoint: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.expected.Config.Cmd, actualOpts.Config.Cmd; !api.Semantic.DeepEqual(e, a) {
			t.Errorf("%v: unexpected command: expected %v, got %v", tc.name, e, a)
		}
	}
}
