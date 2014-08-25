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

package constraint

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func containerWithHostPorts(ports ...int) api.Container {
	c := api.Container{}
	for _, p := range ports {
		c.Ports = append(c.Ports, api.Port{HostPort: p})
	}
	return c
}

func manifestWithContainers(containers ...api.Container) api.ContainerManifest {
	m := api.ContainerManifest{}
	for _, c := range containers {
		m.Containers = append(m.Containers, c)
	}
	return m
}

func TestAllowed(t *testing.T) {
	table := []struct {
		allowed   bool
		manifests []api.ContainerManifest
	}{
		{
			allowed: true,
			manifests: []api.ContainerManifest{
				manifestWithContainers(
					containerWithHostPorts(1, 2, 3),
					containerWithHostPorts(4, 5, 6),
				),
				manifestWithContainers(
					containerWithHostPorts(7, 8, 9),
					containerWithHostPorts(10, 11, 12),
				),
			},
		},
		{
			allowed: true,
			manifests: []api.ContainerManifest{
				manifestWithContainers(
					containerWithHostPorts(0, 0),
					containerWithHostPorts(0, 0),
				),
				manifestWithContainers(
					containerWithHostPorts(0, 0),
					containerWithHostPorts(0, 0),
				),
			},
		},
		{
			allowed: false,
			manifests: []api.ContainerManifest{
				manifestWithContainers(
					containerWithHostPorts(3, 3),
				),
			},
		},
		{
			allowed: false,
			manifests: []api.ContainerManifest{
				manifestWithContainers(
					containerWithHostPorts(6),
				),
				manifestWithContainers(
					containerWithHostPorts(6),
				),
			},
		},
	}

	for _, item := range table {
		if e, a := item.allowed, Allowed(item.manifests); e != a {
			t.Errorf("Expected %v, got %v: \n%v\v", e, a, item.manifests)
		}
	}
}
