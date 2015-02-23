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
	"fmt"
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

func podWithContainers(containers ...api.Container) api.BoundPod {
	m := api.BoundPod{}
	for _, c := range containers {
		m.Spec.Containers = append(m.Spec.Containers, c)
	}
	return m
}

func TestAllowed(t *testing.T) {
	table := []struct {
		err  string
		pods []api.BoundPod
	}{
		{
			err: "[]",
			pods: []api.BoundPod{
				podWithContainers(
					containerWithHostPorts(1, 2, 3),
					containerWithHostPorts(4, 5, 6),
				),
				podWithContainers(
					containerWithHostPorts(7, 8, 9),
					containerWithHostPorts(10, 11, 12),
				),
			},
		},
		{
			err: "[]",
			pods: []api.BoundPod{
				podWithContainers(
					containerWithHostPorts(0, 0),
					containerWithHostPorts(0, 0),
				),
				podWithContainers(
					containerWithHostPorts(0, 0),
					containerWithHostPorts(0, 0),
				),
			},
		},
		{
			err: "[host port 3 is already in use]",
			pods: []api.BoundPod{
				podWithContainers(
					containerWithHostPorts(3, 3),
				),
			},
		},
		{
			err: "[host port 6 is already in use]",
			pods: []api.BoundPod{
				podWithContainers(
					containerWithHostPorts(6),
				),
				podWithContainers(
					containerWithHostPorts(6),
				),
			},
		},
	}

	for _, item := range table {
		if e, a := item.err, Allowed(item.pods); e != fmt.Sprintf("%v", a) {
			t.Errorf("Expected %v, got %v: \n%v\v", e, a, item.pods)
		}
	}
}
