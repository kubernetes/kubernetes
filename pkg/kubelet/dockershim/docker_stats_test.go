//go:build linux && !dockerless
// +build linux,!dockerless

/*
Copyright 2019 The Kubernetes Authors.

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

package dockershim

import (
	"testing"

	dockertypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/stretchr/testify/assert"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

func TestContainerStats(t *testing.T) {
	labels := map[string]string{containerTypeLabelKey: containerTypeLabelContainer}
	tests := map[string]struct {
		containerID    string
		container      *libdocker.FakeContainer
		containerStats *dockertypes.StatsJSON
		calledDetails  []libdocker.CalledDetail
	}{
		"container exists": {
			"k8s_fake_container",
			&libdocker.FakeContainer{
				ID:   "k8s_fake_container",
				Name: "k8s_fake_container_1_2_1",
				Config: &container.Config{
					Labels: labels,
				},
			},
			&dockertypes.StatsJSON{},
			[]libdocker.CalledDetail{
				libdocker.NewCalledDetail("list", nil),
				libdocker.NewCalledDetail("get_container_stats", nil),
				libdocker.NewCalledDetail("inspect_container_withsize", nil),
			},
		},
		"container doesn't exists": {
			"k8s_nonexistant_fake_container",
			&libdocker.FakeContainer{
				ID:   "k8s_fake_container",
				Name: "k8s_fake_container_1_2_1",
				Config: &container.Config{
					Labels: labels,
				},
			},
			&dockertypes.StatsJSON{},
			[]libdocker.CalledDetail{
				libdocker.NewCalledDetail("list", nil),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			ds, fakeDocker, _ := newTestDockerService()
			fakeDocker.SetFakeContainers([]*libdocker.FakeContainer{test.container})
			fakeDocker.InjectContainerStats(map[string]*dockertypes.StatsJSON{test.container.ID: test.containerStats})
			ds.ContainerStats(getTestCTX(), &runtimeapi.ContainerStatsRequest{ContainerId: test.containerID})
			err := fakeDocker.AssertCallDetails(test.calledDetails...)
			assert.NoError(t, err)
		})
	}
}
