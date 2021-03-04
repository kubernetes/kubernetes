// +build !dockerless

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
	"github.com/stretchr/testify/assert"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

func TestContainerStats(t *testing.T) {
	tests := map[string]struct {
		containerID    string
		container      *libdocker.FakeContainer
		containerStats *dockertypes.StatsJSON
		calledDetails  []libdocker.CalledDetail
	}{
		"container exists": {
			"fake_container",
			&libdocker.FakeContainer{ID: "fake_container"},
			&dockertypes.StatsJSON{},
			[]libdocker.CalledDetail{
				libdocker.NewCalledDetail("get_container_stats", nil),
				libdocker.NewCalledDetail("inspect_container_withsize", nil),
				libdocker.NewCalledDetail("inspect_container", nil),
				libdocker.NewCalledDetail("inspect_image", nil),
			},
		},
		"container doesn't exists": {
			"nonexistant_fake_container",
			&libdocker.FakeContainer{ID: "fake_container"},
			&dockertypes.StatsJSON{},
			[]libdocker.CalledDetail{
				libdocker.NewCalledDetail("get_container_stats", nil),
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
