/*
Copyright 2016 The Kubernetes Authors.

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
	"github.com/golang/mock/gomock"
	"testing"
	"time"

	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/mock_network"
	"k8s.io/kubernetes/pkg/util/clock"
)

// newTestNetworkPlugin returns a mock plugin that implements network.NetworkPlugin
func newTestNetworkPlugin(t *testing.T) *mock_network.MockNetworkPlugin {
	ctrl := gomock.NewController(t)
	return mock_network.NewMockNetworkPlugin(ctrl)
}

func newTestDockerService() (*dockerService, *dockertools.FakeDockerClient, *clock.FakeClock) {
	fakeClock := clock.NewFakeClock(time.Time{})
	c := dockertools.NewFakeDockerClientWithClock(fakeClock)
	return &dockerService{client: c, os: &containertest.FakeOS{}, networkPlugin: &network.NoopNetworkPlugin{}}, c, fakeClock
}
