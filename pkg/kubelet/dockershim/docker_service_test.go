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
	"errors"
	"testing"
	"time"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
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

// TestStatus tests the runtime status logic.
func TestStatus(t *testing.T) {
	ds, fDocker, _ := newTestDockerService()

	assertStatus := func(expected map[string]bool, status *runtimeapi.RuntimeStatus) {
		conditions := status.GetConditions()
		assert.Equal(t, len(expected), len(conditions))
		for k, v := range expected {
			for _, c := range conditions {
				if k == c.GetType() {
					assert.Equal(t, v, c.GetStatus())
				}
			}
		}
	}

	// Should report ready status if version returns no error.
	status, err := ds.Status()
	assert.NoError(t, err)
	assertStatus(map[string]bool{
		runtimeapi.RuntimeReady: true,
		runtimeapi.NetworkReady: true,
	}, status)

	// Should not report ready status if version returns error.
	fDocker.InjectError("version", errors.New("test error"))
	status, err = ds.Status()
	assert.NoError(t, err)
	assertStatus(map[string]bool{
		runtimeapi.RuntimeReady: false,
		runtimeapi.NetworkReady: true,
	}, status)

	// Should not report ready status is network plugin returns error.
	mockPlugin := newTestNetworkPlugin(t)
	ds.networkPlugin = mockPlugin
	defer mockPlugin.Finish()
	mockPlugin.EXPECT().Status().Return(errors.New("network error"))
	status, err = ds.Status()
	assert.NoError(t, err)
	assertStatus(map[string]bool{
		runtimeapi.RuntimeReady: true,
		runtimeapi.NetworkReady: false,
	}, status)
}
