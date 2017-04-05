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

	"github.com/blang/semver"
	dockertypes "github.com/docker/engine-api/types"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/clock"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	"k8s.io/kubernetes/pkg/kubelet/network"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
	"k8s.io/kubernetes/pkg/kubelet/util/cache"
)

// newTestNetworkPlugin returns a mock plugin that implements network.NetworkPlugin
func newTestNetworkPlugin(t *testing.T) *nettest.MockNetworkPlugin {
	ctrl := gomock.NewController(t)
	return nettest.NewMockNetworkPlugin(ctrl)
}

func newTestDockerService() (*dockerService, *libdocker.FakeDockerClient, *clock.FakeClock) {
	fakeClock := clock.NewFakeClock(time.Time{})
	c := libdocker.NewFakeDockerClient().WithClock(fakeClock).WithVersion("1.11.2", "1.23")
	pm := network.NewPluginManager(&network.NoopNetworkPlugin{})
	return &dockerService{
		client:            c,
		os:                &containertest.FakeOS{},
		network:           pm,
		legacyCleanup:     legacyCleanupFlag{done: 1},
		checkpointHandler: NewTestPersistentCheckpointHandler(),
		networkReady:      make(map[string]bool),
	}, c, fakeClock
}

func newTestDockerServiceWithVersionCache() (*dockerService, *libdocker.FakeDockerClient, *clock.FakeClock) {
	ds, c, fakeClock := newTestDockerService()
	ds.versionCache = cache.NewObjectCache(
		func() (interface{}, error) {
			return ds.getDockerVersion()
		},
		time.Hour*10,
	)
	return ds, c, fakeClock
}

// TestStatus tests the runtime status logic.
func TestStatus(t *testing.T) {
	ds, fDocker, _ := newTestDockerService()

	assertStatus := func(expected map[string]bool, status *runtimeapi.RuntimeStatus) {
		conditions := status.GetConditions()
		assert.Equal(t, len(expected), len(conditions))
		for k, v := range expected {
			for _, c := range conditions {
				if k == c.Type {
					assert.Equal(t, v, c.Status)
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
	ds.network = network.NewPluginManager(mockPlugin)
	defer mockPlugin.Finish()
	mockPlugin.EXPECT().Status().Return(errors.New("network error"))
	status, err = ds.Status()
	assert.NoError(t, err)
	assertStatus(map[string]bool{
		runtimeapi.RuntimeReady: true,
		runtimeapi.NetworkReady: false,
	}, status)
}

func TestVersion(t *testing.T) {
	ds, _, _ := newTestDockerService()

	expectedVersion := &dockertypes.Version{Version: "1.11.2", APIVersion: "1.23.0"}
	v, err := ds.getDockerVersion()
	require.NoError(t, err)
	assert.Equal(t, expectedVersion, v)

	expectedAPIVersion := &semver.Version{Major: 1, Minor: 23, Patch: 0}
	apiVersion, err := ds.getDockerAPIVersion()
	require.NoError(t, err)
	assert.Equal(t, expectedAPIVersion, apiVersion)
}

func TestAPIVersionWithCache(t *testing.T) {
	ds, _, _ := newTestDockerServiceWithVersionCache()

	expected := &semver.Version{Major: 1, Minor: 23, Patch: 0}
	version, err := ds.getDockerAPIVersion()
	require.NoError(t, err)
	assert.Equal(t, expected, version)
}
