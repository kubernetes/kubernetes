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

package kubelet

import (
	"fmt"
	"testing"

	cadvisorApi "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
)

func testPolicy() DiskSpacePolicy {
	return DiskSpacePolicy{
		DockerFreeDiskMB: 250,
		RootFreeDiskMB:   250,
	}
}

func testValidPolicy(t *testing.T) {
	assert := assert.New(t)
	policy := testPolicy()
	c := new(cadvisor.Mock)
	_, err := newDiskSpaceManager(c, policy)
	require.NoError(t, err)

	policy = testPolicy()
	policy.DockerFreeDiskMB = -1
	_, err = newDiskSpaceManager(c, policy)
	assert.Error(err)

	policy = testPolicy()
	policy.RootFreeDiskMB = -1
	_, err = newDiskSpaceManager(c, policy)
	assert.Error(err)
}

func testSpaceAvailable(t *testing.T) {
	policy := testPolicy()
	mockCadvisor := new(cadvisor.Mock)
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	require.NoError(t, err)
	const mb = 1024 * 1024
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApi.FsInfo{
		Usage:    400 * mb,
		Capacity: 1000 * mb,
	}, nil)
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{
		Usage:    9 * mb,
		Capacity: 10 * mb,
	}, nil)
	ok, err := dm.IsDockerDiskSpaceAvailable()
	require.NoError(t, err)
	require.True(t, ok)
	ok, err = dm.IsRootDiskSpaceAvailable()
	require.NoError(t, err)
	require.True(t, ok)
}

func testRootFsAvailable(t *testing.T) {
	policy := testPolicy()
	policy.RootFreeDiskMB = 10
	mockCadvisor := new(cadvisor.Mock)
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	require.NoError(t, err)

	const mb = 1024 * 1024
	// 500MB available
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     9500 * mb,
		Capacity:  10000 * mb,
		Available: 500 * mb,
	}, nil)
	// 10MB available
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     990 * mb,
		Capacity:  1000 * mb,
		Available: 10 * mb,
	}, nil)
	ok, err := dm.IsDockerDiskSpaceAvailable()
	require.NoError(t, err)
	require.True(t, ok)
	ok, err = dm.IsRootDiskSpaceAvailable()
	require.NoError(t, err)
	require.False(t, ok)
}

func testFsInfoError(t *testing.T) {
	assert := assert.New(t)
	policy := testPolicy()
	policy.RootFreeDiskMB = 10
	mockCadvisor := new(cadvisor.Mock)
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	require.NoError(t, err)

	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApi.FsInfo{}, fmt.Errorf("can't find fs"))
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{}, fmt.Errorf("EBUSY"))
	ok, err := dm.IsDockerDiskSpaceAvailable()
	assert.Error(err)
	require.True(t, ok)
	ok, err = dm.IsRootDiskSpaceAvailable()
	assert.Error(err)
	require.True(t, ok)
}

func testCache(t *testing.T) {
	policy := testPolicy()
	mockCadvisor := new(cadvisor.Mock)
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	require.NoError(t, err)
	const mb = 1024 * 1024
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     400 * mb,
		Capacity:  1000 * mb,
		Available: 300 * mb,
	}, nil).Once()
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     9 * mb,
		Capacity:  10 * mb,
		Available: 500,
	}, nil).Once()
	ok, err := dm.IsDockerDiskSpaceAvailable()
	ok, err = dm.IsRootDiskSpaceAvailable()

	// Checking for space again shouldn't need to mock as cache would serve it.
	ok, err = dm.IsDockerDiskSpaceAvailable()
	require.NoError(t, err)
	require.True(t, ok)
	ok, err = dm.IsRootDiskSpaceAvailable()
	require.NoError(t, err)
	require.True(t, ok)
}
