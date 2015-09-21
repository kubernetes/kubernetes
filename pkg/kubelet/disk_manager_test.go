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

func setUp(t *testing.T) (*assert.Assertions, DiskSpacePolicy, *cadvisor.Mock) {
	assert := assert.New(t)
	policy := testPolicy()
	c := new(cadvisor.Mock)
	return assert, policy, c
}

func TestValidPolicy(t *testing.T) {
	assert, policy, c := setUp(t)
	_, err := newDiskSpaceManager(c, policy)
	assert.NoError(err)

	policy = testPolicy()
	policy.DockerFreeDiskMB = -1
	_, err = newDiskSpaceManager(c, policy)
	assert.Error(err)

	policy = testPolicy()
	policy.RootFreeDiskMB = -1
	_, err = newDiskSpaceManager(c, policy)
	assert.Error(err)
}

func TestSpaceAvailable(t *testing.T) {
	assert, policy, mockCadvisor := setUp(t)
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	assert.NoError(err)

	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     400 * mb,
		Capacity:  1000 * mb,
		Available: 600 * mb,
	}, nil)
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{
		Usage:    9 * mb,
		Capacity: 10 * mb,
	}, nil)

	dm.Unfreeze()

	ok, err := dm.IsDockerDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	ok, err = dm.IsRootDiskSpaceAvailable()
	assert.NoError(err)
	assert.False(ok)
}

// TestIsDockerDiskSpaceAvailableWithSpace verifies IsDockerDiskSpaceAvailable results when
// space is available.
func TestIsDockerDiskSpaceAvailableWithSpace(t *testing.T) {
	assert, policy, mockCadvisor := setUp(t)
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	require.NoError(t, err)

	// 500MB available
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     9500 * mb,
		Capacity:  10000 * mb,
		Available: 500 * mb,
	}, nil)

	dm.Unfreeze()

	ok, err := dm.IsDockerDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)
}

// TestIsDockerDiskSpaceAvailableWithoutSpace verifies IsDockerDiskSpaceAvailable results when
// space is not available.
func TestIsDockerDiskSpaceAvailableWithoutSpace(t *testing.T) {
	// 1MB available
	assert, policy, mockCadvisor := setUp(t)
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     999 * mb,
		Capacity:  1000 * mb,
		Available: 1 * mb,
	}, nil)

	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	require.NoError(t, err)

	dm.Unfreeze()

	ok, err := dm.IsDockerDiskSpaceAvailable()
	assert.NoError(err)
	assert.False(ok)
}

// TestIsRootDiskSpaceAvailableWithSpace verifies IsRootDiskSpaceAvailable results when
// space is available.
func TestIsRootDiskSpaceAvailableWithSpace(t *testing.T) {
	assert, policy, mockCadvisor := setUp(t)
	policy.RootFreeDiskMB = 10
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	assert.NoError(err)

	// 999MB available
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     1 * mb,
		Capacity:  1000 * mb,
		Available: 999 * mb,
	}, nil)

	dm.Unfreeze()

	ok, err := dm.IsRootDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)
}

// TestIsRootDiskSpaceAvailableWithoutSpace verifies IsRootDiskSpaceAvailable results when
// space is not available.
func TestIsRootDiskSpaceAvailableWithoutSpace(t *testing.T) {
	assert, policy, mockCadvisor := setUp(t)
	policy.RootFreeDiskMB = 10
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	assert.NoError(err)

	// 9MB available
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     990 * mb,
		Capacity:  1000 * mb,
		Available: 9 * mb,
	}, nil)

	dm.Unfreeze()

	ok, err := dm.IsRootDiskSpaceAvailable()
	assert.NoError(err)
	assert.False(ok)
}

// TestCache verifies that caching works properly with DiskSpaceAvailable calls
func TestCache(t *testing.T) {
	assert, policy, mockCadvisor := setUp(t)
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	assert.NoError(err)

	dm.Unfreeze()

	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     400 * mb,
		Capacity:  1000 * mb,
		Available: 300 * mb,
	}, nil).Once()
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     500 * mb,
		Capacity:  1000 * mb,
		Available: 500 * mb,
	}, nil).Once()

	// Initial calls which should be recorded in mockCadvisor
	ok, err := dm.IsDockerDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	ok, err = dm.IsRootDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	// Get the current count of calls to mockCadvisor
	cadvisorCallCount := len(mockCadvisor.Calls)

	// Checking for space again shouldn't need to mock as cache would serve it.
	ok, err = dm.IsDockerDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	ok, err = dm.IsRootDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	// Ensure no more calls to the mockCadvisor occured
	assert.Equal(cadvisorCallCount, len(mockCadvisor.Calls))
}

// TestFsInfoError verifies errors are returned  by DiskSpaceAvailable calls
// when FsInfo calls return an error
func TestFsInfoError(t *testing.T) {
	assert, policy, mockCadvisor := setUp(t)
	policy.RootFreeDiskMB = 10
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	assert.NoError(err)

	dm.Unfreeze()
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApi.FsInfo{}, fmt.Errorf("can't find fs"))
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{}, fmt.Errorf("EBUSY"))
	ok, err := dm.IsDockerDiskSpaceAvailable()
	assert.Error(err)
	assert.True(ok)
	ok, err = dm.IsRootDiskSpaceAvailable()
	assert.Error(err)
	assert.True(ok)
}

// Test_getFSInfo verifies multiple possible cases for getFsInfo.
func Test_getFsInfo(t *testing.T) {
	assert, policy, mockCadvisor := setUp(t)

	// Sunny day case
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     10 * mb,
		Capacity:  100 * mb,
		Available: 90 * mb,
	}, nil).Once()

	dm := &realDiskSpaceManager{
		cadvisor:   mockCadvisor,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
		frozen:     false,
	}

	available, err := dm.isSpaceAvailable("root", 10, dm.cadvisor.RootFsInfo)
	assert.True(available)
	assert.NoError(err)

	// Threshold case
	mockCadvisor = new(cadvisor.Mock)
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     9 * mb,
		Capacity:  100 * mb,
		Available: 9 * mb,
	}, nil).Once()

	dm = &realDiskSpaceManager{
		cadvisor:   mockCadvisor,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
		frozen:     false,
	}
	available, err = dm.isSpaceAvailable("root", 10, dm.cadvisor.RootFsInfo)
	assert.False(available)
	assert.NoError(err)

	// Frozen case
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     9 * mb,
		Capacity:  10 * mb,
		Available: 500 * mb,
	}, nil).Once()

	dm = &realDiskSpaceManager{
		cadvisor:   mockCadvisor,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
		frozen:     true,
	}
	available, err = dm.isSpaceAvailable("root", 10, dm.cadvisor.RootFsInfo)
	assert.True(available)
	assert.NoError(err)

	// Capacity error case
	mockCadvisor = new(cadvisor.Mock)
	mockCadvisor.On("RootFsInfo").Return(cadvisorApi.FsInfo{
		Usage:     9 * mb,
		Capacity:  0,
		Available: 500 * mb,
	}, nil).Once()

	dm = &realDiskSpaceManager{
		cadvisor:   mockCadvisor,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
		frozen:     false,
	}
	available, err = dm.isSpaceAvailable("root", 10, dm.cadvisor.RootFsInfo)
	assert.True(available)
	assert.Error(err)
	assert.Contains(fmt.Sprintf("%s", err), "could not determine capacity")

	// Available error case skipped as v2.FSInfo uses uint64 and this
	// can not be less than 0
}

// TestUnfreeze verifies that Unfreze does infact change the frozen
// private field in master
func TestUnfreeze(t *testing.T) {
	dm := &realDiskSpaceManager{
		cadvisor:   new(cadvisor.Mock),
		policy:     testPolicy(),
		cachedInfo: map[string]fsInfo{},
		frozen:     true,
	}

	dm.Unfreeze()

	if dm.frozen {
		t.Errorf("DiskSpaceManager did not unfreeze: %+v", dm)
	}
}
