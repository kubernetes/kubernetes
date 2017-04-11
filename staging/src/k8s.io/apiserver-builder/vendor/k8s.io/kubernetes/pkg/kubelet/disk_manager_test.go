/*
Copyright 2015 The Kubernetes Authors.

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

	cadvisorapi "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
)

func testPolicy() DiskSpacePolicy {
	return DiskSpacePolicy{
		DockerFreeDiskMB: 250,
		RootFreeDiskMB:   250,
	}
}

func setUp(t *testing.T) (*assert.Assertions, DiskSpacePolicy, *cadvisortest.Mock) {
	assert := assert.New(t)
	policy := testPolicy()
	c := new(cadvisortest.Mock)
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

	mockCadvisor.On("ImagesFsInfo").Return(cadvisorapi.FsInfo{
		Usage:     400 * mb,
		Capacity:  1000 * mb,
		Available: 600 * mb,
	}, nil)
	mockCadvisor.On("RootFsInfo").Return(cadvisorapi.FsInfo{
		Usage:    9 * mb,
		Capacity: 10 * mb,
	}, nil)

	ok, err := dm.IsRuntimeDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	ok, err = dm.IsRootDiskSpaceAvailable()
	assert.NoError(err)
	assert.False(ok)
}

// TestIsRuntimeDiskSpaceAvailableWithSpace verifies IsRuntimeDiskSpaceAvailable results when
// space is available.
func TestIsRuntimeDiskSpaceAvailableWithSpace(t *testing.T) {
	assert, policy, mockCadvisor := setUp(t)
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	require.NoError(t, err)

	// 500MB available
	mockCadvisor.On("ImagesFsInfo").Return(cadvisorapi.FsInfo{
		Usage:     9500 * mb,
		Capacity:  10000 * mb,
		Available: 500 * mb,
	}, nil)

	ok, err := dm.IsRuntimeDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)
}

// TestIsRuntimeDiskSpaceAvailableWithoutSpace verifies IsRuntimeDiskSpaceAvailable results when
// space is not available.
func TestIsRuntimeDiskSpaceAvailableWithoutSpace(t *testing.T) {
	// 1MB available
	assert, policy, mockCadvisor := setUp(t)
	mockCadvisor.On("ImagesFsInfo").Return(cadvisorapi.FsInfo{
		Usage:     999 * mb,
		Capacity:  1000 * mb,
		Available: 1 * mb,
	}, nil)

	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	require.NoError(t, err)

	ok, err := dm.IsRuntimeDiskSpaceAvailable()
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
	mockCadvisor.On("RootFsInfo").Return(cadvisorapi.FsInfo{
		Usage:     1 * mb,
		Capacity:  1000 * mb,
		Available: 999 * mb,
	}, nil)

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
	mockCadvisor.On("RootFsInfo").Return(cadvisorapi.FsInfo{
		Usage:     990 * mb,
		Capacity:  1000 * mb,
		Available: 9 * mb,
	}, nil)

	ok, err := dm.IsRootDiskSpaceAvailable()
	assert.NoError(err)
	assert.False(ok)
}

// TestCache verifies that caching works properly with DiskSpaceAvailable calls
func TestCache(t *testing.T) {
	assert, policy, mockCadvisor := setUp(t)
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	assert.NoError(err)

	mockCadvisor.On("ImagesFsInfo").Return(cadvisorapi.FsInfo{
		Usage:     400 * mb,
		Capacity:  1000 * mb,
		Available: 300 * mb,
	}, nil).Once()
	mockCadvisor.On("RootFsInfo").Return(cadvisorapi.FsInfo{
		Usage:     500 * mb,
		Capacity:  1000 * mb,
		Available: 500 * mb,
	}, nil).Once()

	// Initial calls which should be recorded in mockCadvisor
	ok, err := dm.IsRuntimeDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	ok, err = dm.IsRootDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	// Get the current count of calls to mockCadvisor
	cadvisorCallCount := len(mockCadvisor.Calls)

	// Checking for space again shouldn't need to mock as cache would serve it.
	ok, err = dm.IsRuntimeDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	ok, err = dm.IsRootDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	// Ensure no more calls to the mockCadvisor occurred
	assert.Equal(cadvisorCallCount, len(mockCadvisor.Calls))
}

// TestFsInfoError verifies errors are returned  by DiskSpaceAvailable calls
// when FsInfo calls return an error
func TestFsInfoError(t *testing.T) {
	assert, policy, mockCadvisor := setUp(t)
	policy.RootFreeDiskMB = 10
	dm, err := newDiskSpaceManager(mockCadvisor, policy)
	assert.NoError(err)

	mockCadvisor.On("ImagesFsInfo").Return(cadvisorapi.FsInfo{}, fmt.Errorf("can't find fs"))
	mockCadvisor.On("RootFsInfo").Return(cadvisorapi.FsInfo{}, fmt.Errorf("EBUSY"))
	ok, err := dm.IsRuntimeDiskSpaceAvailable()
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
	mockCadvisor.On("RootFsInfo").Return(cadvisorapi.FsInfo{
		Usage:     10 * mb,
		Capacity:  100 * mb,
		Available: 90 * mb,
	}, nil).Once()

	dm := &realDiskSpaceManager{
		cadvisor:   mockCadvisor,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
	}

	available, err := dm.isSpaceAvailable("root", 10, dm.cadvisor.RootFsInfo)
	assert.True(available)
	assert.NoError(err)

	// Threshold case
	mockCadvisor = new(cadvisortest.Mock)
	mockCadvisor.On("RootFsInfo").Return(cadvisorapi.FsInfo{
		Usage:     9 * mb,
		Capacity:  100 * mb,
		Available: 9 * mb,
	}, nil).Once()

	dm = &realDiskSpaceManager{
		cadvisor:   mockCadvisor,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
	}
	available, err = dm.isSpaceAvailable("root", 10, dm.cadvisor.RootFsInfo)
	assert.False(available)
	assert.NoError(err)

	// Frozen case
	mockCadvisor.On("RootFsInfo").Return(cadvisorapi.FsInfo{
		Usage:     9 * mb,
		Capacity:  10 * mb,
		Available: 500 * mb,
	}, nil).Once()

	dm = &realDiskSpaceManager{
		cadvisor:   mockCadvisor,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
	}
	available, err = dm.isSpaceAvailable("root", 10, dm.cadvisor.RootFsInfo)
	assert.True(available)
	assert.NoError(err)

	// Capacity error case
	mockCadvisor = new(cadvisortest.Mock)
	mockCadvisor.On("RootFsInfo").Return(cadvisorapi.FsInfo{
		Usage:     9 * mb,
		Capacity:  0,
		Available: 500 * mb,
	}, nil).Once()

	dm = &realDiskSpaceManager{
		cadvisor:   mockCadvisor,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
	}
	available, err = dm.isSpaceAvailable("root", 10, dm.cadvisor.RootFsInfo)
	assert.True(available)
	assert.Error(err)
	assert.Contains(fmt.Sprintf("%s", err), "could not determine capacity")

	// Available error case skipped as v2.FSInfo uses uint64 and this
	// can not be less than 0
}
