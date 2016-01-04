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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/pkg/kubelet/collector"
)

func testPolicy() DiskSpacePolicy {
	return DiskSpacePolicy{
		DockerFreeDiskMB: 250,
		RootFreeDiskMB:   250,
	}
}

func setUp(t *testing.T) (*assert.Assertions, DiskSpacePolicy, *collector.Mock) {
	assert := assert.New(t)
	policy := testPolicy()
	c := new(collector.Mock)
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
	assert, policy, mockCollector := setUp(t)
	dm, err := newDiskSpaceManager(mockCollector, policy)
	assert.NoError(err)

	mockCollector.On("FsInfo", collector.LabelDockerImages).Return(&collector.FsInfo{
		Usage:     400 * mb,
		Capacity:  1000 * mb,
		Available: 600 * mb,
	}, nil)
	mockCollector.On("FsInfo", collector.LabelSystemRoot).Return(&collector.FsInfo{
		Usage:    9 * mb,
		Capacity: 10 * mb,
	}, nil)

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
	assert, policy, mockCollector := setUp(t)
	dm, err := newDiskSpaceManager(mockCollector, policy)
	require.NoError(t, err)

	// 500MB available
	mockCollector.On("FsInfo", collector.LabelDockerImages).Return(&collector.FsInfo{
		Usage:     9500 * mb,
		Capacity:  10000 * mb,
		Available: 500 * mb,
	}, nil)

	ok, err := dm.IsDockerDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)
}

// TestIsDockerDiskSpaceAvailableWithoutSpace verifies IsDockerDiskSpaceAvailable results when
// space is not available.
func TestIsDockerDiskSpaceAvailableWithoutSpace(t *testing.T) {
	// 1MB available
	assert, policy, mockCollector := setUp(t)
	mockCollector.On("FsInfo", collector.LabelDockerImages).Return(&collector.FsInfo{
		Usage:     999 * mb,
		Capacity:  1000 * mb,
		Available: 1 * mb,
	}, nil)

	dm, err := newDiskSpaceManager(mockCollector, policy)
	require.NoError(t, err)

	ok, err := dm.IsDockerDiskSpaceAvailable()
	assert.NoError(err)
	assert.False(ok)
}

// TestIsRootDiskSpaceAvailableWithSpace verifies IsRootDiskSpaceAvailable results when
// space is available.
func TestIsRootDiskSpaceAvailableWithSpace(t *testing.T) {
	assert, policy, mockCollector := setUp(t)
	policy.RootFreeDiskMB = 10
	dm, err := newDiskSpaceManager(mockCollector, policy)
	assert.NoError(err)

	// 999MB available
	mockCollector.On("FsInfo", collector.LabelSystemRoot).Return(&collector.FsInfo{
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
	assert, policy, mockCollector := setUp(t)
	policy.RootFreeDiskMB = 10
	dm, err := newDiskSpaceManager(mockCollector, policy)
	assert.NoError(err)

	// 9MB available
	mockCollector.On("FsInfo", collector.LabelSystemRoot).Return(&collector.FsInfo{
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
	assert, policy, mockCollector := setUp(t)
	dm, err := newDiskSpaceManager(mockCollector, policy)
	assert.NoError(err)

	mockCollector.On("FsInfo", collector.LabelDockerImages).Return(&collector.FsInfo{
		Usage:     400 * mb,
		Capacity:  1000 * mb,
		Available: 300 * mb,
	}, nil).Once()
	mockCollector.On("FsInfo", collector.LabelSystemRoot).Return(&collector.FsInfo{
		Usage:     500 * mb,
		Capacity:  1000 * mb,
		Available: 500 * mb,
	}, nil).Once()

	// Initial calls which should be recorded in mockCollector
	ok, err := dm.IsDockerDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	ok, err = dm.IsRootDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	// Get the current count of calls to mockCollector
	collectorCallCount := len(mockCollector.Calls)

	// Checking for space again shouldn't need to mock as cache would serve it.
	ok, err = dm.IsDockerDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	ok, err = dm.IsRootDiskSpaceAvailable()
	assert.NoError(err)
	assert.True(ok)

	// Ensure no more calls to the mockCollector occured
	assert.Equal(collectorCallCount, len(mockCollector.Calls))
}

// TestFsInfoError verifies errors are returned  by DiskSpaceAvailable calls
// when FsInfo calls return an error
func TestFsInfoError(t *testing.T) {
	assert, policy, mockCollector := setUp(t)
	policy.RootFreeDiskMB = 10
	dm, err := newDiskSpaceManager(mockCollector, policy)
	assert.NoError(err)

	mockCollector.On("FsInfo", collector.LabelDockerImages).Return(&collector.FsInfo{}, fmt.Errorf("can't find fs"))
	mockCollector.On("FsInfo", collector.LabelSystemRoot).Return(&collector.FsInfo{}, fmt.Errorf("EBUSY"))
	ok, err := dm.IsDockerDiskSpaceAvailable()
	assert.Error(err)
	assert.True(ok)
	ok, err = dm.IsRootDiskSpaceAvailable()
	assert.Error(err)
	assert.True(ok)
}

// Test_getFSInfo verifies multiple possible cases for getFsInfo.
func Test_getFsInfo(t *testing.T) {
	assert, policy, mockCollector := setUp(t)

	// Sunny day case
	mockCollector.On("FsInfo", collector.LabelSystemRoot).Return(&collector.FsInfo{
		Usage:     10 * mb,
		Capacity:  100 * mb,
		Available: 90 * mb,
	}, nil).Once()

	dm := &realDiskSpaceManager{
		collector:  mockCollector,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
	}

	available, err := dm.isSpaceAvailable("root", 10, dm.collector.FsInfo, collector.LabelSystemRoot)
	assert.True(available)
	assert.NoError(err)

	// Threshold case
	mockCollector = new(collector.Mock)
	mockCollector.On("FsInfo", collector.LabelSystemRoot).Return(&collector.FsInfo{
		Usage:     9 * mb,
		Capacity:  100 * mb,
		Available: 9 * mb,
	}, nil).Once()

	dm = &realDiskSpaceManager{
		collector:  mockCollector,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
	}
	available, err = dm.isSpaceAvailable("root", 10, dm.collector.FsInfo, collector.LabelSystemRoot)
	assert.False(available)
	assert.NoError(err)

	// Frozen case
	mockCollector.On("FsInfo", collector.LabelSystemRoot).Return(&collector.FsInfo{
		Usage:     9 * mb,
		Capacity:  10 * mb,
		Available: 500 * mb,
	}, nil).Once()

	dm = &realDiskSpaceManager{
		collector:  mockCollector,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
	}
	available, err = dm.isSpaceAvailable("root", 10, dm.collector.FsInfo, collector.LabelSystemRoot)
	assert.True(available)
	assert.NoError(err)

	// Capacity error case
	mockCollector = new(collector.Mock)
	mockCollector.On("FsInfo", collector.LabelSystemRoot).Return(&collector.FsInfo{
		Usage:     9 * mb,
		Capacity:  0,
		Available: 500 * mb,
	}, nil).Once()

	dm = &realDiskSpaceManager{
		collector:  mockCollector,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
	}
	available, err = dm.isSpaceAvailable("root", 10, dm.collector.FsInfo, collector.LabelSystemRoot)
	assert.True(available)
	assert.Error(err)
	assert.Contains(fmt.Sprintf("%s", err), "could not determine capacity")

	// Available error case skipped as v2.FSInfo uses uint64 and this
	// can not be less than 0
}
