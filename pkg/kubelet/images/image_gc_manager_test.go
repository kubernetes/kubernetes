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

package images

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/client-go/tools/record"
	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	statstest "k8s.io/kubernetes/pkg/kubelet/server/stats/testing"
)

var zero time.Time
var sandboxImage = "k8s.gcr.io/pause-amd64:latest"

func newRealImageGCManager(policy ImageGCPolicy) (*realImageGCManager, *containertest.FakeRuntime, *statstest.StatsProvider) {
	fakeRuntime := &containertest.FakeRuntime{}
	mockStatsProvider := new(statstest.StatsProvider)
	return &realImageGCManager{
		runtime:       fakeRuntime,
		policy:        policy,
		imageRecords:  make(map[string]*imageRecord),
		statsProvider: mockStatsProvider,
		recorder:      &record.FakeRecorder{},
		sandboxImage:  sandboxImage,
	}, fakeRuntime, mockStatsProvider
}

// Accessors used for thread-safe testing.
func (im *realImageGCManager) imageRecordsLen() int {
	im.imageRecordsLock.Lock()
	defer im.imageRecordsLock.Unlock()
	return len(im.imageRecords)
}
func (im *realImageGCManager) getImageRecord(name string) (*imageRecord, bool) {
	im.imageRecordsLock.Lock()
	defer im.imageRecordsLock.Unlock()
	v, ok := im.imageRecords[name]
	vCopy := *v
	return &vCopy, ok
}

// Returns the id of the image with the given ID.
func imageID(id int) string {
	return fmt.Sprintf("image-%d", id)
}

// Returns the name of the image with the given ID.
func imageName(id int) string {
	return imageID(id) + "-name"
}

// Make an image with the specified ID.
func makeImage(id int, size int64) container.Image {
	return container.Image{
		ID:   imageID(id),
		Size: size,
	}
}

// Make a container with the specified ID. It will use the image with the same ID.
func makeContainer(id int) *container.Container {
	return &container.Container{
		ID:      container.ContainerID{Type: "test", ID: fmt.Sprintf("container-%d", id)},
		Image:   imageName(id),
		ImageID: imageID(id),
	}
}

func TestDetectImagesInitialDetect(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageGCManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
		makeImage(2, 2048),
	}
	fakeRuntime.AllPodList = []*containertest.FakePod{
		{Pod: &container.Pod{
			Containers: []*container.Container{
				{
					ID:      container.ContainerID{Type: "test", ID: fmt.Sprintf("container-%d", 1)},
					ImageID: imageID(1),
					// The image filed is not set to simulate a no-name image
				},
				{
					ID:      container.ContainerID{Type: "test", ID: fmt.Sprintf("container-%d", 2)},
					Image:   imageName(2),
					ImageID: imageID(2),
				},
			},
		}},
	}

	startTime := time.Now().Add(-time.Millisecond)
	_, err := manager.detectImages(zero)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 3)
	noContainer, ok := manager.getImageRecord(imageID(0))
	require.True(t, ok)
	assert.Equal(zero, noContainer.firstDetected)
	assert.Equal(zero, noContainer.lastUsed)
	withContainerUsingNoNameImage, ok := manager.getImageRecord(imageID(1))
	require.True(t, ok)
	assert.Equal(zero, withContainerUsingNoNameImage.firstDetected)
	assert.True(withContainerUsingNoNameImage.lastUsed.After(startTime))
	withContainer, ok := manager.getImageRecord(imageID(2))
	require.True(t, ok)
	assert.Equal(zero, withContainer.firstDetected)
	assert.True(withContainer.lastUsed.After(startTime))
}

func TestDetectImagesWithNewImage(t *testing.T) {
	// Just one image initially.
	manager, fakeRuntime, _ := newRealImageGCManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeRuntime.AllPodList = []*containertest.FakePod{
		{Pod: &container.Pod{
			Containers: []*container.Container{
				makeContainer(1),
			},
		}},
	}

	_, err := manager.detectImages(zero)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)

	// Add a new image.
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 1024),
		makeImage(2, 1024),
	}

	detectedTime := zero.Add(time.Second)
	startTime := time.Now().Add(-time.Millisecond)
	_, err = manager.detectImages(detectedTime)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 3)
	noContainer, ok := manager.getImageRecord(imageID(0))
	require.True(t, ok)
	assert.Equal(zero, noContainer.firstDetected)
	assert.Equal(zero, noContainer.lastUsed)
	withContainer, ok := manager.getImageRecord(imageID(1))
	require.True(t, ok)
	assert.Equal(zero, withContainer.firstDetected)
	assert.True(withContainer.lastUsed.After(startTime))
	newContainer, ok := manager.getImageRecord(imageID(2))
	require.True(t, ok)
	assert.Equal(detectedTime, newContainer.firstDetected)
	assert.Equal(zero, noContainer.lastUsed)
}

func TestDeleteUnusedImagesExemptSandboxImage(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageGCManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		{
			ID:   sandboxImage,
			Size: 1024,
		},
	}

	err := manager.DeleteUnusedImages()
	assert := assert.New(t)
	assert.Len(fakeRuntime.ImageList, 1)
	require.NoError(t, err)
}

func TestDetectImagesContainerStopped(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageGCManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeRuntime.AllPodList = []*containertest.FakePod{
		{Pod: &container.Pod{
			Containers: []*container.Container{
				makeContainer(1),
			},
		}},
	}

	_, err := manager.detectImages(zero)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)
	withContainer, ok := manager.getImageRecord(imageID(1))
	require.True(t, ok)

	// Simulate container being stopped.
	fakeRuntime.AllPodList = []*containertest.FakePod{}
	_, err = manager.detectImages(time.Now())
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)
	container1, ok := manager.getImageRecord(imageID(0))
	require.True(t, ok)
	assert.Equal(zero, container1.firstDetected)
	assert.Equal(zero, container1.lastUsed)
	container2, ok := manager.getImageRecord(imageID(1))
	require.True(t, ok)
	assert.Equal(zero, container2.firstDetected)
	assert.True(container2.lastUsed.Equal(withContainer.lastUsed))
}

func TestDetectImagesWithRemovedImages(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageGCManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeRuntime.AllPodList = []*containertest.FakePod{
		{Pod: &container.Pod{
			Containers: []*container.Container{
				makeContainer(1),
			},
		}},
	}

	_, err := manager.detectImages(zero)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)

	// Simulate both images being removed.
	fakeRuntime.ImageList = []container.Image{}
	_, err = manager.detectImages(time.Now())
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 0)
}

func TestFreeSpaceImagesInUseContainersAreIgnored(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageGCManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeRuntime.AllPodList = []*containertest.FakePod{
		{Pod: &container.Pod{
			Containers: []*container.Container{
				makeContainer(1),
			},
		}},
	}

	spaceFreed, err := manager.freeSpace(2048, time.Now())
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(1024, spaceFreed)
	assert.Len(fakeRuntime.ImageList, 1)
}

func TestDeleteUnusedImagesRemoveAllUnusedImages(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageGCManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
		makeImage(2, 2048),
	}
	fakeRuntime.AllPodList = []*containertest.FakePod{
		{Pod: &container.Pod{
			Containers: []*container.Container{
				makeContainer(2),
			},
		}},
	}

	err := manager.DeleteUnusedImages()
	assert := assert.New(t)
	require.NoError(t, err)
	assert.Len(fakeRuntime.ImageList, 1)
}

func TestFreeSpaceRemoveByLeastRecentlyUsed(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageGCManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeRuntime.AllPodList = []*containertest.FakePod{
		{Pod: &container.Pod{
			Containers: []*container.Container{
				makeContainer(0),
				makeContainer(1),
			},
		}},
	}

	// Make 1 be more recently used than 0.
	_, err := manager.detectImages(zero)
	require.NoError(t, err)
	fakeRuntime.AllPodList = []*containertest.FakePod{
		{Pod: &container.Pod{
			Containers: []*container.Container{
				makeContainer(1),
			},
		}},
	}
	_, err = manager.detectImages(time.Now())
	require.NoError(t, err)
	fakeRuntime.AllPodList = []*containertest.FakePod{
		{Pod: &container.Pod{
			Containers: []*container.Container{},
		}},
	}
	_, err = manager.detectImages(time.Now())
	require.NoError(t, err)
	require.Equal(t, manager.imageRecordsLen(), 2)

	spaceFreed, err := manager.freeSpace(1024, time.Now())
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(1024, spaceFreed)
	assert.Len(fakeRuntime.ImageList, 1)
}

func TestFreeSpaceTiesBrokenByDetectedTime(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageGCManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
	}
	fakeRuntime.AllPodList = []*containertest.FakePod{
		{Pod: &container.Pod{
			Containers: []*container.Container{
				makeContainer(0),
			},
		}},
	}

	// Make 1 more recently detected but used at the same time as 0.
	_, err := manager.detectImages(zero)
	require.NoError(t, err)
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	_, err = manager.detectImages(time.Now())
	require.NoError(t, err)
	fakeRuntime.AllPodList = []*containertest.FakePod{}
	_, err = manager.detectImages(time.Now())
	require.NoError(t, err)
	require.Equal(t, manager.imageRecordsLen(), 2)

	spaceFreed, err := manager.freeSpace(1024, time.Now())
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(2048, spaceFreed)
	assert.Len(fakeRuntime.ImageList, 1)
}

func TestGarbageCollectBelowLowThreshold(t *testing.T) {
	policy := ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	manager, _, mockStatsProvider := newRealImageGCManager(policy)

	// Expect 40% usage.
	mockStatsProvider.On("ImageFsStats").Return(&statsapi.FsStats{
		AvailableBytes: uint64Ptr(600),
		CapacityBytes:  uint64Ptr(1000),
	}, nil)

	assert.NoError(t, manager.GarbageCollect())
}

func TestGarbageCollectCadvisorFailure(t *testing.T) {
	policy := ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	manager, _, mockStatsProvider := newRealImageGCManager(policy)

	mockStatsProvider.On("ImageFsStats").Return(&statsapi.FsStats{}, fmt.Errorf("error"))
	assert.NotNil(t, manager.GarbageCollect())
}

func TestGarbageCollectBelowSuccess(t *testing.T) {
	policy := ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	manager, fakeRuntime, mockStatsProvider := newRealImageGCManager(policy)

	// Expect 95% usage and most of it gets freed.
	mockStatsProvider.On("ImageFsStats").Return(&statsapi.FsStats{
		AvailableBytes: uint64Ptr(50),
		CapacityBytes:  uint64Ptr(1000),
	}, nil)
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 450),
	}

	assert.NoError(t, manager.GarbageCollect())
}

func TestGarbageCollectNotEnoughFreed(t *testing.T) {
	policy := ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	manager, fakeRuntime, mockStatsProvider := newRealImageGCManager(policy)

	// Expect 95% usage and little of it gets freed.
	mockStatsProvider.On("ImageFsStats").Return(&statsapi.FsStats{
		AvailableBytes: uint64Ptr(50),
		CapacityBytes:  uint64Ptr(1000),
	}, nil)
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 50),
	}

	assert.NotNil(t, manager.GarbageCollect())
}

func TestGarbageCollectImageNotOldEnough(t *testing.T) {
	policy := ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
		MinAge:               time.Minute * 1,
	}
	fakeRuntime := &containertest.FakeRuntime{}
	mockStatsProvider := new(statstest.StatsProvider)
	manager := &realImageGCManager{
		runtime:       fakeRuntime,
		policy:        policy,
		imageRecords:  make(map[string]*imageRecord),
		statsProvider: mockStatsProvider,
		recorder:      &record.FakeRecorder{},
	}

	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	// 1 image is in use, and another one is not old enough
	fakeRuntime.AllPodList = []*containertest.FakePod{
		{Pod: &container.Pod{
			Containers: []*container.Container{
				makeContainer(1),
			},
		}},
	}

	fakeClock := clock.NewFakeClock(time.Now())
	t.Log(fakeClock.Now())
	_, err := manager.detectImages(fakeClock.Now())
	require.NoError(t, err)
	require.Equal(t, manager.imageRecordsLen(), 2)
	// no space freed since one image is in used, and another one is not old enough
	spaceFreed, err := manager.freeSpace(1024, fakeClock.Now())
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(0, spaceFreed)
	assert.Len(fakeRuntime.ImageList, 2)

	// move clock by minAge duration, then 1 image will be garbage collected
	fakeClock.Step(policy.MinAge)
	spaceFreed, err = manager.freeSpace(1024, fakeClock.Now())
	require.NoError(t, err)
	assert.EqualValues(1024, spaceFreed)
	assert.Len(fakeRuntime.ImageList, 1)
}

func TestValidateImageGCPolicy(t *testing.T) {
	testCases := []struct {
		name          string
		imageGCPolicy ImageGCPolicy
		expectErr     string
	}{
		{
			name: "Test for LowThresholdPercent < HighThresholdPercent",
			imageGCPolicy: ImageGCPolicy{
				HighThresholdPercent: 2,
				LowThresholdPercent:  1,
			},
		},
		{
			name: "Test for HighThresholdPercent < 0,",
			imageGCPolicy: ImageGCPolicy{
				HighThresholdPercent: -1,
			},
			expectErr: "invalid HighThresholdPercent -1, must be in range [0-100]",
		},
		{
			name: "Test for HighThresholdPercent > 100",
			imageGCPolicy: ImageGCPolicy{
				HighThresholdPercent: 101,
			},
			expectErr: "invalid HighThresholdPercent 101, must be in range [0-100]",
		},
		{
			name: "Test for LowThresholdPercent < 0",
			imageGCPolicy: ImageGCPolicy{
				LowThresholdPercent: -1,
			},
			expectErr: "invalid LowThresholdPercent -1, must be in range [0-100]",
		},
		{
			name: "Test for LowThresholdPercent > 100",
			imageGCPolicy: ImageGCPolicy{
				LowThresholdPercent: 101,
			},
			expectErr: "invalid LowThresholdPercent 101, must be in range [0-100]",
		},
		{
			name: "Test for LowThresholdPercent > HighThresholdPercent",
			imageGCPolicy: ImageGCPolicy{
				HighThresholdPercent: 1,
				LowThresholdPercent:  2,
			},
			expectErr: "LowThresholdPercent 2 can not be higher than HighThresholdPercent 1",
		},
	}

	for _, tc := range testCases {
		if _, err := NewImageGCManager(nil, nil, nil, nil, tc.imageGCPolicy, ""); err != nil {
			if err.Error() != tc.expectErr {
				t.Errorf("[%s:]Expected err:%v, but got:%v", tc.name, tc.expectErr, err.Error())
			}
		}
	}
}

func uint64Ptr(i uint64) *uint64 {
	return &i
}
