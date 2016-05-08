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
	"time"

	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/pkg/client/record"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
	"k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/util"
)

var zero time.Time

func newRealImageManager(policy ImageGCPolicy) (*realImageManager, *containertest.FakeRuntime, *cadvisortest.Mock) {
	fakeRuntime := &containertest.FakeRuntime{}
	mockCadvisor := new(cadvisortest.Mock)
	return &realImageManager{
		runtime:      fakeRuntime,
		policy:       policy,
		imageRecords: make(map[string]*imageRecord),
		cadvisor:     mockCadvisor,
		recorder:     &record.FakeRecorder{},
	}, fakeRuntime, mockCadvisor
}

// Accessors used for thread-safe testing.
func (im *realImageManager) imageRecordsLen() int {
	im.imageRecordsLock.Lock()
	defer im.imageRecordsLock.Unlock()
	return len(im.imageRecords)
}
func (im *realImageManager) getImageRecord(name string) (*imageRecord, bool) {
	im.imageRecordsLock.Lock()
	defer im.imageRecordsLock.Unlock()
	v, ok := im.imageRecords[name]
	vCopy := *v
	return &vCopy, ok
}

// Returns the name of the image with the given ID.
func imageName(id int) string {
	return fmt.Sprintf("image-%d", id)
}

// Make an image with the specified ID.
func makeImage(id int, size int64) container.Image {
	return container.Image{
		ID:   imageName(id),
		Size: size,
	}
}

// Make a container with the specified ID. It will use the image with the same ID.
func makeContainer(id int) *container.Container {
	return &container.Container{
		ID:    container.ContainerID{Type: "test", ID: fmt.Sprintf("container-%d", id)},
		Image: imageName(id),
	}
}

func TestDetectImagesInitialDetect(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeRuntime.AllPodList = []*container.Pod{
		{
			Containers: []*container.Container{
				makeContainer(1),
			},
		},
	}

	startTime := time.Now().Add(-time.Millisecond)
	err := manager.detectImages(zero)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)
	noContainer, ok := manager.getImageRecord(imageName(0))
	require.True(t, ok)
	assert.Equal(zero, noContainer.firstDetected)
	assert.Equal(zero, noContainer.lastUsed)
	withContainer, ok := manager.getImageRecord(imageName(1))
	require.True(t, ok)
	assert.Equal(zero, withContainer.firstDetected)
	assert.True(withContainer.lastUsed.After(startTime))
}

func TestDetectImagesWithNewImage(t *testing.T) {
	// Just one image initially.
	manager, fakeRuntime, _ := newRealImageManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeRuntime.AllPodList = []*container.Pod{
		{
			Containers: []*container.Container{
				makeContainer(1),
			},
		},
	}

	err := manager.detectImages(zero)
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
	err = manager.detectImages(detectedTime)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 3)
	noContainer, ok := manager.getImageRecord(imageName(0))
	require.True(t, ok)
	assert.Equal(zero, noContainer.firstDetected)
	assert.Equal(zero, noContainer.lastUsed)
	withContainer, ok := manager.getImageRecord(imageName(1))
	require.True(t, ok)
	assert.Equal(zero, withContainer.firstDetected)
	assert.True(withContainer.lastUsed.After(startTime))
	newContainer, ok := manager.getImageRecord(imageName(2))
	require.True(t, ok)
	assert.Equal(detectedTime, newContainer.firstDetected)
	assert.Equal(zero, noContainer.lastUsed)
}

func TestDetectImagesContainerStopped(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeRuntime.AllPodList = []*container.Pod{
		{
			Containers: []*container.Container{
				makeContainer(1),
			},
		},
	}

	err := manager.detectImages(zero)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)
	withContainer, ok := manager.getImageRecord(imageName(1))
	require.True(t, ok)

	// Simulate container being stopped.
	fakeRuntime.AllPodList = []*container.Pod{}
	err = manager.detectImages(time.Now())
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)
	container1, ok := manager.getImageRecord(imageName(0))
	require.True(t, ok)
	assert.Equal(zero, container1.firstDetected)
	assert.Equal(zero, container1.lastUsed)
	container2, ok := manager.getImageRecord(imageName(1))
	require.True(t, ok)
	assert.Equal(zero, container2.firstDetected)
	assert.True(container2.lastUsed.Equal(withContainer.lastUsed))
}

func TestDetectImagesWithRemovedImages(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeRuntime.AllPodList = []*container.Pod{
		{
			Containers: []*container.Container{
				makeContainer(1),
			},
		},
	}

	err := manager.detectImages(zero)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)

	// Simulate both images being removed.
	fakeRuntime.ImageList = []container.Image{}
	err = manager.detectImages(time.Now())
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 0)
}

func TestFreeSpaceImagesInUseContainersAreIgnored(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeRuntime.AllPodList = []*container.Pod{
		{
			Containers: []*container.Container{
				makeContainer(1),
			},
		},
	}

	spaceFreed, err := manager.freeSpace(2048, time.Now())
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(1024, spaceFreed)
	assert.Len(fakeRuntime.ImageList, 1)
}

func TestFreeSpaceRemoveByLeastRecentlyUsed(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeRuntime.AllPodList = []*container.Pod{
		{
			Containers: []*container.Container{
				makeContainer(0),
				makeContainer(1),
			},
		},
	}

	// Make 1 be more recently used than 0.
	require.NoError(t, manager.detectImages(zero))
	fakeRuntime.AllPodList = []*container.Pod{
		{
			Containers: []*container.Container{
				makeContainer(1),
			},
		},
	}
	require.NoError(t, manager.detectImages(time.Now()))
	fakeRuntime.AllPodList = []*container.Pod{
		{
			Containers: []*container.Container{},
		},
	}
	require.NoError(t, manager.detectImages(time.Now()))
	require.Equal(t, manager.imageRecordsLen(), 2)

	spaceFreed, err := manager.freeSpace(1024, time.Now())
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(1024, spaceFreed)
	assert.Len(fakeRuntime.ImageList, 1)
}

func TestFreeSpaceTiesBrokenByDetectedTime(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
	}
	fakeRuntime.AllPodList = []*container.Pod{
		{
			Containers: []*container.Container{
				makeContainer(0),
			},
		},
	}

	// Make 1 more recently detected but used at the same time as 0.
	require.NoError(t, manager.detectImages(zero))
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	require.NoError(t, manager.detectImages(time.Now()))
	fakeRuntime.AllPodList = []*container.Pod{}
	require.NoError(t, manager.detectImages(time.Now()))
	require.Equal(t, manager.imageRecordsLen(), 2)

	spaceFreed, err := manager.freeSpace(1024, time.Now())
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(2048, spaceFreed)
	assert.Len(fakeRuntime.ImageList, 1)
}

func TestFreeSpaceImagesAlsoDoesLookupByRepoTags(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		{
			ID:       "5678",
			RepoTags: []string{"potato", "salad"},
			Size:     2048,
		},
	}
	fakeRuntime.AllPodList = []*container.Pod{
		{
			Containers: []*container.Container{
				{
					ID:    container.ContainerID{Type: "test", ID: "c5678"},
					Image: "salad",
				},
			},
		},
	}

	spaceFreed, err := manager.freeSpace(1024, time.Now())
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(1024, spaceFreed)
	assert.Len(fakeRuntime.ImageList, 1)
}

func TestFreeSpaceImagesAlsoDoesLookupByRepoDigests(t *testing.T) {
	manager, fakeRuntime, _ := newRealImageManager(ImageGCPolicy{})
	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		{
			ID:          "5678",
			RepoDigests: []string{"potato", "salad"},
			Size:        2048,
		},
	}
	fakeRuntime.AllPodList = []*container.Pod{
		{
			Containers: []*container.Container{
				{
					ID:    container.ContainerID{Type: "test", ID: "c5678"},
					Image: "salad",
				},
			},
		},
	}

	spaceFreed, err := manager.freeSpace(1024, time.Now())
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(1024, spaceFreed)
	assert.Len(fakeRuntime.ImageList, 1)
}

func TestGarbageCollectBelowLowThreshold(t *testing.T) {
	policy := ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	manager, _, mockCadvisor := newRealImageManager(policy)

	// Expect 40% usage.
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{
		Usage:    400,
		Capacity: 1000,
	}, nil)

	assert.NoError(t, manager.GarbageCollect())
}

func TestGarbageCollectCadvisorFailure(t *testing.T) {
	policy := ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	manager, _, mockCadvisor := newRealImageManager(policy)

	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, fmt.Errorf("error"))
	assert.NotNil(t, manager.GarbageCollect())
}

func TestGarbageCollectBelowSuccess(t *testing.T) {
	policy := ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	manager, fakeRuntime, mockCadvisor := newRealImageManager(policy)

	// Expect 95% usage and most of it gets freed.
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{
		Usage:    950,
		Capacity: 1000,
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
	manager, fakeRuntime, mockCadvisor := newRealImageManager(policy)

	// Expect 95% usage and little of it gets freed.
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{
		Usage:    950,
		Capacity: 1000,
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
	mockCadvisor := new(cadvisortest.Mock)
	manager := &realImageManager{
		runtime:      fakeRuntime,
		policy:       policy,
		imageRecords: make(map[string]*imageRecord),
		cadvisor:     mockCadvisor,
		recorder:     &record.FakeRecorder{},
	}

	fakeRuntime.ImageList = []container.Image{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	// 1 image is in use, and another one is not old enough
	fakeRuntime.AllPodList = []*container.Pod{
		{
			Containers: []*container.Container{
				makeContainer(1),
			},
		},
	}

	fakeClock := util.NewFakeClock(time.Now())
	t.Log(fakeClock.Now())
	require.NoError(t, manager.detectImages(fakeClock.Now()))
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
