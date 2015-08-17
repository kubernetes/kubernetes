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

	docker "github.com/fsouza/go-dockerclient"
	cadvisorApiV2 "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/util"
)

var zero time.Time

func newRealImageManager(policy ImageGCPolicy) (*realImageManager, *dockertools.FakeDockerClient, *cadvisor.Mock) {
	fakeDocker := &dockertools.FakeDockerClient{
		RemovedImages: util.NewStringSet(),
	}
	mockCadvisor := new(cadvisor.Mock)
	return &realImageManager{
		dockerClient: fakeDocker,
		policy:       policy,
		imageRecords: make(map[string]*imageRecord),
		cadvisor:     mockCadvisor,
		recorder:     &record.FakeRecorder{},
	}, fakeDocker, mockCadvisor
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
func makeImage(id int, size int64) docker.APIImages {
	return docker.APIImages{
		ID:          imageName(id),
		VirtualSize: size,
	}
}

// Make a container with the specified ID. It will use the image with the same ID.
func makeContainer(id int) docker.APIContainers {
	return docker.APIContainers{
		ID:    fmt.Sprintf("container-%d", id),
		Image: imageName(id),
	}
}

func TestDetectImagesInitialDetect(t *testing.T) {
	manager, fakeDocker, _ := newRealImageManager(ImageGCPolicy{})
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		makeContainer(1),
	}

	startTime := time.Now().Add(-time.Millisecond)
	err := manager.detectImages(zero)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)
	noContainer, ok := manager.getImageRecord(imageName(0))
	require.True(t, ok)
	assert.Equal(zero, noContainer.detected)
	assert.Equal(zero, noContainer.lastUsed)
	withContainer, ok := manager.getImageRecord(imageName(1))
	require.True(t, ok)
	assert.Equal(zero, withContainer.detected)
	assert.True(withContainer.lastUsed.After(startTime))
}

func TestDetectImagesWithNewImage(t *testing.T) {
	// Just one image initially.
	manager, fakeDocker, _ := newRealImageManager(ImageGCPolicy{})
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		makeContainer(1),
	}

	err := manager.detectImages(zero)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)

	// Add a new image.
	fakeDocker.Images = []docker.APIImages{
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
	assert.Equal(zero, noContainer.detected)
	assert.Equal(zero, noContainer.lastUsed)
	withContainer, ok := manager.getImageRecord(imageName(1))
	require.True(t, ok)
	assert.Equal(zero, withContainer.detected)
	assert.True(withContainer.lastUsed.After(startTime))
	newContainer, ok := manager.getImageRecord(imageName(2))
	require.True(t, ok)
	assert.Equal(detectedTime, newContainer.detected)
	assert.Equal(zero, noContainer.lastUsed)
}

func TestDetectImagesContainerStopped(t *testing.T) {
	manager, fakeDocker, _ := newRealImageManager(ImageGCPolicy{})
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		makeContainer(1),
	}

	err := manager.detectImages(zero)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)
	withContainer, ok := manager.getImageRecord(imageName(1))
	require.True(t, ok)

	// Simulate container being stopped.
	fakeDocker.ContainerList = []docker.APIContainers{}
	err = manager.detectImages(time.Now())
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)
	container1, ok := manager.getImageRecord(imageName(0))
	require.True(t, ok)
	assert.Equal(zero, container1.detected)
	assert.Equal(zero, container1.lastUsed)
	container2, ok := manager.getImageRecord(imageName(1))
	require.True(t, ok)
	assert.Equal(zero, container2.detected)
	assert.True(container2.lastUsed.Equal(withContainer.lastUsed))
}

func TestDetectImagesWithRemovedImages(t *testing.T) {
	manager, fakeDocker, _ := newRealImageManager(ImageGCPolicy{})
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		makeContainer(1),
	}

	err := manager.detectImages(zero)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 2)

	// Simulate both images being removed.
	fakeDocker.Images = []docker.APIImages{}
	err = manager.detectImages(time.Now())
	require.NoError(t, err)
	assert.Equal(manager.imageRecordsLen(), 0)
}

func TestFreeSpaceImagesInUseContainersAreIgnored(t *testing.T) {
	manager, fakeDocker, _ := newRealImageManager(ImageGCPolicy{})
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		makeContainer(1),
	}

	spaceFreed, err := manager.freeSpace(2048)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(1024, spaceFreed)
	assert.Len(fakeDocker.RemovedImages, 1)
	assert.True(fakeDocker.RemovedImages.Has(imageName(0)))
}

func TestFreeSpaceRemoveByLeastRecentlyUsed(t *testing.T) {
	manager, fakeDocker, _ := newRealImageManager(ImageGCPolicy{})
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		makeContainer(0),
		makeContainer(1),
	}

	// Make 1 be more recently used than 0.
	require.NoError(t, manager.detectImages(zero))
	fakeDocker.ContainerList = []docker.APIContainers{
		makeContainer(1),
	}
	require.NoError(t, manager.detectImages(time.Now()))
	fakeDocker.ContainerList = []docker.APIContainers{}
	require.NoError(t, manager.detectImages(time.Now()))
	require.Equal(t, manager.imageRecordsLen(), 2)

	spaceFreed, err := manager.freeSpace(1024)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(1024, spaceFreed)
	assert.Len(fakeDocker.RemovedImages, 1)
	assert.True(fakeDocker.RemovedImages.Has(imageName(0)))
}

func TestFreeSpaceTiesBrokenByDetectedTime(t *testing.T) {
	manager, fakeDocker, _ := newRealImageManager(ImageGCPolicy{})
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 1024),
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		makeContainer(0),
	}

	// Make 1 more recently detected but used at the same time as 0.
	require.NoError(t, manager.detectImages(zero))
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		makeContainer(0),
		makeContainer(1),
	}
	require.NoError(t, manager.detectImages(time.Now()))
	fakeDocker.ContainerList = []docker.APIContainers{}
	require.NoError(t, manager.detectImages(time.Now()))
	require.Equal(t, manager.imageRecordsLen(), 2)

	spaceFreed, err := manager.freeSpace(1024)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(1024, spaceFreed)
	assert.Len(fakeDocker.RemovedImages, 1)
	assert.True(fakeDocker.RemovedImages.Has(imageName(0)))
}

func TestFreeSpaceImagesAlsoDoesLookupByRepoTags(t *testing.T) {
	manager, fakeDocker, _ := newRealImageManager(ImageGCPolicy{})
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 1024),
		{
			ID:          "5678",
			RepoTags:    []string{"potato", "salad"},
			VirtualSize: 2048,
		},
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    "c5678",
			Image: "salad",
		},
	}

	spaceFreed, err := manager.freeSpace(1024)
	assert := assert.New(t)
	require.NoError(t, err)
	assert.EqualValues(1024, spaceFreed)
	assert.Len(fakeDocker.RemovedImages, 1)
	assert.True(fakeDocker.RemovedImages.Has(imageName(0)))
}

func TestGarbageCollectBelowLowThreshold(t *testing.T) {
	policy := ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	manager, _, mockCadvisor := newRealImageManager(policy)

	// Expect 40% usage.
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiV2.FsInfo{
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

	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiV2.FsInfo{}, fmt.Errorf("error"))
	assert.NotNil(t, manager.GarbageCollect())
}

func TestGarbageCollectBelowSuccess(t *testing.T) {
	policy := ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	manager, fakeDocker, mockCadvisor := newRealImageManager(policy)

	// Expect 95% usage and most of it gets freed.
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiV2.FsInfo{
		Usage:    950,
		Capacity: 1000,
	}, nil)
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 450),
	}

	assert.NoError(t, manager.GarbageCollect())
}

func TestGarbageCollectNotEnoughFreed(t *testing.T) {
	policy := ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	manager, fakeDocker, mockCadvisor := newRealImageManager(policy)

	// Expect 95% usage and little of it gets freed.
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiV2.FsInfo{
		Usage:    950,
		Capacity: 1000,
	}, nil)
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 50),
	}

	assert.NotNil(t, manager.GarbageCollect())
}
