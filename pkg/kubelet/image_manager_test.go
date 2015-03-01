/*
Copyright 2015 Google Inc. All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	docker "github.com/fsouza/go-dockerclient"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var zero time.Time

func newRealImageManager(dockerClient dockertools.DockerInterface) *realImageManager {
	return newImageManager(dockerClient).(*realImageManager)
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
	fakeDocker := &dockertools.FakeDockerClient{
		Images: []docker.APIImages{
			makeImage(0, 1024),
			makeImage(1, 2048),
		},
		ContainerList: []docker.APIContainers{
			makeContainer(1),
		},
	}
	manager := newRealImageManager(fakeDocker)

	startTime := time.Now().Add(-time.Millisecond)
	err := manager.detectImages(zero)
	assert := assert.New(t)
	require.Nil(t, err)
	assert.Len(manager.imageRecords, 2)
	noContainer, ok := manager.imageRecords[imageName(0)]
	require.True(t, ok)
	assert.Equal(zero, noContainer.detected)
	assert.Equal(zero, noContainer.lastUsed)
	withContainer, ok := manager.imageRecords[imageName(1)]
	require.True(t, ok)
	assert.Equal(zero, withContainer.detected)
	assert.True(withContainer.lastUsed.After(startTime))
}

func TestDetectImagesWithNewImage(t *testing.T) {
	// Just one image initially.
	fakeDocker := &dockertools.FakeDockerClient{
		Images: []docker.APIImages{
			makeImage(0, 1024),
			makeImage(1, 2048),
		},
		ContainerList: []docker.APIContainers{
			makeContainer(1),
		},
	}
	manager := newRealImageManager(fakeDocker)

	err := manager.detectImages(zero)
	assert := assert.New(t)
	require.Nil(t, err)
	assert.Len(manager.imageRecords, 2)

	// Add a new image.
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 1024),
		makeImage(1, 1024),
		makeImage(2, 1024),
	}

	detectedTime := zero.Add(time.Second)
	startTime := time.Now().Add(-time.Millisecond)
	err = manager.detectImages(detectedTime)
	require.Nil(t, err)
	assert.Len(manager.imageRecords, 3)
	noContainer, ok := manager.imageRecords[imageName(0)]
	require.True(t, ok)
	assert.Equal(zero, noContainer.detected)
	assert.Equal(zero, noContainer.lastUsed)
	withContainer, ok := manager.imageRecords[imageName(1)]
	require.True(t, ok)
	assert.Equal(zero, withContainer.detected)
	assert.True(withContainer.lastUsed.After(startTime))
	newContainer, ok := manager.imageRecords[imageName(2)]
	require.True(t, ok)
	assert.Equal(detectedTime, newContainer.detected)
	assert.Equal(zero, noContainer.lastUsed)
}

func TestDetectImagesContainerStopped(t *testing.T) {
	fakeDocker := &dockertools.FakeDockerClient{
		Images: []docker.APIImages{
			makeImage(0, 1024),
			makeImage(1, 2048),
		},
		ContainerList: []docker.APIContainers{
			makeContainer(1),
		},
	}
	manager := newRealImageManager(fakeDocker)

	err := manager.detectImages(zero)
	assert := assert.New(t)
	require.Nil(t, err)
	assert.Len(manager.imageRecords, 2)
	withContainer, ok := manager.imageRecords[imageName(1)]
	require.True(t, ok)

	// Simulate container being stopped.
	fakeDocker.ContainerList = []docker.APIContainers{}
	err = manager.detectImages(time.Now())
	require.Nil(t, err)
	assert.Len(manager.imageRecords, 2)
	container1, ok := manager.imageRecords[imageName(0)]
	require.True(t, ok)
	assert.Equal(zero, container1.detected)
	assert.Equal(zero, container1.lastUsed)
	container2, ok := manager.imageRecords[imageName(1)]
	require.True(t, ok)
	assert.Equal(zero, container2.detected)
	assert.True(container2.lastUsed.Equal(withContainer.lastUsed))
}

func TestDetectImagesWithRemovedImages(t *testing.T) {
	fakeDocker := &dockertools.FakeDockerClient{
		Images: []docker.APIImages{
			makeImage(0, 1024),
			makeImage(1, 2048),
		},
		ContainerList: []docker.APIContainers{
			makeContainer(1),
		},
	}
	manager := newRealImageManager(fakeDocker)

	err := manager.detectImages(zero)
	assert := assert.New(t)
	require.Nil(t, err)
	assert.Len(manager.imageRecords, 2)

	// Simulate both images being removed.
	fakeDocker.Images = []docker.APIImages{}
	err = manager.detectImages(time.Now())
	require.Nil(t, err)
	assert.Len(manager.imageRecords, 0)
}

func TestFreeSpaceImagesInUseContainersAreIgnored(t *testing.T) {
	fakeDocker := &dockertools.FakeDockerClient{
		Images: []docker.APIImages{
			makeImage(0, 1024),
			makeImage(1, 2048),
		},
		ContainerList: []docker.APIContainers{
			makeContainer(1),
		},
		RemovedImages: util.NewStringSet(),
	}
	manager := newRealImageManager(fakeDocker)

	spaceFreed, err := manager.FreeSpace(2048)
	assert := assert.New(t)
	require.Nil(t, err)
	assert.Equal(1024, spaceFreed)
	assert.Len(fakeDocker.RemovedImages, 1)
	assert.True(fakeDocker.RemovedImages.Has(imageName(0)))
}

func TestFreeSpaceRemoveByLeastRecentlyUsed(t *testing.T) {
	fakeDocker := &dockertools.FakeDockerClient{
		Images: []docker.APIImages{
			makeImage(0, 1024),
			makeImage(1, 2048),
		},
		ContainerList: []docker.APIContainers{
			makeContainer(0),
			makeContainer(1),
		},
		RemovedImages: util.NewStringSet(),
	}
	manager := newRealImageManager(fakeDocker)

	// Make 1 be more recently used than 0.
	require.Nil(t, manager.detectImages(zero))
	fakeDocker.ContainerList = []docker.APIContainers{
		makeContainer(1),
	}
	require.Nil(t, manager.detectImages(time.Now()))
	fakeDocker.ContainerList = []docker.APIContainers{}
	require.Nil(t, manager.detectImages(time.Now()))
	require.Len(t, manager.imageRecords, 2)

	spaceFreed, err := manager.FreeSpace(1024)
	assert := assert.New(t)
	require.Nil(t, err)
	assert.Equal(1024, spaceFreed)
	assert.Len(fakeDocker.RemovedImages, 1)
	assert.True(fakeDocker.RemovedImages.Has(imageName(0)))
}

func TestFreeSpaceTiesBrokenByDetectedTime(t *testing.T) {
	fakeDocker := &dockertools.FakeDockerClient{
		Images: []docker.APIImages{
			makeImage(0, 1024),
		},
		ContainerList: []docker.APIContainers{
			makeContainer(0),
		},
		RemovedImages: util.NewStringSet(),
	}
	manager := newRealImageManager(fakeDocker)

	// Make 1 more recently detected but used at the same time as 0.
	require.Nil(t, manager.detectImages(zero))
	fakeDocker.Images = []docker.APIImages{
		makeImage(0, 1024),
		makeImage(1, 2048),
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		makeContainer(0),
		makeContainer(1),
	}
	require.Nil(t, manager.detectImages(time.Now()))
	fakeDocker.ContainerList = []docker.APIContainers{}
	require.Nil(t, manager.detectImages(time.Now()))
	require.Len(t, manager.imageRecords, 2)

	spaceFreed, err := manager.FreeSpace(1024)
	assert := assert.New(t)
	require.Nil(t, err)
	assert.Equal(1024, spaceFreed)
	assert.Len(fakeDocker.RemovedImages, 1)
	assert.True(fakeDocker.RemovedImages.Has(imageName(0)))
}

func TestFreeSpaceImagesAlsoDoesLookupByRepoTags(t *testing.T) {
	fakeDocker := &dockertools.FakeDockerClient{
		Images: []docker.APIImages{
			makeImage(0, 1024),
			{
				ID:          "5678",
				RepoTags:    []string{"potato", "salad"},
				VirtualSize: 2048,
			},
		},
		ContainerList: []docker.APIContainers{
			{
				ID:    "c5678",
				Image: "salad",
			},
		},
		RemovedImages: util.NewStringSet(),
	}
	manager := newRealImageManager(fakeDocker)

	spaceFreed, err := manager.FreeSpace(1024)
	assert := assert.New(t)
	require.Nil(t, err)
	assert.Equal(1024, spaceFreed)
	assert.Len(fakeDocker.RemovedImages, 1)
	assert.True(fakeDocker.RemovedImages.Has(imageName(0)))
}
