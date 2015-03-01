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
	"sort"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

// Manages lifecycle of all images.
//
// Class is thread-safe.
type imageManager interface {
	// Starts the image manager.
	Start() error

	// Tries to free bytesToFree worth of images on the disk.
	//
	// Returns the number of bytes free and an error if any occured. The number of
	// bytes freed is always returned.
	// Note that error may be nil and the number of bytes free may be less
	// than bytesToFree.
	FreeSpace(bytesToFree int64) (int64, error)

	// TODO(vmarmol): Have this subsume pulls as well.
}

type realImageManager struct {
	// Connection to the Docker daemon.
	dockerClient dockertools.DockerInterface

	// Records of images and their use.
	imageRecords map[string]*imageRecord

	// Lock for imageRecords.
	imageRecordsLock sync.Mutex
}

// Information about the images we track.
type imageRecord struct {
	// Time when this image was first detected.
	detected time.Time

	// Time when we last saw this image being used.
	lastUsed time.Time

	// Size of the image in bytes.
	size int64
}

func newImageManager(dockerClient dockertools.DockerInterface) imageManager {
	return &realImageManager{
		dockerClient: dockerClient,
		imageRecords: make(map[string]*imageRecord),
	}
}

func (self *realImageManager) Start() error {
	// Initial detection make detected time "unknown" in the past.
	var zero time.Time
	err := self.detectImages(zero)
	if err != nil {
		return err
	}

	util.Forever(func() {
		err := self.detectImages(time.Now())
		if err != nil {
			glog.Warningf("[ImageManager] Failed to monitor images: %v", err)
		}
	}, 5*time.Minute)

	return nil
}

func (self *realImageManager) detectImages(detected time.Time) error {
	images, err := self.dockerClient.ListImages(docker.ListImagesOptions{})
	if err != nil {
		return err
	}
	containers, err := self.dockerClient.ListContainers(docker.ListContainersOptions{
		All: true,
	})
	if err != nil {
		return err
	}

	// Make a set of images in use by containers.
	imagesInUse := util.NewStringSet()
	for _, container := range containers {
		imagesInUse.Insert(container.Image)
	}

	// Add new images and record those being used.
	now := time.Now()
	currentImages := util.NewStringSet()
	self.imageRecordsLock.Lock()
	defer self.imageRecordsLock.Unlock()
	for _, image := range images {
		currentImages.Insert(image.ID)

		// New image, set it as detected now.
		if _, ok := self.imageRecords[image.ID]; !ok {
			self.imageRecords[image.ID] = &imageRecord{
				detected: detected,
			}
		}

		// Set last used time to now if the image is being used.
		if isImageUsed(&image, imagesInUse) {
			self.imageRecords[image.ID].lastUsed = now
		}

		self.imageRecords[image.ID].size = image.VirtualSize
	}

	// Remove old images from our records.
	for image := range self.imageRecords {
		if !currentImages.Has(image) {
			delete(self.imageRecords, image)
		}
	}

	return nil
}

func (self *realImageManager) FreeSpace(bytesToFree int64) (int64, error) {
	startTime := time.Now()
	err := self.detectImages(startTime)
	if err != nil {
		return 0, err
	}

	self.imageRecordsLock.Lock()
	defer self.imageRecordsLock.Unlock()

	// Get all images in eviction order.
	images := make([]evictionInfo, 0, len(self.imageRecords))
	for image, record := range self.imageRecords {
		images = append(images, evictionInfo{
			id:          image,
			imageRecord: *record,
		})
	}
	sort.Sort(byLastUsedAndDetected(images))

	// Delete unused images until we've freed up enough space.
	var lastErr error
	spaceFreed := int64(0)
	for _, image := range images {
		// Images that are currently in used were given a newer lastUsed.
		if image.lastUsed.After(startTime) {
			break
		}

		// Remove image. Continue despite errors.
		err := self.dockerClient.RemoveImage(image.id)
		if err != nil {
			lastErr = err
			continue
		}
		delete(self.imageRecords, image.id)
		spaceFreed += image.size

		if spaceFreed >= bytesToFree {
			break
		}
	}

	return spaceFreed, lastErr
}

type evictionInfo struct {
	id string
	imageRecord
}

type byLastUsedAndDetected []evictionInfo

func (self byLastUsedAndDetected) Len() int      { return len(self) }
func (self byLastUsedAndDetected) Swap(i, j int) { self[i], self[j] = self[j], self[i] }
func (self byLastUsedAndDetected) Less(i, j int) bool {
	// Sort by last used, break ties by detected.
	if self[i].lastUsed.Equal(self[j].lastUsed) {
		return self[i].detected.Before(self[j].detected)
	} else {
		return self[i].lastUsed.Before(self[j].lastUsed)
	}
}

func isImageUsed(image *docker.APIImages, imagesInUse util.StringSet) bool {
	// Check the image ID and all the RepoTags.
	if _, ok := imagesInUse[image.ID]; ok {
		return true
	}
	for _, tag := range image.RepoTags {
		if _, ok := imagesInUse[tag]; ok {
			return true
		}
	}
	return false
}
