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
	"sort"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/cadvisor"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

// Manages lifecycle of all images.
//
// Implementation is thread-safe.
type imageManager interface {
	// Applies the garbage collection policy. Errors include being unable to free
	// enough space as per the garbage collection policy.
	GarbageCollect() error

	// TODO(vmarmol): Have this subsume pulls as well.
}

// A policy for garbage collecting images. Policy defines an allowed band in
// which garbage collection will be run.
type ImageGCPolicy struct {
	// Any usage above this threshold will always trigger garbage collection.
	// This is the highest usage we will allow.
	HighThresholdPercent int

	// Any usage below this threshold will never trigger garbage collection.
	// This is the lowest threshold we will try to garbage collect to.
	LowThresholdPercent int
}

type realImageManager struct {
	// Connection to the Docker daemon.
	dockerClient dockertools.DockerInterface

	// Records of images and their use.
	imageRecords     map[string]*imageRecord
	imageRecordsLock sync.Mutex

	// The image garbage collection policy in use.
	policy ImageGCPolicy

	// cAdvisor instance.
	cadvisor cadvisor.Interface
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

func newImageManager(dockerClient dockertools.DockerInterface, cadvisorInterface cadvisor.Interface, policy ImageGCPolicy) (imageManager, error) {
	// Validate policy.
	if policy.HighThresholdPercent < 0 || policy.HighThresholdPercent > 100 {
		return nil, fmt.Errorf("invalid HighThresholdPercent %d, must be in range [0-100]", policy.HighThresholdPercent)
	}
	if policy.LowThresholdPercent < 0 || policy.LowThresholdPercent > 100 {
		return nil, fmt.Errorf("invalid LowThresholdPercent %d, must be in range [0-100]", policy.LowThresholdPercent)
	}
	im := &realImageManager{
		dockerClient: dockerClient,
		policy:       policy,
		imageRecords: make(map[string]*imageRecord),
		cadvisor:     cadvisorInterface,
	}

	err := im.start()
	if err != nil {
		return nil, fmt.Errorf("failed to start image manager: %v", err)
	}

	return im, nil
}

func (self *realImageManager) start() error {
	// Initial detection make detected time "unknown" in the past.
	var zero time.Time
	err := self.detectImages(zero)
	if err != nil {
		return err
	}

	go util.Forever(func() {
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

func (self *realImageManager) GarbageCollect() error {
	// Get disk usage on disk holding images.
	fsInfo, err := self.cadvisor.DockerImagesFsInfo()
	if err != nil {
		return err
	}
	usage := int64(fsInfo.Usage)
	capacity := int64(fsInfo.Capacity)

	// Check valid capacity.
	if capacity == 0 {
		// TODO(vmarmol): Surface event.
		return fmt.Errorf("invalid capacity %d on device %q at mount point %q", capacity, fsInfo.Device, fsInfo.Mountpoint)
	}

	// If over the max threshold, free enough to place us at the lower threshold.
	usagePercent := int(usage * 100 / capacity)
	if usagePercent >= self.policy.HighThresholdPercent {
		amountToFree := usage - (int64(self.policy.LowThresholdPercent) * capacity / 100)
		glog.Infof("[ImageManager]: Disk usage on %q (%s) is at %d%% which is over the high threshold (%d%%). Trying to free %d bytes", fsInfo.Device, fsInfo.Mountpoint, usagePercent, self.policy.HighThresholdPercent, amountToFree)
		freed, err := self.freeSpace(amountToFree)
		if err != nil {
			return err
		}

		if freed < amountToFree {
			// TODO(vmarmol): Surface event.
			return fmt.Errorf("failed to garbage collect required amount of images. Wanted to free %d, but freed %d", amountToFree, freed)
		}
	}

	return nil
}

// Tries to free bytesToFree worth of images on the disk.
//
// Returns the number of bytes free and an error if any occured. The number of
// bytes freed is always returned.
// Note that error may be nil and the number of bytes free may be less
// than bytesToFree.
func (self *realImageManager) freeSpace(bytesToFree int64) (int64, error) {
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
		glog.Infof("[ImageManager]: Removing image %q to free %d bytes", image.id, image.size)
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
