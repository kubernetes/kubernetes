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
	"sort"
	"sync"
	"time"

	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
)

// Manages lifecycle of all images.
//
// Implementation is thread-safe.
type imageManager interface {
	// Applies the garbage collection policy. Errors include being unable to free
	// enough space as per the garbage collection policy.
	GarbageCollect() error

	// Start async garbage collection of images.
	Start() error

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

	// Recorder for Kubernetes events.
	recorder record.EventRecorder

	// Reference to this node.
	nodeRef *api.ObjectReference
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

func newImageManager(dockerClient dockertools.DockerInterface, cadvisorInterface cadvisor.Interface, recorder record.EventRecorder, nodeRef *api.ObjectReference, policy ImageGCPolicy) (imageManager, error) {
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
		recorder:     recorder,
		nodeRef:      nodeRef,
	}

	return im, nil
}

func (im *realImageManager) Start() error {
	// Initial detection make detected time "unknown" in the past.
	var zero time.Time
	err := im.detectImages(zero)
	if err != nil {
		return err
	}

	go util.Until(func() {
		err := im.detectImages(time.Now())
		if err != nil {
			glog.Warningf("[ImageManager] Failed to monitor images: %v", err)
		}
	}, 5*time.Minute, util.NeverStop)

	return nil
}

func (im *realImageManager) detectImages(detected time.Time) error {
	images, err := im.dockerClient.ListImages(docker.ListImagesOptions{})
	if err != nil {
		return err
	}
	containers, err := im.dockerClient.ListContainers(docker.ListContainersOptions{
		All: true,
	})
	if err != nil {
		return err
	}

	// Make a set of images in use by containers.
	imagesInUse := sets.NewString()
	for _, container := range containers {
		imagesInUse.Insert(container.Image)
	}

	// Add new images and record those being used.
	now := time.Now()
	currentImages := sets.NewString()
	im.imageRecordsLock.Lock()
	defer im.imageRecordsLock.Unlock()
	for _, image := range images {
		currentImages.Insert(image.ID)

		// New image, set it as detected now.
		if _, ok := im.imageRecords[image.ID]; !ok {
			im.imageRecords[image.ID] = &imageRecord{
				detected: detected,
			}
		}

		// Set last used time to now if the image is being used.
		if isImageUsed(&image, imagesInUse) {
			im.imageRecords[image.ID].lastUsed = now
		}

		im.imageRecords[image.ID].size = image.VirtualSize
	}

	// Remove old images from our records.
	for image := range im.imageRecords {
		if !currentImages.Has(image) {
			delete(im.imageRecords, image)
		}
	}

	return nil
}

func (im *realImageManager) GarbageCollect() error {
	// Get disk usage on disk holding images.
	fsInfo, err := im.cadvisor.DockerImagesFsInfo()
	if err != nil {
		return err
	}
	usage := int64(fsInfo.Usage)
	capacity := int64(fsInfo.Capacity)

	// Check valid capacity.
	if capacity == 0 {
		err := fmt.Errorf("invalid capacity %d on device %q at mount point %q", capacity, fsInfo.Device, fsInfo.Mountpoint)
		im.recorder.Eventf(im.nodeRef, "InvalidDiskCapacity", err.Error())
		return err
	}

	// If over the max threshold, free enough to place us at the lower threshold.
	usagePercent := int(usage * 100 / capacity)
	if usagePercent >= im.policy.HighThresholdPercent {
		amountToFree := usage - (int64(im.policy.LowThresholdPercent) * capacity / 100)
		glog.Infof("[ImageManager]: Disk usage on %q (%s) is at %d%% which is over the high threshold (%d%%). Trying to free %d bytes", fsInfo.Device, fsInfo.Mountpoint, usagePercent, im.policy.HighThresholdPercent, amountToFree)
		freed, err := im.freeSpace(amountToFree)
		if err != nil {
			return err
		}

		if freed < amountToFree {
			err := fmt.Errorf("failed to garbage collect required amount of images. Wanted to free %d, but freed %d", amountToFree, freed)
			im.recorder.Eventf(im.nodeRef, "FreeDiskSpaceFailed", err.Error())
			return err
		}
	}

	return nil
}

// Tries to free bytesToFree worth of images on the disk.
//
// Returns the number of bytes free and an error if any occurred. The number of
// bytes freed is always returned.
// Note that error may be nil and the number of bytes free may be less
// than bytesToFree.
func (im *realImageManager) freeSpace(bytesToFree int64) (int64, error) {
	startTime := time.Now()
	err := im.detectImages(startTime)
	if err != nil {
		return 0, err
	}

	im.imageRecordsLock.Lock()
	defer im.imageRecordsLock.Unlock()

	// Get all images in eviction order.
	images := make([]evictionInfo, 0, len(im.imageRecords))
	for image, record := range im.imageRecords {
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
		err := im.dockerClient.RemoveImage(image.id)
		if err != nil {
			lastErr = err
			continue
		}
		delete(im.imageRecords, image.id)
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

func (ev byLastUsedAndDetected) Len() int      { return len(ev) }
func (ev byLastUsedAndDetected) Swap(i, j int) { ev[i], ev[j] = ev[j], ev[i] }
func (ev byLastUsedAndDetected) Less(i, j int) bool {
	// Sort by last used, break ties by detected.
	if ev[i].lastUsed.Equal(ev[j].lastUsed) {
		return ev[i].detected.Before(ev[j].detected)
	} else {
		return ev[i].lastUsed.Before(ev[j].lastUsed)
	}
}

func isImageUsed(image *docker.APIImages, imagesInUse sets.String) bool {
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
