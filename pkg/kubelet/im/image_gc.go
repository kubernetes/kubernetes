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

package im

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	defaultGCAge = time.Minute * 5
)

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

type realImageGC struct {
	// Container runtime
	runtime container.Runtime

	// Records of images and their use.
	imageRecords     map[string]*imageRecord
	imageRecordsLock sync.Mutex

	// The image garbage collection policy in use.
	policy ImageGCPolicy

	// Minimum age at which a image can be garbage collected, zero for no limit.
	// TODO(mqliang): move it to ImageGCPolicy and make it configurable
	minAge time.Duration

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
	firstDetected time.Time

	// Time when this image was last detected.
	lastDetected time.Time

	// Time when we last saw this image being used.
	lastUsed time.Time

	// Size of the image in bytes.
	size int64
}

func NewImageGC(runtime container.Runtime, cadvisorInterface cadvisor.Interface, recorder record.EventRecorder, nodeRef *api.ObjectReference, policy ImageGCPolicy) (ImageGC, error) {
	// Validate policy.
	if policy.HighThresholdPercent < 0 || policy.HighThresholdPercent > 100 {
		return nil, fmt.Errorf("invalid HighThresholdPercent %d, must be in range [0-100]", policy.HighThresholdPercent)
	}
	if policy.LowThresholdPercent < 0 || policy.LowThresholdPercent > 100 {
		return nil, fmt.Errorf("invalid LowThresholdPercent %d, must be in range [0-100]", policy.LowThresholdPercent)
	}
	if policy.LowThresholdPercent > policy.HighThresholdPercent {
		return nil, fmt.Errorf("LowThresholdPercent %d can not be higher than HighThresholdPercent %d", policy.LowThresholdPercent, policy.HighThresholdPercent)
	}

	im := &realImageGC{
		runtime:      runtime,
		policy:       policy,
		minAge:       defaultGCAge,
		imageRecords: make(map[string]*imageRecord),
		cadvisor:     cadvisorInterface,
		recorder:     recorder,
		nodeRef:      nodeRef,
	}

	return im, nil
}

func (im *realImageGC) GarbageCollect() error {
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
		im.recorder.Eventf(im.nodeRef, api.EventTypeWarning, container.InvalidDiskCapacity, err.Error())
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
			im.recorder.Eventf(im.nodeRef, api.EventTypeWarning, container.FreeDiskSpaceFailed, err.Error())
			return err
		}
	}

	return nil
}

func (im *realImageGC) detectImages(detectTime time.Time) error {
	images, err := im.runtime.ListImages()
	if err != nil {
		return err
	}
	pods, err := im.runtime.GetPods(true)
	if err != nil {
		return err
	}

	// Make a set of images in use by containers.
	imagesInUse := sets.NewString()
	for _, pod := range pods {
		for _, container := range pod.Containers {
			imagesInUse.Insert(container.Image)
		}
	}

	// Add new images and record those being used.
	now := time.Now()
	currentImages := sets.NewString()
	im.imageRecordsLock.Lock()
	defer im.imageRecordsLock.Unlock()
	for _, image := range images {
		currentImages.Insert(image.ID)

		// New image, set its first detected time.
		if _, ok := im.imageRecords[image.ID]; !ok {
			im.imageRecords[image.ID] = &imageRecord{
				firstDetected: detectTime,
			}
		}

		im.imageRecords[image.ID].lastDetected = detectTime

		// Set last used time to now if the image is being used.
		if isImageUsed(image, imagesInUse) {
			im.imageRecords[image.ID].lastUsed = now
		}

		im.imageRecords[image.ID].size = image.Size
	}

	// Remove old images from our records.
	for image := range im.imageRecords {
		if !currentImages.Has(image) {
			delete(im.imageRecords, image)
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
func (im *realImageGC) freeSpace(bytesToFree int64) (int64, error) {
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

		// Avoid garbage collect the image if the image is not old enough.
		// In such a case, the image may has just been pulled down, and will be used by a container right away.
		if image.firstDetected.Add(im.minAge).After(startTime) {
			continue
		}

		// Remove image. Continue despite errors.
		glog.Infof("[ImageManager]: Removing image %q to free %d bytes", image.id, image.size)
		err := im.runtime.RemoveImage(container.ImageSpec{Image: image.id})
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
		return ev[i].firstDetected.Before(ev[j].firstDetected)
	} else {
		return ev[i].lastUsed.Before(ev[j].lastUsed)
	}
}

func isImageUsed(image container.Image, imagesInUse sets.String) bool {
	// Check the image ID and all the RepoTags.
	if _, ok := imagesInUse[image.ID]; ok {
		return true
	}
	for _, tag := range image.Tags {
		if _, ok := imagesInUse[tag]; ok {
			return true
		}
	}
	return false
}
