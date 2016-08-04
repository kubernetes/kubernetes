/*
Copyright 2016 The Kubernetes Authors.

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
	"math"
	"sort"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
)

// Information about the images we track.
type imageRecord struct {
	// Time when this image was first detected.
	firstDetected time.Time

	// Time when we last saw this image being used.
	lastUsed time.Time

	// Size of the image in bytes.
	size int64
}

// imageManager provides the functionalities for image pulling.
type imageManager struct {
	runtime kubecontainer.Runtime
	backOff *flowcontrol.Backoff

	// It will check the presence of the image, and report the 'image pulling', image pulled' events correspondingly.
	puller imagePuller

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

	// Track initialization
	initialized bool
}

var _ ImageManager = &imageManager{}

func NewImageManager(recorder record.EventRecorder, runtime kubecontainer.Runtime, imageBackOff *flowcontrol.Backoff, serialized bool,
	cadvisorInterface cadvisor.Interface, nodeRef *api.ObjectReference, policy ImageGCPolicy) (ImageManager, error) {
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

	var puller imagePuller
	if serialized {
		puller = newSerialImagePuller(runtime)
	} else {
		puller = newParallelImagePuller(runtime)
	}
	return &imageManager{
		runtime:      runtime,
		backOff:      imageBackOff,
		puller:       puller,
		imageRecords: make(map[string]*imageRecord),
		cadvisor:     cadvisorInterface,
		recorder:     recorder,
		nodeRef:      nodeRef,
		initialized:  false,
	}, nil
}

// shouldPullImage returns whether we should pull an image according to
// the presence and pull policy of the image.
func shouldPullImage(container *api.Container, imagePresent bool) bool {
	if container.ImagePullPolicy == api.PullNever {
		return false
	}

	if container.ImagePullPolicy == api.PullAlways ||
		(container.ImagePullPolicy == api.PullIfNotPresent && (!imagePresent)) {
		return true
	}

	return false
}

// records an event using ref, event msg.  log to glog using prefix, msg, logFn
func (m *imageManager) logIt(ref *api.ObjectReference, eventtype, event, prefix, msg string, logFn func(args ...interface{})) {
	if ref != nil {
		m.recorder.Event(ref, eventtype, event, msg)
	} else {
		logFn(fmt.Sprint(prefix, " ", msg))
	}
}

// EnsureImageExists pulls the image for the specified pod and container.
func (m *imageManager) EnsureImageExists(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) (error, string) {
	logPrefix := fmt.Sprintf("%s/%s", pod.Name, container.Image)
	ref, err := kubecontainer.GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}

	spec := kubecontainer.ImageSpec{Image: container.Image}
	present, err := m.runtime.IsImagePresent(spec)
	if err != nil {
		msg := fmt.Sprintf("Failed to inspect image %q: %v", container.Image, err)
		m.logIt(ref, api.EventTypeWarning, events.FailedToInspectImage, logPrefix, msg, glog.Warning)
		return ErrImageInspect, msg
	}

	if !shouldPullImage(container, present) {
		if present {
			msg := fmt.Sprintf("Container image %q already present on machine", container.Image)
			m.logIt(ref, api.EventTypeNormal, events.PulledImage, logPrefix, msg, glog.Info)
			return nil, ""
		} else {
			msg := fmt.Sprintf("Container image %q is not present with pull policy of Never", container.Image)
			m.logIt(ref, api.EventTypeWarning, events.ErrImageNeverPullPolicy, logPrefix, msg, glog.Warning)
			return ErrImageNeverPull, msg
		}
	}

	backOffKey := fmt.Sprintf("%s_%s", pod.UID, container.Image)
	if m.backOff.IsInBackOffSinceUpdate(backOffKey, m.backOff.Clock.Now()) {
		msg := fmt.Sprintf("Back-off pulling image %q", container.Image)
		m.logIt(ref, api.EventTypeNormal, events.BackOffPullImage, logPrefix, msg, glog.Info)
		return ErrImagePullBackOff, msg
	}
	m.logIt(ref, api.EventTypeNormal, events.PullingImage, logPrefix, fmt.Sprintf("pulling image %q", container.Image), glog.Info)
	errChan := make(chan error)
	m.puller.pullImage(spec, pullSecrets, errChan)
	if err := <-errChan; err != nil {
		m.logIt(ref, api.EventTypeWarning, events.FailedToPullImage, logPrefix, fmt.Sprintf("Failed to pull image %q: %v", container.Image, err), glog.Warning)
		m.backOff.Next(backOffKey, m.backOff.Clock.Now())
		if err == RegistryUnavailable {
			msg := fmt.Sprintf("image pull failed for %s because the registry is unavailable.", container.Image)
			return err, msg
		} else {
			return ErrImagePull, err.Error()
		}
	}
	m.logIt(ref, api.EventTypeNormal, events.PulledImage, logPrefix, fmt.Sprintf("Successfully pulled image %q", container.Image), glog.Info)
	m.backOff.GC()
	return nil, ""
}

func (im *imageManager) Start() error {
	go wait.Until(func() {
		// Initial detection make detected time "unknown" in the past.
		var ts time.Time
		if im.initialized {
			ts = time.Now()
		}
		err := im.detectImages(ts)
		if err != nil {
			glog.Warningf("[ImageManager] Failed to monitor images: %v", err)
		} else {
			im.initialized = true
		}
	}, 5*time.Minute, wait.NeverStop)

	return nil
}

// Get a list of images on this node
func (im *imageManager) GetImageList() ([]kubecontainer.Image, error) {
	images, err := im.runtime.ListImages()
	if err != nil {
		return nil, err
	}
	return images, nil
}

func (im *imageManager) detectImages(detectTime time.Time) error {
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
			glog.V(5).Infof("Pod %s/%s, container %s uses image %s", pod.Namespace, pod.Name, container.Name, container.Image)
			imagesInUse.Insert(container.ImageID)
		}
	}

	// Add new images and record those being used.
	now := time.Now()
	currentImages := sets.NewString()
	im.imageRecordsLock.Lock()
	defer im.imageRecordsLock.Unlock()
	for _, image := range images {
		glog.V(5).Infof("Adding image ID %s to currentImages", image.ID)
		currentImages.Insert(image.ID)

		// New image, set it as detected now.
		if _, ok := im.imageRecords[image.ID]; !ok {
			glog.V(5).Infof("Image ID %s is new", image.ID)
			im.imageRecords[image.ID] = &imageRecord{
				firstDetected: detectTime,
			}
		}

		// Set last used time to now if the image is being used.
		if isImageUsed(image, imagesInUse) {
			glog.V(5).Infof("Setting Image ID %s lastUsed to %v", image.ID, now)
			im.imageRecords[image.ID].lastUsed = now
		}

		glog.V(5).Infof("Image ID %s has size %d", image.ID, image.Size)
		im.imageRecords[image.ID].size = image.Size
	}

	// Remove old images from our records.
	for image := range im.imageRecords {
		if !currentImages.Has(image) {
			glog.V(5).Infof("Image ID %s is no longer present; removing from imageRecords", image)
			delete(im.imageRecords, image)
		}
	}

	return nil
}

func (im *imageManager) GarbageCollect() error {
	// Get disk usage on disk holding images.
	fsInfo, err := im.cadvisor.ImagesFsInfo()
	if err != nil {
		return err
	}
	capacity := int64(fsInfo.Capacity)
	available := int64(fsInfo.Available)
	if available > capacity {
		glog.Warningf("available %d is larger than capacity %d", available, capacity)
		available = capacity
	}

	// Check valid capacity.
	if capacity == 0 {
		err := fmt.Errorf("invalid capacity %d on device %q at mount point %q", capacity, fsInfo.Device, fsInfo.Mountpoint)
		im.recorder.Eventf(im.nodeRef, api.EventTypeWarning, events.InvalidDiskCapacity, err.Error())
		return err
	}

	// If over the max threshold, free enough to place us at the lower threshold.
	usagePercent := 100 - int(available*100/capacity)
	if usagePercent >= im.policy.HighThresholdPercent {
		amountToFree := capacity*int64(100-im.policy.LowThresholdPercent)/100 - available
		glog.Infof("[ImageManager]: Disk usage on %q (%s) is at %d%% which is over the high threshold (%d%%). Trying to free %d bytes", fsInfo.Device, fsInfo.Mountpoint, usagePercent, im.policy.HighThresholdPercent, amountToFree)
		freed, err := im.freeSpace(amountToFree, time.Now())
		if err != nil {
			return err
		}

		if freed < amountToFree {
			err := fmt.Errorf("failed to garbage collect required amount of images. Wanted to free %d, but freed %d", amountToFree, freed)
			im.recorder.Eventf(im.nodeRef, api.EventTypeWarning, events.FreeDiskSpaceFailed, err.Error())
			return err
		}
	}

	return nil
}

func (im *imageManager) DeleteUnusedImages() (int64, error) {
	return im.freeSpace(math.MaxInt64, time.Now())
}

// Tries to free bytesToFree worth of images on the disk.
//
// Returns the number of bytes free and an error if any occurred. The number of
// bytes freed is always returned.
// Note that error may be nil and the number of bytes free may be less
// than bytesToFree.
func (im *imageManager) freeSpace(bytesToFree int64, freeTime time.Time) (int64, error) {
	err := im.detectImages(freeTime)
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
	var deletionErrors []error
	spaceFreed := int64(0)
	for _, image := range images {
		glog.V(5).Infof("Evaluating image ID %s for possible garbage collection", image.id)
		// Images that are currently in used were given a newer lastUsed.
		if image.lastUsed.Equal(freeTime) || image.lastUsed.After(freeTime) {
			glog.V(5).Infof("Image ID %s has lastUsed=%v which is >= freeTime=%v, not eligible for garbage collection", image.id, image.lastUsed, freeTime)
			break
		}

		// Avoid garbage collect the image if the image is not old enough.
		// In such a case, the image may have just been pulled down, and will be used by a container right away.

		if freeTime.Sub(image.firstDetected) < im.policy.MinAge {
			glog.V(5).Infof("Image ID %s has age %v which is less than the policy's minAge of %v, not eligible for garbage collection", image.id, freeTime.Sub(image.firstDetected), im.policy.MinAge)
			continue
		}

		// Remove image. Continue despite errors.
		glog.Infof("[ImageManager]: Removing image %q to free %d bytes", image.id, image.size)
		err := im.runtime.RemoveImage(kubecontainer.ImageSpec{Image: image.id})
		if err != nil {
			deletionErrors = append(deletionErrors, err)
			continue
		}
		delete(im.imageRecords, image.id)
		spaceFreed += image.size

		if spaceFreed >= bytesToFree {
			break
		}
	}

	if len(deletionErrors) > 0 {
		return spaceFreed, fmt.Errorf("wanted to free %d, but freed %d space with errors in image deletion: %v", bytesToFree, spaceFreed, errors.NewAggregate(deletionErrors))
	}
	return spaceFreed, nil
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

func isImageUsed(image kubecontainer.Image, imagesInUse sets.String) bool {
	// Check the image ID.
	if _, ok := imagesInUse[image.ID]; ok {
		return true
	}
	return false
}
