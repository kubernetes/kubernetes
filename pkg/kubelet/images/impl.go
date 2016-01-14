/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"sync"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	runtime "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

type imageManager struct {
	recorder record.EventRecorder
	backOff  *flowcontrol.Backoff
	runtime  runtime.Runtime
	puller   imagePuller
	images   map[string]uint64
	sync.Mutex
}

var _ ImageManager = &imageManager{}

// TODO: Add support for tracking images with PullNever policy and avoid garbage collecting them.
func NewImageManager(recorder record.EventRecorder, runtime runtime.Runtime, backOff *flowcontrol.Backoff, serialized bool) (ImageManager, error) {
	var puller imagePuller
	if serialized {
		puller = newSerialImagePuller(runtime)
	} else {
		puller = newParallelImagePuller(runtime)
	}
	im := &imageManager{
		recorder: recorder,
		runtime:  runtime,
		backOff:  backOff,
		puller:   puller,
		images:   make(map[string]uint64),
	}
	return im, im.detectExistingImages()
}

func (im *imageManager) detectExistingImages() error {
	images, err := im.runtime.ListImages()
	if err != nil {
		return err
	}
	for _, image := range images {
		for _, tag := range image.RepoTags {
			// register images using their logical names since that is what is used by users.
			im.images[tag] = 0
		}
	}
	allPods, err := im.runtime.GetPods(true)
	if err != nil {
		return err
	}
	for _, pod := range allPods {
		for _, container := range pod.Containers {
			im.incrementImageUsage(container.Image)
		}
	}
	return nil
}

func (im *imageManager) DecImageUsage(container *api.Container) error {
	im.Lock()
	defer im.Unlock()
	usage, exists := im.images[container.Image]
	if !exists {
		return fmt.Errorf("Critical Error: Image %q unknown to Image Manager", container.Image)
	}
	im.images[container.Image] = usage - 1
	return nil
}

func (im *imageManager) DeleteUnusedImages() {
	im.Lock()
	defer im.Unlock()

	for image, usage := range im.images {
		if usage > 0 {
			continue
		}
		err := im.runtime.RemoveImage(runtime.ImageSpec{image})
		if err != nil {
			glog.V(2).Infof("failed to remove unused image %q", image)
		}
	}
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
func (im *imageManager) logIt(ref *api.ObjectReference, eventtype, event, prefix, msg string, logFn func(args ...interface{})) {
	if ref != nil {
		im.recorder.Event(ref, eventtype, event, msg)
	} else {
		logFn(fmt.Sprint(prefix, " ", msg))
	}
}

func (im *imageManager) incrementImageUsage(image string) {
	im.Lock()
	defer im.Unlock()
	usage := im.images[image]
	im.images[image] = usage + 1
}

func (im *imageManager) imageExists(spec ImageSpec) bool {
	im.Lock()
	defer im.Unlock()
	_, exists := im.images[spec.Image]
	return exists
}

// PullImage pulls the image for the specified pod and runtime.
func (im *imageManager) EnsureImageExists(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) error {
	logPrefix := fmt.Sprintf("%s/%s", pod.Name, container.Image)
	ref, err := runtime.GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}

	spec := ImageSpec{container.Image, pullSecrets}

	exists := im.imageExists(spec)
	if !shouldPullImage(container, exists) {
		if exists {
			msg := fmt.Sprintf("Container image %q already present on machine", container.Image)
			im.logIt(ref, api.EventTypeNormal, "Pulled", logPrefix, msg, glog.Info)
			im.incrementImageUsage(spec.Image)
			return nil
		} else {
			msg := fmt.Sprintf("Container image %q is not present with pull policy of Never", container.Image)
			im.logIt(ref, api.EventTypeWarning, events.ErrImageNeverPullPolicy, logPrefix, msg, glog.Warning)
			return ErrImageNeverPull
		}
	}

	backOffKey := fmt.Sprintf("%s_%s", pod.UID, container.Image)
	if im.backOff.IsInBackOffSinceUpdate(backOffKey, im.backOff.Clock.Now()) {
		msg := fmt.Sprintf("Back-off pulling image %q", container.Image)
		im.logIt(ref, api.EventTypeNormal, events.BackOffPullImage, logPrefix, msg, glog.Info)
		return ErrImagePullBackOff
	}
	prePullChan := make(chan struct{})
	errChan := make(chan error)
	im.puller.pullImage(spec, prePullChan, errChan)
	// Block until a message has been posted to prePullChan.
	_ = <-prePullChan
	im.logIt(ref, api.EventTypeNormal, "Pulling", logPrefix, fmt.Sprintf("pulling image %q", container.Image), glog.Info)
	// Block until image puller has posted the result of the pull operation.
	if err = <-errChan; err != nil {
		im.logIt(ref, api.EventTypeWarning, events.FailedToPullImage, logPrefix, fmt.Sprintf("Failed to pull image %q: %v", container.Image, err), glog.Warning)
		im.backOff.Next(backOffKey, im.backOff.Clock.Now())
		if err == RegistryUnavailable {
			msg := fmt.Sprintf("image pull failed for %s because the registry is unavailable.", container.Image)
			im.logIt(ref, api.EventTypeNormal, events.FailedToPullImage, logPrefix, msg, glog.Info)
			return ErrImagePull
		} else {
			return ErrImagePull
		}
	}
	im.logIt(ref, api.EventTypeNormal, events.PulledImage, logPrefix, fmt.Sprintf("Successfully pulled image %q", container.Image), glog.Info)
	im.backOff.DeleteEntry(backOffKey)
	im.incrementImageUsage(container.Image)
	im.backOff.GC()

	return nil
}
