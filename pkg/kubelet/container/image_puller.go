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

package container

import (
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/util"
)

// imagePuller pulls the image using Runtime.PullImage().
// It will check the presence of the image, and report the 'image pulling',
// 'image pulled' events correspondingly.
type imagePuller struct {
	recorder record.EventRecorder
	runtime  Runtime
	backOff  *util.Backoff
}

// NewImagePuller takes an event recorder and container runtime to create a
// image puller that wraps the container runtime's PullImage interface.
func NewImagePuller(recorder record.EventRecorder, runtime Runtime, imageBackOff *util.Backoff) ImagePuller {
	return &imagePuller{
		recorder: recorder,
		runtime:  runtime,
		backOff:  imageBackOff,
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
func (puller *imagePuller) logIt(ref *api.ObjectReference, event, prefix, msg string, logFn func(args ...interface{})) {
	if ref != nil {
		puller.recorder.Eventf(ref, event, msg)
	} else {
		logFn(fmt.Sprint(prefix, " ", msg))
	}
}

// PullImage pulls the image for the specified pod and container.
func (puller *imagePuller) PullImage(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) (error, string) {
	logPrefix := fmt.Sprintf("%s/%s", pod.Name, container.Image)
	ref, err := GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}
	spec := ImageSpec{container.Image}
	present, err := puller.runtime.IsImagePresent(spec)
	if err != nil {
		msg := fmt.Sprintf("Failed to inspect image %q: %v", container.Image, err)
		puller.logIt(ref, "Failed", logPrefix, msg, glog.Warning)
		return ErrImageInspect, msg
	}

	if !shouldPullImage(container, present) {
		if present {
			msg := fmt.Sprintf("Container image %q already present on machine", container.Image)
			puller.logIt(ref, "Pulled", logPrefix, msg, glog.Info)
			return nil, ""
		} else {
			msg := fmt.Sprintf("Container image %q is not present with pull policy of Never", container.Image)
			puller.logIt(ref, "ErrImageNeverPull", logPrefix, msg, glog.Warning)
			return ErrImageNeverPull, msg
		}
	}

	backOffKey := fmt.Sprintf("%s_%s", pod.Name, container.Image)
	if puller.backOff.IsInBackOffSinceUpdate(backOffKey, puller.backOff.Clock.Now()) {
		msg := fmt.Sprintf("Back-off pulling image %q", container.Image)
		puller.logIt(ref, "Back-off", logPrefix, msg, glog.Info)
		return ErrImagePullBackOff, msg
	}
	puller.logIt(ref, "Pulling", logPrefix, fmt.Sprintf("pulling image %q", container.Image), glog.Info)
	if err = puller.runtime.PullImage(spec, pullSecrets); err != nil {
		puller.logIt(ref, "Failed", logPrefix, fmt.Sprintf("Failed to pull image %q: %v", container.Image, err), glog.Warning)
		puller.backOff.Next(backOffKey, puller.backOff.Clock.Now())
		return ErrImagePull, err.Error()
	}
	puller.logIt(ref, "Pulled", logPrefix, fmt.Sprintf("Successfully pulled image %q", container.Image), glog.Info)
	puller.backOff.GC()
	return nil, ""
}
