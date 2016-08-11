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

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

// imageManager provides the functionalities for image pulling.
type imageManager struct {
	recorder record.EventRecorder
	runtime  kubecontainer.Runtime
	backOff  *flowcontrol.Backoff
	// It will check the presence of the image, and report the 'image pulling', image pulled' events correspondingly.
	puller imagePuller
}

var _ ImageManager = &imageManager{}

func NewImageManager(recorder record.EventRecorder, runtime kubecontainer.Runtime, imageBackOff *flowcontrol.Backoff, serialized bool) ImageManager {
	var puller imagePuller
	if serialized {
		puller = newSerialImagePuller(runtime)
	} else {
		puller = newParallelImagePuller(runtime)
	}
	return &imageManager{
		recorder: recorder,
		runtime:  runtime,
		backOff:  imageBackOff,
		puller:   puller,
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
