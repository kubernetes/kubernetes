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
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/wait"
)

type imagePullRequest struct {
	spec        ImageSpec
	container   *api.Container
	pullSecrets []api.Secret
	logPrefix   string
	ref         *api.ObjectReference
	returnChan  chan<- error
}

// serializedImagePuller pulls the image using Runtime.PullImage().
// It will check the presence of the image, and report the 'image pulling',
// 'image pulled' events correspondingly.
type serializedImagePuller struct {
	recorder     record.EventRecorder
	runtime      Runtime
	backOff      *flowcontrol.Backoff
	pullRequests chan *imagePullRequest
}

// enforce compatibility.
var _ imagePuller = &serializedImagePuller{}

// NewSerializedImagePuller takes an event recorder and container runtime to create a
// image puller that wraps the container runtime's PullImage interface.
// Pulls one image at a time.
// Issue #10959 has the rationale behind serializing image pulls.
func NewSerializedImagePuller(recorder record.EventRecorder, runtime Runtime, imageBackOff *flowcontrol.Backoff) imagePuller {
	imagePuller := &serializedImagePuller{
		recorder:     recorder,
		runtime:      runtime,
		backOff:      imageBackOff,
		pullRequests: make(chan *imagePullRequest, 10),
	}
	go wait.Until(imagePuller.pullImages, time.Second, wait.NeverStop)
	return imagePuller
}

// records an event using ref, event msg.  log to glog using prefix, msg, logFn
func (puller *serializedImagePuller) logIt(ref *api.ObjectReference, eventtype, event, prefix, msg string, logFn func(args ...interface{})) {
	if ref != nil {
		puller.recorder.Event(ref, eventtype, event, msg)
	} else {
		logFn(fmt.Sprint(prefix, " ", msg))
	}
}

// PullImage pulls the image for the specified pod and container.
func (puller *serializedImagePuller) PullImage(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) (error, string) {
	logPrefix := fmt.Sprintf("%s/%s", pod.Name, container.Image)
	ref, err := GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}

	spec := ImageSpec{container.Image}
	present, err := puller.runtime.IsImagePresent(spec)
	if err != nil {
		msg := fmt.Sprintf("Failed to inspect image %q: %v", container.Image, err)
		puller.logIt(ref, api.EventTypeWarning, events.FailedToInspectImage, logPrefix, msg, glog.Warning)
		return ErrImageInspect, msg
	}

	if !shouldPullImage(container, present) {
		if present {
			msg := fmt.Sprintf("Container image %q already present on machine", container.Image)
			puller.logIt(ref, api.EventTypeNormal, events.PulledImage, logPrefix, msg, glog.Info)
			return nil, ""
		} else {
			msg := fmt.Sprintf("Container image %q is not present with pull policy of Never", container.Image)
			puller.logIt(ref, api.EventTypeWarning, events.ErrImageNeverPullPolicy, logPrefix, msg, glog.Warning)
			return ErrImageNeverPull, msg
		}
	}

	backOffKey := fmt.Sprintf("%s_%s", pod.Name, container.Image)
	if puller.backOff.IsInBackOffSinceUpdate(backOffKey, puller.backOff.Clock.Now()) {
		msg := fmt.Sprintf("Back-off pulling image %q", container.Image)
		puller.logIt(ref, api.EventTypeNormal, events.BackOffPullImage, logPrefix, msg, glog.Info)
		return ErrImagePullBackOff, msg
	}

	// enqueue image pull request and wait for response.
	returnChan := make(chan error)
	puller.pullRequests <- &imagePullRequest{
		spec:        spec,
		container:   container,
		pullSecrets: pullSecrets,
		logPrefix:   logPrefix,
		ref:         ref,
		returnChan:  returnChan,
	}
	if err = <-returnChan; err != nil {
		puller.logIt(ref, api.EventTypeWarning, events.FailedToPullImage, logPrefix, fmt.Sprintf("Failed to pull image %q: %v", container.Image, err), glog.Warning)
		puller.backOff.Next(backOffKey, puller.backOff.Clock.Now())
		if err == RegistryUnavailable {
			msg := fmt.Sprintf("image pull failed for %s because the registry is unavailable.", container.Image)
			return err, msg
		} else {
			return ErrImagePull, err.Error()
		}
	}
	puller.logIt(ref, api.EventTypeNormal, events.PulledImage, logPrefix, fmt.Sprintf("Successfully pulled image %q", container.Image), glog.Info)
	puller.backOff.GC()
	return nil, ""
}

func (puller *serializedImagePuller) pullImages() {
	for pullRequest := range puller.pullRequests {
		puller.logIt(pullRequest.ref, api.EventTypeNormal, events.PullingImage, pullRequest.logPrefix, fmt.Sprintf("pulling image %q", pullRequest.container.Image), glog.Info)
		pullRequest.returnChan <- puller.runtime.PullImage(pullRequest.spec, pullRequest.pullSecrets)
	}
}
