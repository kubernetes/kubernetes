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
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/util"
)

type imagePullRequest struct {
	spec        ImageSpec
	container   *api.Container
	pullSecrets []api.Secret
	ref         *api.ObjectReference
	returnChan  chan<- error
}

// serializedImagePuller pulls the image using Runtime.PullImage().
// It will check the presence of the image, and report the 'image pulling',
// 'image pulled' events correspondingly.
type serializedImagePuller struct {
	recorder     record.EventRecorder
	runtime      Runtime
	pullRequests chan *imagePullRequest
}

// enforce compatibility.
var _ ImagePuller = &serializedImagePuller{}

// NewSerializedImagePuller takes an event recorder and container runtime to create a
// image puller that wraps the container runtime's PullImage interface.
// Pulls one image at a time.
// Issue #10959 has the rationale behind serializing image pulls.
func NewSerializedImagePuller(recorder record.EventRecorder, runtime Runtime) ImagePuller {
	imagePuller := &serializedImagePuller{
		recorder:     recorder,
		runtime:      runtime,
		pullRequests: make(chan *imagePullRequest, 10),
	}
	go util.Until(imagePuller.pullImages, time.Second, util.NeverStop)
	return imagePuller
}

// reportImagePull reports 'image pulling', 'image pulled' or 'image pulling failed' events.
func (puller *serializedImagePuller) reportImagePull(ref *api.ObjectReference, event string, image string, pullError error) {
	if ref == nil {
		return
	}

	switch event {
	case "pulling":
		puller.recorder.Eventf(ref, "Pulling", "Pulling image %q", image)
	case "pulled":
		puller.recorder.Eventf(ref, "Pulled", "Successfully pulled image %q", image)
	case "failed":
		puller.recorder.Eventf(ref, "Failed", "Failed to pull image %q: %v", image, pullError)

	}
}

// PullImage pulls the image for the specified pod and container.
func (puller *serializedImagePuller) PullImage(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) error {
	ref, err := GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}

	spec := ImageSpec{container.Image}
	present, err := puller.runtime.IsImagePresent(spec)
	if err != nil {
		if ref != nil {
			puller.recorder.Eventf(ref, "Failed", "Failed to inspect image %q: %v", container.Image, err)
		}
		return fmt.Errorf("failed to inspect image %q: %v", container.Image, err)
	}

	if !shouldPullImage(container, present) {
		if present && ref != nil {
			puller.recorder.Eventf(ref, "Pulled", "Container image %q already present on machine", container.Image)
		}
		return nil
	}

	// enqueue image pull request and wait for response.
	returnChan := make(chan error)
	puller.pullRequests <- &imagePullRequest{
		spec:        spec,
		container:   container,
		pullSecrets: pullSecrets,
		ref:         ref,
		returnChan:  returnChan,
	}
	if err := <-returnChan; err != nil {
		puller.reportImagePull(ref, "failed", container.Image, err)
		return err
	}
	puller.reportImagePull(ref, "pulled", container.Image, nil)
	return nil
}

func (puller *serializedImagePuller) pullImages() {
	for pullRequest := range puller.pullRequests {
		puller.reportImagePull(pullRequest.ref, "pulling", pullRequest.container.Image, nil)
		pullRequest.returnChan <- puller.runtime.PullImage(pullRequest.spec, pullRequest.pullSecrets)
	}
}
