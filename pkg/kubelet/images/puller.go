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

import "k8s.io/kubernetes/pkg/kubelet/container"

// imagePuller pulls an image.
type imagePuller interface {
	pullImage(spec container.ImageSpec, prePullChan chan<- struct{}, errChan chan<- error)
}

type parallelImagePuller struct {
	runtime container.Runtime
}

func newParallelImagePuller(runtime container.Runtime) imagePuller {
	return &parallelImagePuller{runtime}
}

func (pip *parallelImagePuller) pullImage(spec container.ImageSpec, prePullChan chan<- struct{}, errChan chan<- error) {
	go func() {
		prePullChan <- struct{}{}
		errChan <- pip.runtime.PullImage(spec, pullSecrets)
	}()
}

// Maximum number of image pull requests than can be queued.
const maxImagePullRequests = 10

type serialImagePuller struct {
	runtime      container.Runtime
	pullRequests chan *imagePullRequest
}

func newSerialImagePuller(runtime container.Runtime) imagePuller {
	return &serialImagePuller{runtime, make(chan *imagePullRequest, maxImagePullRequests)}
}

type imagePullRequest struct {
	spec        container.ImageSpec
	prePullChan chan<- struct{}
	errChan     chan<- error
}

func (sip *serialImagePuller) pullImage(spec container.ImageSpec, prePullChan chan<- struct{}, errChan chan<- error) {
	sip.pullRequests <- &imagePullRequest{
		spec:        spec,
		prePullChan: prePullChan,
		errChan:     errChan,
	}
}

func (sip *serialImagePuller) processImagePullRequests() {
	for pullRequest := range puller.pullRequests {
		prePullChan <- struct{}{}
		errChan <- pip.runtime.PullImage(spec, pullSecrets)
	}
}
