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
	"time"

	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/wait"
)

type imagePuller interface {
	pullImage(kubecontainer.ImageSpec, []api.Secret, chan<- error)
}

var _, _ imagePuller = &parallelImagePuller{}, &serialImagePuller{}

type parallelImagePuller struct {
	runtime kubecontainer.Runtime
}

func newParallelImagePuller(runtime kubecontainer.Runtime) imagePuller {
	return &parallelImagePuller{runtime}
}

func (pip *parallelImagePuller) pullImage(spec kubecontainer.ImageSpec, pullSecrets []api.Secret, errChan chan<- error) {
	go func() {
		errChan <- pip.runtime.PullImage(spec, pullSecrets)
	}()
}

// Maximum number of image pull requests than can be queued.
const maxImagePullRequests = 10

type serialImagePuller struct {
	runtime      kubecontainer.Runtime
	pullRequests chan *imagePullRequest
}

func newSerialImagePuller(runtime kubecontainer.Runtime) imagePuller {
	imagePuller := &serialImagePuller{runtime, make(chan *imagePullRequest, maxImagePullRequests)}
	go wait.Until(imagePuller.processImagePullRequests, time.Second, wait.NeverStop)
	return imagePuller
}

type imagePullRequest struct {
	spec        kubecontainer.ImageSpec
	pullSecrets []api.Secret
	errChan     chan<- error
}

func (sip *serialImagePuller) pullImage(spec kubecontainer.ImageSpec, pullSecrets []api.Secret, errChan chan<- error) {
	sip.pullRequests <- &imagePullRequest{
		spec:        spec,
		pullSecrets: pullSecrets,
		errChan:     errChan,
	}
}

func (sip *serialImagePuller) processImagePullRequests() {
	for pullRequest := range sip.pullRequests {
		pullRequest.errChan <- sip.runtime.PullImage(pullRequest.spec, pullRequest.pullSecrets)
	}
}
