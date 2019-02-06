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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type pullResult struct {
	imageRef string
	err      error
}

type imagePuller interface {
	pullImage(kubecontainer.ImageSpec, []v1.Secret, chan<- pullResult, *runtimeapi.PodSandboxConfig)
}

var _, _ imagePuller = &parallelImagePuller{}, &serialImagePuller{}

type parallelImagePuller struct {
	imageService kubecontainer.ImageService
}

func newParallelImagePuller(imageService kubecontainer.ImageService) imagePuller {
	return &parallelImagePuller{imageService}
}

func (pip *parallelImagePuller) pullImage(spec kubecontainer.ImageSpec, pullSecrets []v1.Secret, pullChan chan<- pullResult, podSandboxConfig *runtimeapi.PodSandboxConfig) {
	go func() {
		imageRef, err := pip.imageService.PullImage(spec, pullSecrets, podSandboxConfig)
		pullChan <- pullResult{
			imageRef: imageRef,
			err:      err,
		}
	}()
}

// Maximum number of image pull requests than can be queued.
const maxImagePullRequests = 10

type serialImagePuller struct {
	imageService kubecontainer.ImageService
	pullRequests chan *imagePullRequest
}

func newSerialImagePuller(imageService kubecontainer.ImageService) imagePuller {
	imagePuller := &serialImagePuller{imageService, make(chan *imagePullRequest, maxImagePullRequests)}
	go wait.Until(imagePuller.processImagePullRequests, time.Second, wait.NeverStop)
	return imagePuller
}

type imagePullRequest struct {
	spec             kubecontainer.ImageSpec
	pullSecrets      []v1.Secret
	pullChan         chan<- pullResult
	podSandboxConfig *runtimeapi.PodSandboxConfig
}

func (sip *serialImagePuller) pullImage(spec kubecontainer.ImageSpec, pullSecrets []v1.Secret, pullChan chan<- pullResult, podSandboxConfig *runtimeapi.PodSandboxConfig) {
	sip.pullRequests <- &imagePullRequest{
		spec:             spec,
		pullSecrets:      pullSecrets,
		pullChan:         pullChan,
		podSandboxConfig: podSandboxConfig,
	}
}

func (sip *serialImagePuller) processImagePullRequests() {
	for pullRequest := range sip.pullRequests {
		imageRef, err := sip.imageService.PullImage(pullRequest.spec, pullRequest.pullSecrets, pullRequest.podSandboxConfig)
		pullRequest.pullChan <- pullResult{
			imageRef: imageRef,
			err:      err,
		}
	}
}
