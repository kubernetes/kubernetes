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
	"context"
	"sort"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type pullResult struct {
	imageRef     string
	imageSize    uint64
	err          error
	pullDuration time.Duration
}

type imagePuller interface {
	pullImage(context.Context, kubecontainer.ImageSpec, []v1.Secret, chan pullResult, *runtimeapi.PodSandboxConfig, v1.PullPolicy)
}

var _, _ imagePuller = &parallelImagePuller{}, &serialImagePuller{}

type parallelImagePuller struct {
	imageService kubecontainer.ImageService
	tokens       chan struct{}
	pulls        sync.Map
}

type pullBroker struct {
	subscribers sync.Map
}

func (p *pullBroker) subscribe(ch chan pullResult) {
	p.subscribers.Store(ch, struct{}{})
}

func (p *pullBroker) publish(res pullResult) {
	p.subscribers.Range(func(k, v any) bool {
		ch, ok := k.(chan pullResult)
		if !ok {
			return false
		}

		select {
		case ch <- res:
		default:
		}

		return true
	})
}

func newParallelImagePuller(imageService kubecontainer.ImageService, maxParallelImagePulls *int32) imagePuller {
	if maxParallelImagePulls == nil || *maxParallelImagePulls < 1 {
		return &parallelImagePuller{imageService, nil, sync.Map{}}
	}
	return &parallelImagePuller{imageService, make(chan struct{}, *maxParallelImagePulls), sync.Map{}}
}

func pullKey(spec kubecontainer.ImageSpec) string {
	res := []string{}
	for _, annotation := range spec.Annotations {
		res = append(res, annotation.Name+"-"+annotation.Value)
	}
	sort.Strings(res)
	if spec.RuntimeHandler != "" {
		res = append([]string{spec.RuntimeHandler}, res...)
	}
	if spec.Image != "" {
		res = append([]string{spec.Image}, res...)
	}
	return strings.Join(res, "-")
}

func (pip *parallelImagePuller) pullImage(ctx context.Context, spec kubecontainer.ImageSpec, pullSecrets []v1.Secret, pullChan chan pullResult, podSandboxConfig *runtimeapi.PodSandboxConfig, pullPolicy v1.PullPolicy) {
	go func() {
		key := pullKey(spec)

		// Create a new pull broker.
		value, loaded := pip.pulls.LoadOrStore(key, &pullBroker{})
		broker, ok := value.(*pullBroker)
		if !ok {
			// should be unreachable
			return
		}

		// Reuse existing pull broker if available. A pull policy of 'Always'
		// will result in calling CRI's PullImage every time.
		if pullPolicy != v1.PullAlways && loaded {
			broker.subscribe(pullChan)
			return
		}

		if pip.tokens != nil {
			pip.tokens <- struct{}{}
			defer func() { <-pip.tokens }()
		}

		broker.subscribe(pullChan)

		startTime := time.Now()
		imageRef, err := pip.imageService.PullImage(ctx, spec, pullSecrets, podSandboxConfig)
		var size uint64
		if err == nil && imageRef != "" {
			// Getting the image size with best effort, ignoring the error.
			size, _ = pip.imageService.GetImageSize(ctx, spec)
		}

		broker.publish(pullResult{
			imageRef:     imageRef,
			imageSize:    size,
			err:          err,
			pullDuration: time.Since(startTime),
		})

		pip.pulls.Delete(key)
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
	ctx              context.Context
	spec             kubecontainer.ImageSpec
	pullSecrets      []v1.Secret
	pullChan         chan<- pullResult
	podSandboxConfig *runtimeapi.PodSandboxConfig
}

func (sip *serialImagePuller) pullImage(ctx context.Context, spec kubecontainer.ImageSpec, pullSecrets []v1.Secret, pullChan chan pullResult, podSandboxConfig *runtimeapi.PodSandboxConfig, pullPolicy v1.PullPolicy) {
	sip.pullRequests <- &imagePullRequest{
		ctx:              ctx,
		spec:             spec,
		pullSecrets:      pullSecrets,
		pullChan:         pullChan,
		podSandboxConfig: podSandboxConfig,
	}
}

func (sip *serialImagePuller) processImagePullRequests() {
	for pullRequest := range sip.pullRequests {
		startTime := time.Now()
		imageRef, err := sip.imageService.PullImage(pullRequest.ctx, pullRequest.spec, pullRequest.pullSecrets, pullRequest.podSandboxConfig)
		var size uint64
		if err == nil && imageRef != "" {
			// Getting the image size with best effort, ignoring the error.
			size, _ = sip.imageService.GetImageSize(pullRequest.ctx, pullRequest.spec)
		}
		pullRequest.pullChan <- pullResult{
			imageRef:  imageRef,
			imageSize: size,
			err:       err,
			// Note: pullDuration includes credential resolution and getting the image size.
			pullDuration: time.Since(startTime),
		}
	}
}
