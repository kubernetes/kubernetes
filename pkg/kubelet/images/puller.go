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
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type pullResult struct {
	imageRef        string
	imageSize       uint64
	err             error
	pullDuration    time.Duration
	credentialsUsed *credentialprovider.TrackedAuthConfig
}

type imagePuller interface {
	pullImage(context.Context, kubecontainer.ImageSpec, []credentialprovider.TrackedAuthConfig, chan<- pullResult, *runtimeapi.PodSandboxConfig)
	// pullSecurityProfile pulls a security profile, sharing the limits of
	// image pulls, and returns whether it was already present.
	pullSecurityProfile(context.Context, kubecontainer.ImageSpec, []credentialprovider.TrackedAuthConfig, *runtimeapi.PodSandboxConfig, runtimeapi.SecurityProfileKind) (bool, error)
}

var _, _ imagePuller = &parallelImagePuller{}, &serialImagePuller{}

type parallelImagePuller struct {
	imageService kubecontainer.ImageService
	tokens       chan struct{}
}

func newParallelImagePuller(imageService kubecontainer.ImageService, maxParallelImagePulls *int32) imagePuller {
	if maxParallelImagePulls == nil || *maxParallelImagePulls < 1 {
		return &parallelImagePuller{imageService, nil}
	}
	return &parallelImagePuller{imageService, make(chan struct{}, *maxParallelImagePulls)}
}

func (pip *parallelImagePuller) pullImage(ctx context.Context, spec kubecontainer.ImageSpec, credentials []credentialprovider.TrackedAuthConfig, pullChan chan<- pullResult, podSandboxConfig *runtimeapi.PodSandboxConfig) {
	if pip.tokens != nil {
		pip.tokens <- struct{}{}
		defer func() { <-pip.tokens }()
	}
	startTime := time.Now()
	imageRef, creds, err := pip.imageService.PullImage(ctx, spec, credentials, podSandboxConfig)
	var size uint64
	if err == nil && imageRef != "" {
		// Getting the image size with best effort, ignoring the error.
		size, _ = pip.imageService.GetImageSize(ctx, spec)
	}
	pullChan <- pullResult{
		imageRef:        imageRef,
		imageSize:       size,
		err:             err,
		pullDuration:    time.Since(startTime),
		credentialsUsed: creds,
	}
}

func (pip *parallelImagePuller) pullSecurityProfile(ctx context.Context, spec kubecontainer.ImageSpec, credentials []credentialprovider.TrackedAuthConfig, podSandboxConfig *runtimeapi.PodSandboxConfig, kind runtimeapi.SecurityProfileKind) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if pip.tokens != nil {
		select {
		case pip.tokens <- struct{}{}:
		case <-ctx.Done():
			return false, ctx.Err()
		}
		defer func() { <-pip.tokens }()
	}
	return pip.imageService.PullSecurityProfile(ctx, spec, credentials, podSandboxConfig, kind)
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
	credentials      []credentialprovider.TrackedAuthConfig
	pullChan         chan<- pullResult
	podSandboxConfig *runtimeapi.PodSandboxConfig
	// pull, if set, replaces the image pull, for pulls that share the queue.
	pull func()
}

func (sip *serialImagePuller) pullImage(ctx context.Context, spec kubecontainer.ImageSpec, credentials []credentialprovider.TrackedAuthConfig, pullChan chan<- pullResult, podSandboxConfig *runtimeapi.PodSandboxConfig) {
	sip.pullRequests <- &imagePullRequest{
		ctx:              ctx,
		spec:             spec,
		credentials:      credentials,
		pullChan:         pullChan,
		podSandboxConfig: podSandboxConfig,
	}
}

func (sip *serialImagePuller) pullSecurityProfile(ctx context.Context, spec kubecontainer.ImageSpec, credentials []credentialprovider.TrackedAuthConfig, podSandboxConfig *runtimeapi.PodSandboxConfig, kind runtimeapi.SecurityProfileKind) (bool, error) {
	type result struct {
		cached bool
		err    error
	}
	done := make(chan result, 1)
	request := &imagePullRequest{
		pull: func() {
			if err := ctx.Err(); err != nil {
				done <- result{err: err}
				return
			}
			cached, err := sip.imageService.PullSecurityProfile(ctx, spec, credentials, podSandboxConfig, kind)
			done <- result{cached: cached, err: err}
		},
	}
	select {
	case sip.pullRequests <- request:
	case <-ctx.Done():
		return false, ctx.Err()
	}
	select {
	case r := <-done:
		return r.cached, r.err
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

func (sip *serialImagePuller) processImagePullRequests() {
	for pullRequest := range sip.pullRequests {
		if pullRequest.pull != nil {
			pullRequest.pull()
			continue
		}
		startTime := time.Now()
		imageRef, creds, err := sip.imageService.PullImage(pullRequest.ctx, pullRequest.spec, pullRequest.credentials, pullRequest.podSandboxConfig)
		var size uint64
		if err == nil && imageRef != "" {
			// Getting the image size with best effort, ignoring the error.
			size, _ = sip.imageService.GetImageSize(pullRequest.ctx, pullRequest.spec)
		}
		pullRequest.pullChan <- pullResult{
			imageRef:  imageRef,
			imageSize: size,
			err:       err,
			// Note: pullDuration includes getting the image size.
			pullDuration:    time.Since(startTime),
			credentialsUsed: creds,
		}
	}
}
