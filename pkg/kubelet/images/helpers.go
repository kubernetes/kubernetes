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
	"fmt"
	"sort"
	"strings"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/util/flowcontrol"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// throttleImagePulling wraps kubecontainer.ImageService to throttle image
// pulling based on the given QPS and burst limits. If QPS is zero, defaults
// to no throttling.
func throttleImagePulling(imageService kubecontainer.ImageService, qps float32, burst int) kubecontainer.ImageService {
	if qps == 0.0 {
		return imageService
	}
	return &throttledImageService{
		ImageService: imageService,
		limiter:      flowcontrol.NewTokenBucketRateLimiter(qps, burst),
	}
}

type throttledImageService struct {
	kubecontainer.ImageService
	limiter flowcontrol.RateLimiter
}

func (ts throttledImageService) PullImage(ctx context.Context, image kubecontainer.ImageSpec, secrets []v1.Secret, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	if ts.limiter.TryAccept() {
		return ts.ImageService.PullImage(ctx, image, secrets, podSandboxConfig)
	}
	return "", fmt.Errorf("pull QPS exceeded")
}

// ParallelBlocker for parallel puller control.
type ParallelBlocker interface {
	// Wait if it needs block pull request.
	// Recover is used to unblock after process.
	Wait(image kubecontainer.ImageSpec) Recover
}

// Recover function of release lock of ParallelBlocker from Wait() was called.
type Recover func()

// defaultParallelBlocker creates a ParallelBlocker with default implementation blockerWithMaxParallel.
// Blocker will not limit concurrency  with maxParallelImagePulls nil or less than 1.
func defaultParallelBlocker(maxParallelImagePulls *int32) ParallelBlocker {
	if maxParallelImagePulls == nil || *maxParallelImagePulls < 1 {
		return &blockerWithMaxParallel{
			imagePullingMap:          sync.Map{},
			tokensForDistinctImageID: nil,
		}
	} else {
		return &blockerWithMaxParallel{
			imagePullingMap:          sync.Map{},
			tokensForDistinctImageID: make(chan struct{}, *maxParallelImagePulls),
		}
	}
}

// blockerWithMaxParallel implements ParallelBlocker as default.
// limits the number of concurrent calls.
type blockerWithMaxParallel struct {
	imagePullingMap          sync.Map
	tokensForDistinctImageID chan struct{}
}

// Wait limits with tokensForDistinctImageID
// Image pull requests with the same imageID will not be executed concurrently.
// This ensures that if a Node serving numerous Pods with the same image `X`
// and one Pod with the different image `Y`, The same image name will be pulled serially,
// instead of `Y` waiting on `maxParallelImagePulls`'s pulls of `X`.
func (b *blockerWithMaxParallel) Wait(image kubecontainer.ImageSpec) Recover {
	if b.tokensForDistinctImageID != nil {
		tokensForSameImageID, _ := b.imagePullingMap.LoadOrStore(genKeyOfPullingMap(image), make(chan struct{}, 1))
		tokensForSameImageID.(chan struct{}) <- struct{}{}
		b.tokensForDistinctImageID <- struct{}{}
		return func() {
			<-b.tokensForDistinctImageID
			<-tokensForSameImageID.(chan struct{})
		}
	} else {
		return func() {}
	}
}

func genKeyOfPullingMap(image kubecontainer.ImageSpec) string {
	res := []string{}
	for _, annotation := range image.Annotations {
		res = append(res, fmt.Sprintf("%s-%s", annotation.Name, annotation.Value))
	}
	sort.Strings(res)
	if image.RuntimeHandler != "" {
		res = append([]string{image.RuntimeHandler}, res...)
	}
	if image.Image != "" {
		res = append([]string{image.Image}, res...)
	}
	return strings.Join(res, "-")
}
