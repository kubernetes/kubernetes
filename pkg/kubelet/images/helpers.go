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
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
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
		ImageService:     imageService,
		limiter:          flowcontrol.NewTokenBucketRateLimiter(qps, burst),
		qps:              qps,
		burst:            burst,
		currentPullCount: 0,
	}
}

type throttledImageService struct {
	kubecontainer.ImageService
	limiter          flowcontrol.RateLimiter
	qps              float32
	burst            int
	currentPullCount int
	lock             sync.Mutex
}

var (
	errPullBurstExceeded error = fmt.Errorf("pull burst exceeded more than 1 minute")
	errPullQPSExceeded   error = fmt.Errorf("pull QPS exceeded")
)

func (ts *throttledImageService) PullImage(ctx context.Context, image kubecontainer.ImageSpec, secrets []v1.Secret, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	if ts.limiter.TryAccept() {
		err := wait.PollImmediate(100*time.Microsecond, 5*time.Minute, func() (done bool, err error) {
			if ts.AcceptOne() {
				return true, nil
			} else {
				return false, errPullBurstExceeded
			}
		})
		if err != nil {
			return ts.ImageService.PullImage(ctx, image, secrets, podSandboxConfig)
		}
	}

	return "", errPullQPSExceeded
}

func (ts *throttledImageService) AcceptOne() bool {
	ts.lock.Lock()
	defer ts.lock.Unlock()
	if ts.burst > ts.currentPullCount {
		ts.currentPullCount++
		return true
	}
	return false
}

func (ts *throttledImageService) ReleaseOne() {
	ts.lock.Lock()
	defer ts.lock.Unlock()
	if ts.currentPullCount > 0 {
		ts.currentPullCount--
	}
}
