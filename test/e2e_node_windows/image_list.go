//go:build windows

/*
Copyright The Kubernetes Authors.

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

package e2enodewindows

import (
	"context"
	"os/user"
	"sync"
	"time"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	commontest "k8s.io/kubernetes/test/e2e/common"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	// Number of attempts to pull an image.
	maxImagePullRetries = 5
	// Sleep duration between image pull retry attempts.
	imagePullRetryDelay = time.Second
	// Number of parallel count to pull images.
	maxParallelImagePullCount = 5
)

// NodePrePullImageList is a list of images used in Windows node e2e tests.
// These images will be prepulled before test running so that the image pulling
// won't fail in actual test.
var NodePrePullImageList = sets.NewString(
	imageutils.GetE2EImage(imageutils.Agnhost),
	busyboxImage,
	imageutils.GetPauseImageName(),
)

// updateImageAllowList updates the e2epod.ImagePrePullList with the Windows test images.
func updateImageAllowList(ctx context.Context) {
	e2epod.ImagePrePullList = NodePrePullImageList.Union(commontest.PrePulledImages)
}

// puller represents a generic image puller
type puller interface {
	// Pull pulls an image by name
	Pull(ctx context.Context, image string) ([]byte, error)
	// Name returns the name of the specific puller implementation
	Name() string
}

type remotePuller struct {
	imageService internalapi.ImageManagerService
}

func (rp *remotePuller) Name() string {
	return "CRI"
}

func (rp *remotePuller) Pull(ctx context.Context, image string) ([]byte, error) {
	resp, err := rp.imageService.ImageStatus(ctx, &runtimeapi.ImageSpec{Image: image}, false)
	if err == nil && resp.GetImage() != nil {
		return nil, nil
	}
	_, err = rp.imageService.PullImage(ctx, &runtimeapi.ImageSpec{Image: image}, nil, nil)
	return nil, err
}

func getPuller(ctx context.Context) (puller, error) {
	_, is, err := getCRIClient(ctx)
	if err != nil {
		return nil, err
	}
	return &remotePuller{
		imageService: is,
	}, nil
}

// PrePullAllImages pre-fetches all images tests depend on so that we don't fail in an actual test.
func PrePullAllImages(ctx context.Context) error {
	puller, err := getPuller(ctx)
	if err != nil {
		return err
	}
	usr, err := user.Current()
	if err != nil {
		return err
	}
	images := e2epod.ImagePrePullList.List()
	klog.V(4).Infof("Pre-pulling images with %s %+v", puller.Name(), images)

	imageCh := make(chan int, len(images))
	for i := range images {
		imageCh <- i
	}
	close(imageCh)

	pullErrs := make([]error, len(images))
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	parallelImagePullCount := maxParallelImagePullCount
	if len(images) < parallelImagePullCount {
		parallelImagePullCount = len(images)
	}

	var wg sync.WaitGroup
	wg.Add(parallelImagePullCount)
	for i := 0; i < parallelImagePullCount; i++ {
		go func() {
			defer wg.Done()

			for i := range imageCh {
				var (
					pullErr error
					output  []byte
				)
				for retryCount := range maxImagePullRetries {
					select {
					case <-ctx.Done():
						return
					default:
					}

					if retryCount > 0 {
						time.Sleep(imagePullRetryDelay)
					}
					if output, pullErr = puller.Pull(ctx, images[i]); pullErr == nil {
						break
					}
					klog.Warningf("Failed to pull %s as user %q, retrying in %s (%d of %d): %v",
						images[i], usr.Username, imagePullRetryDelay.String(), retryCount+1, maxImagePullRetries, pullErr)
				}
				if pullErr != nil {
					klog.Warningf("Could not pre-pull image %s %v output: %s", images[i], pullErr, output)
					pullErrs[i] = pullErr
					cancel()
					return
				}
			}
		}()
	}

	wg.Wait()
	return utilerrors.NewAggregate(pullErrs)
}
