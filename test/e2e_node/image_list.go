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

package e2enode

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/user"
	"sync"
	"time"

	"k8s.io/klog/v2"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/test/e2e/framework"
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

// NodePrePullImageList is a list of images used in node e2e test. These images will be prepulled
// before test running so that the image pulling won't fail in actual test.
var NodePrePullImageList = sets.NewString(
	imageutils.GetE2EImage(imageutils.Agnhost),
	"google/cadvisor:latest",
	"k8s.gcr.io/stress:v1",
	busyboxImage,
	"k8s.gcr.io/busybox@sha256:4bdd623e848417d96127e16037743f0cd8b528c026e9175e22a84f639eca58ff",
	imageutils.GetE2EImage(imageutils.Nginx),
	imageutils.GetE2EImage(imageutils.Perl),
	imageutils.GetE2EImage(imageutils.Nonewprivs),
	imageutils.GetPauseImageName(),
	imageutils.GetE2EImage(imageutils.NodePerfNpbEp),
	imageutils.GetE2EImage(imageutils.NodePerfNpbIs),
	imageutils.GetE2EImage(imageutils.NodePerfTfWideDeep),
)

// puller represents a generic image puller
type puller interface {
	// Pull pulls an image by name
	Pull(image string) ([]byte, error)
	// Name returns the name of the specific puller implementation
	Name() string
}

type dockerPuller struct {
}

func (dp *dockerPuller) Name() string {
	return "docker"
}

func (dp *dockerPuller) Pull(image string) ([]byte, error) {
	// TODO(random-liu): Use docker client to get rid of docker binary dependency.
	if exec.Command("docker", "inspect", "--type=image", image).Run() != nil {
		return exec.Command("docker", "pull", image).CombinedOutput()
	}
	return nil, nil
}

type remotePuller struct {
	imageService internalapi.ImageManagerService
}

func (rp *remotePuller) Name() string {
	return "CRI"
}

func (rp *remotePuller) Pull(image string) ([]byte, error) {
	imageStatus, err := rp.imageService.ImageStatus(&runtimeapi.ImageSpec{Image: image})
	if err == nil && imageStatus != nil {
		return nil, nil
	}
	_, err = rp.imageService.PullImage(&runtimeapi.ImageSpec{Image: image}, nil, nil)
	return nil, err
}

func getPuller() (puller, error) {
	runtime := framework.TestContext.ContainerRuntime
	switch runtime {
	case "docker":
		return &dockerPuller{}, nil
	case "remote":
		_, is, err := getCRIClient()
		if err != nil {
			return nil, err
		}
		return &remotePuller{
			imageService: is,
		}, nil
	}
	return nil, fmt.Errorf("can't prepull images, unknown container runtime %q", runtime)
}

// PrePullAllImages pre-fetches all images tests depend on so that we don't fail in an actual test.
func PrePullAllImages() error {
	puller, err := getPuller()
	if err != nil {
		return err
	}
	usr, err := user.Current()
	if err != nil {
		return err
	}
	images := framework.ImagePrePullList.List()
	klog.V(4).Infof("Pre-pulling images with %s %+v", puller.Name(), images)

	imageCh := make(chan int, len(images))
	for i := range images {
		imageCh <- i
	}
	close(imageCh)

	pullErrs := make([]error, len(images))
	ctx, cancel := context.WithCancel(context.Background())
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
				for retryCount := 0; retryCount < maxImagePullRetries; retryCount++ {
					select {
					case <-ctx.Done():
						return
					default:
					}

					if retryCount > 0 {
						time.Sleep(imagePullRetryDelay)
					}
					if output, pullErr = puller.Pull(images[i]); pullErr == nil {
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

// Shared code across linux and cgo build tags within e2e. Requires lint ignores.
// nolint:deadcode
func getNodeProblemDetectorImage() string {
	const defaultImage string = "k8s.gcr.io/node-problem-detector:v0.6.2"
	image := os.Getenv("NODE_PROBLEM_DETECTOR_IMAGE")
	if image == "" {
		image = defaultImage
	}
	return image
}
