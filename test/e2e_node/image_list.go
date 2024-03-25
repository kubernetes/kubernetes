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
	"os/user"
	"runtime"
	"sync"
	"time"

	"k8s.io/klog/v2"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	commontest "k8s.io/kubernetes/test/e2e/common"
	e2egpu "k8s.io/kubernetes/test/e2e/framework/gpu"
	e2emanifest "k8s.io/kubernetes/test/e2e/framework/manifest"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
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
	"gcr.io/cadvisor/cadvisor:v0.47.2",
	busyboxImage,
	"registry.k8s.io/e2e-test-images/busybox@sha256:a9155b13325b2abef48e71de77bb8ac015412a566829f621d06bfae5c699b1b9",
	imageutils.GetE2EImage(imageutils.Nginx),
	imageutils.GetE2EImage(imageutils.Perl),
	imageutils.GetE2EImage(imageutils.Nonewprivs),
	imageutils.GetPauseImageName(),
	imageutils.GetE2EImage(imageutils.NodePerfNpbEp),
	imageutils.GetE2EImage(imageutils.NodePerfNpbIs),
	imageutils.GetE2EImage(imageutils.Etcd),
)

// updateImageAllowList updates the e2epod.ImagePrePullList with
// 1. the hard coded lists
// 2. the ones passed in from framework.TestContext.ExtraEnvs
// So this function needs to be called after the extra envs are applied.
func updateImageAllowList(ctx context.Context) {
	// Architecture-specific image
	if !isRunningOnArm64() {
		// NodePerfTfWideDeep is only supported on x86_64, pulling in arm64 will fail
		NodePrePullImageList = NodePrePullImageList.Insert(imageutils.GetE2EImage(imageutils.NodePerfTfWideDeep))
	}
	// Union NodePrePullImageList and PrePulledImages into the framework image pre-pull list.
	e2epod.ImagePrePullList = NodePrePullImageList.Union(commontest.PrePulledImages)
	// Images from extra envs
	e2epod.ImagePrePullList.Insert(getNodeProblemDetectorImage())
	if sriovDevicePluginImage, err := getSRIOVDevicePluginImage(); err != nil {
		klog.Errorln(err)
	} else {
		e2epod.ImagePrePullList.Insert(sriovDevicePluginImage)
	}
	if gpuDevicePluginImage, err := getGPUDevicePluginImage(ctx); err != nil {
		klog.Errorln(err)
	} else {
		e2epod.ImagePrePullList.Insert(gpuDevicePluginImage)
	}
	if samplePluginImage, err := getContainerImageFromE2ETestDaemonset(SampleDevicePluginDSYAML); err != nil {
		klog.Errorln(err)
	} else {
		e2epod.ImagePrePullList.Insert(samplePluginImage)
	}
	if samplePluginImageCtrlReg, err := getContainerImageFromE2ETestDaemonset(SampleDevicePluginControlRegistrationDSYAML); err != nil {
		klog.Errorln(err)
	} else {
		e2epod.ImagePrePullList.Insert(samplePluginImageCtrlReg)
	}
}

func isRunningOnArm64() bool {
	return runtime.GOARCH == "arm64"
}

func getNodeProblemDetectorImage() string {
	const defaultImage string = "registry.k8s.io/node-problem-detector/node-problem-detector:v0.8.16"
	image := os.Getenv("NODE_PROBLEM_DETECTOR_IMAGE")
	if image == "" {
		image = defaultImage
	}
	return image
}

// puller represents a generic image puller
type puller interface {
	// Pull pulls an image by name
	Pull(image string) ([]byte, error)
	// Name returns the name of the specific puller implementation
	Name() string
}

type remotePuller struct {
	imageService internalapi.ImageManagerService
}

func (rp *remotePuller) Name() string {
	return "CRI"
}

func (rp *remotePuller) Pull(image string) ([]byte, error) {
	resp, err := rp.imageService.ImageStatus(context.Background(), &runtimeapi.ImageSpec{Image: image}, false)
	if err == nil && resp.GetImage() != nil {
		return nil, nil
	}
	_, err = rp.imageService.PullImage(context.Background(), &runtimeapi.ImageSpec{Image: image}, nil, nil)
	return nil, err
}

func getPuller() (puller, error) {
	_, is, err := getCRIClient()
	if err != nil {
		return nil, err
	}
	return &remotePuller{
		imageService: is,
	}, nil
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
	images := e2epod.ImagePrePullList.List()
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

// getGPUDevicePluginImage returns the image of GPU device plugin.
func getGPUDevicePluginImage(ctx context.Context) (string, error) {
	ds, err := e2emanifest.DaemonSetFromURL(ctx, e2egpu.GPUDevicePluginDSYAML)
	if err != nil {
		return "", fmt.Errorf("failed to parse the device plugin image: %w", err)
	}
	if ds == nil {
		return "", fmt.Errorf("failed to parse the device plugin image: the extracted DaemonSet is nil")
	}
	if len(ds.Spec.Template.Spec.Containers) < 1 {
		return "", fmt.Errorf("failed to parse the device plugin image: cannot extract the container from YAML")
	}
	return ds.Spec.Template.Spec.Containers[0].Image, nil
}

func getContainerImageFromE2ETestDaemonset(dsYamlPath string) (string, error) {
	data, err := e2etestfiles.Read(dsYamlPath)
	if err != nil {
		return "", fmt.Errorf("failed to read the daemonset yaml: %w", err)
	}

	ds, err := e2emanifest.DaemonSetFromData(data)
	if err != nil {
		return "", fmt.Errorf("failed to parse daemonset yaml: %w", err)
	}

	if len(ds.Spec.Template.Spec.Containers) < 1 {
		return "", fmt.Errorf("failed to parse the container image: cannot extract the container from YAML")
	}
	return ds.Spec.Template.Spec.Containers[0].Image, nil
}

// getSRIOVDevicePluginImage returns the image of SRIOV device plugin.
func getSRIOVDevicePluginImage() (string, error) {
	data, err := e2etestfiles.Read(SRIOVDevicePluginDSYAML)
	if err != nil {
		return "", fmt.Errorf("failed to read the device plugin manifest: %w", err)
	}
	ds, err := e2emanifest.DaemonSetFromData(data)
	if err != nil {
		return "", fmt.Errorf("failed to parse the device plugin image: %w", err)
	}
	if ds == nil {
		return "", fmt.Errorf("failed to parse the device plugin image: the extracted DaemonSet is nil")
	}
	if len(ds.Spec.Template.Spec.Containers) < 1 {
		return "", fmt.Errorf("failed to parse the device plugin image: cannot extract the container from YAML")
	}
	return ds.Spec.Template.Spec.Containers[0].Image, nil
}
