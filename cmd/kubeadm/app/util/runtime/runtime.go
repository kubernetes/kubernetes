/*
Copyright 2018 The Kubernetes Authors.

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

// Package runtime provides the kubeadm container runtime implementation.
package runtime

import (
	"context"
	"encoding/json"
	"strings"
	"time"

	"github.com/pkg/errors"
	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	criapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// defaultKnownCRISockets holds the set of known CRI endpoints
var defaultKnownCRISockets = []string{
	constants.CRISocketContainerd,
	constants.CRISocketCRIO,
	constants.CRISocketDocker,
}

// ContainerRuntime is an interface for working with container runtimes
type ContainerRuntime interface {
	Connect() error
	SetImpl(impl)
	IsRunning() error
	ListKubeContainers() ([]string, error)
	RemoveContainers(containers []string) error
	PullImage(image string) error
	PullImagesInParallel(images []string, ifNotPresent bool) error
	ImageExists(image string) bool
	SandboxImage() (string, error)
}

// CRIRuntime is a struct that interfaces with the CRI
type CRIRuntime struct {
	impl           impl
	criSocket      string
	runtimeService criapi.RuntimeService
	imageService   criapi.ImageManagerService
}

// defaultTimeout is the default timeout inherited by crictl
const defaultTimeout = 2 * time.Second

// NewContainerRuntime sets up and returns a ContainerRuntime struct
func NewContainerRuntime(criSocket string) ContainerRuntime {
	return &CRIRuntime{
		impl:      &defaultImpl{},
		criSocket: criSocket,
	}
}

// SetImpl can be used to set the internal implementation for testing purposes.
func (runtime *CRIRuntime) SetImpl(impl impl) {
	runtime.impl = impl
}

// Connect establishes a connection with the CRI runtime.
func (runtime *CRIRuntime) Connect() error {
	runtimeService, err := runtime.impl.NewRemoteRuntimeService(runtime.criSocket, defaultTimeout)
	if err != nil {
		return errors.Wrap(err, "failed to create new CRI runtime service")
	}
	runtime.runtimeService = runtimeService

	imageService, err := runtime.impl.NewRemoteImageService(runtime.criSocket, defaultTimeout)
	if err != nil {
		return errors.Wrap(err, "failed to create new CRI image service")
	}
	runtime.imageService = imageService

	return nil
}

// IsRunning checks if runtime is running.
func (runtime *CRIRuntime) IsRunning() error {
	ctx, cancel := defaultContext()
	defer cancel()

	res, err := runtime.impl.Status(ctx, runtime.runtimeService, false)
	if err != nil {
		return errors.Wrap(err, "container runtime is not running")
	}

	for _, condition := range res.GetStatus().GetConditions() {
		if condition.GetType() == runtimeapi.RuntimeReady && // NetworkReady will not be tested on purpose
			!condition.GetStatus() {
			return errors.Errorf(
				"container runtime condition %q is not true. reason: %s, message: %s",
				condition.GetType(), condition.GetReason(), condition.GetMessage(),
			)
		}
	}

	return nil
}

// ListKubeContainers lists running k8s CRI pods
func (runtime *CRIRuntime) ListKubeContainers() ([]string, error) {
	ctx, cancel := defaultContext()
	defer cancel()

	sandboxes, err := runtime.impl.ListPodSandbox(ctx, runtime.runtimeService, nil)
	if err != nil {
		return nil, errors.Wrap(err, "failed to list pod sandboxes")
	}

	pods := []string{}
	for _, sandbox := range sandboxes {
		pods = append(pods, sandbox.GetId())
	}
	return pods, nil
}

// RemoveContainers removes running k8s pods
func (runtime *CRIRuntime) RemoveContainers(containers []string) error {
	errs := []error{}
	for _, container := range containers {
		var lastErr error
		for i := 0; i < constants.RemoveContainerRetry; i++ {
			klog.V(5).Infof("Attempting to remove container %v", container)

			ctx, cancel := defaultContext()
			if err := runtime.impl.StopPodSandbox(ctx, runtime.runtimeService, container); err != nil {
				lastErr = errors.Wrapf(err, "failed to stop running pod %s", container)
				cancel()
				continue
			}
			cancel()

			ctx, cancel = defaultContext()
			if err := runtime.impl.RemovePodSandbox(ctx, runtime.runtimeService, container); err != nil {
				lastErr = errors.Wrapf(err, "failed to remove pod %s", container)
				cancel()
				continue
			}
			cancel()

			lastErr = nil
			break
		}

		if lastErr != nil {
			errs = append(errs, lastErr)
		}
	}
	return errorsutil.NewAggregate(errs)
}

// PullImage pulls the image
func (runtime *CRIRuntime) PullImage(image string) (err error) {
	for i := 0; i < constants.PullImageRetry; i++ {
		if _, err = runtime.impl.PullImage(context.Background(), runtime.imageService, &runtimeapi.ImageSpec{Image: image}, nil, nil); err == nil {
			return nil
		}
	}
	return errors.Wrapf(err, "failed to pull image %s", image)
}

// PullImagesInParallel pulls a list of images in parallel
func (runtime *CRIRuntime) PullImagesInParallel(images []string, ifNotPresent bool) error {
	errs := pullImagesInParallelImpl(images, ifNotPresent, runtime.ImageExists, runtime.PullImage)
	return errorsutil.NewAggregate(errs)
}

func defaultContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), defaultTimeout)
}

func pullImagesInParallelImpl(images []string, ifNotPresent bool,
	imageExistsFunc func(string) bool, pullImageFunc func(string) error) []error {

	var errs []error
	errChan := make(chan error, len(images))

	klog.V(1).Info("pulling all images in parallel")
	for _, img := range images {
		image := img
		go func() {
			if ifNotPresent {
				exists := imageExistsFunc(image)
				if exists {
					klog.V(1).Infof("image exists: %s", image)
					errChan <- nil
					return
				}
			}
			err := pullImageFunc(image)
			if err != nil {
				err = errors.WithMessagef(err, "failed to pull image %s", image)
			} else {
				klog.V(1).Infof("done pulling: %s", image)
			}
			errChan <- err
		}()
	}

	for i := 0; i < len(images); i++ {
		if err := <-errChan; err != nil {
			errs = append(errs, err)
		}
	}

	return errs
}

// ImageExists checks to see if the image exists on the system
func (runtime *CRIRuntime) ImageExists(image string) bool {
	ctx, cancel := defaultContext()
	defer cancel()
	resp, err := runtime.impl.ImageStatus(ctx, runtime.imageService, &runtimeapi.ImageSpec{Image: image}, false)
	if err != nil {
		klog.Warningf("Failed to get image status, image: %q, error: %v", image, err)
		return false
	}
	if resp == nil || resp.Image == nil {
		return false
	}
	return true
}

// detectCRISocketImpl is separated out only for test purposes, DON'T call it directly, use DetectCRISocket instead
func detectCRISocketImpl(isSocket func(string) bool, knownCRISockets []string) (string, error) {
	foundCRISockets := []string{}

	for _, socket := range knownCRISockets {
		if isSocket(socket) {
			foundCRISockets = append(foundCRISockets, socket)
		}
	}

	switch len(foundCRISockets) {
	case 0:
		// Fall back to the default socket if no CRI is detected, we can error out later on if we need it
		return constants.DefaultCRISocket, nil
	case 1:
		// Precisely one CRI found, use that
		return foundCRISockets[0], nil
	default:
		// Multiple CRIs installed?
		return "", errors.Errorf("found multiple CRI endpoints on the host. Please define which one do you wish "+
			"to use by setting the 'criSocket' field in the kubeadm configuration file: %s",
			strings.Join(foundCRISockets, ", "))
	}
}

// DetectCRISocket uses a list of known CRI sockets to detect one. If more than one or none is discovered, an error is returned.
func DetectCRISocket() (string, error) {
	return detectCRISocketImpl(isExistingSocket, defaultKnownCRISockets)
}

// SandboxImage returns the sandbox image used by the container runtime
func (runtime *CRIRuntime) SandboxImage() (string, error) {
	ctx, cancel := defaultContext()
	defer cancel()
	status, err := runtime.impl.Status(ctx, runtime.runtimeService, true)
	if err != nil {
		return "", errors.Wrap(err, "failed to get runtime status")
	}

	infoConfig, ok := status.GetInfo()["config"]
	if !ok {
		return "", errors.Errorf("no 'config' field in CRI info: %+v", status)
	}

	type config struct {
		SandboxImage string `json:"sandboxImage,omitempty"`
	}
	c := config{}

	if err := json.Unmarshal([]byte(infoConfig), &c); err != nil {
		return "", errors.Wrap(err, "failed to unmarshal CRI info config")
	}

	if c.SandboxImage == "" {
		return "", errors.New("no 'sandboxImage' field in CRI info config")
	}

	return c.SandboxImage, nil
}
