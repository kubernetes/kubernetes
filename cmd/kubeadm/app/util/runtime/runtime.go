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

package runtime

import (
	"os"
	"strings"

	"github.com/pkg/errors"

	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/klog/v2"
	utilsexec "k8s.io/utils/exec"

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
	Socket() string
	IsRunning() error
	ListKubeContainers() ([]string, error)
	RemoveContainers(containers []string) error
	PullImage(image string) error
	PullImagesInParallel(images []string, ifNotPresent bool) error
	ImageExists(image string) (bool, error)
	SandboxImage() (string, error)
}

// CRIRuntime is a struct that interfaces with the CRI
type CRIRuntime struct {
	exec       utilsexec.Interface
	criSocket  string
	crictlPath string
}

// NewContainerRuntime sets up and returns a ContainerRuntime struct
func NewContainerRuntime(execer utilsexec.Interface, criSocket string) (ContainerRuntime, error) {
	const toolName = "crictl"
	crictlPath, err := execer.LookPath(toolName)
	if err != nil {
		return nil, errors.Wrapf(err, "%s is required by the container runtime", toolName)
	}
	return &CRIRuntime{execer, criSocket, crictlPath}, nil
}

// Socket returns the CRI socket endpoint
func (runtime *CRIRuntime) Socket() string {
	return runtime.criSocket
}

// crictl creates a crictl command for the provided args.
func (runtime *CRIRuntime) crictl(args ...string) utilsexec.Cmd {
	cmd := runtime.exec.Command(runtime.crictlPath, append([]string{"-r", runtime.Socket(), "-i", runtime.Socket()}, args...)...)
	cmd.SetEnv(os.Environ())
	return cmd
}

// IsRunning checks if runtime is running
func (runtime *CRIRuntime) IsRunning() error {
	if out, err := runtime.crictl("info").CombinedOutput(); err != nil {
		return errors.Wrapf(err, "container runtime is not running: output: %s, error", string(out))
	}
	return nil
}

// ListKubeContainers lists running k8s CRI pods
func (runtime *CRIRuntime) ListKubeContainers() ([]string, error) {
	// Disable debug mode regardless how the crictl is configured so that the debug info won't be
	// iterpreted to the Pod ID.
	args := []string{"-D=false", "pods", "-q"}
	out, err := runtime.crictl(args...).CombinedOutput()
	if err != nil {
		return nil, errors.Wrapf(err, "output: %s, error", string(out))
	}
	pods := []string{}
	pods = append(pods, strings.Fields(string(out))...)
	return pods, nil
}

// RemoveContainers removes running k8s pods
func (runtime *CRIRuntime) RemoveContainers(containers []string) error {
	errs := []error{}
	for _, container := range containers {
		var lastErr error
		for i := 0; i < constants.RemoveContainerRetry; i++ {
			klog.V(5).Infof("Attempting to remove container %v", container)
			out, err := runtime.crictl("stopp", container).CombinedOutput()
			if err != nil {
				lastErr = errors.Wrapf(err, "failed to stop running pod %s: output: %s", container, string(out))
				continue
			}
			out, err = runtime.crictl("rmp", container).CombinedOutput()
			if err != nil {
				lastErr = errors.Wrapf(err, "failed to remove running container %s: output: %s", container, string(out))
				continue
			}
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
func (runtime *CRIRuntime) PullImage(image string) error {
	var err error
	var out []byte
	for i := 0; i < constants.PullImageRetry; i++ {
		out, err = runtime.crictl("pull", image).CombinedOutput()
		if err == nil {
			return nil
		}
	}
	return errors.Wrapf(err, "output: %s, error", out)
}

// PullImagesInParallel pulls a list of images in parallel
func (runtime *CRIRuntime) PullImagesInParallel(images []string, ifNotPresent bool) error {
	errs := pullImagesInParallelImpl(images, ifNotPresent, runtime.ImageExists, runtime.PullImage)
	return errorsutil.NewAggregate(errs)
}

func pullImagesInParallelImpl(images []string, ifNotPresent bool,
	imageExistsFunc func(string) (bool, error), pullImageFunc func(string) error) []error {

	var errs []error
	errChan := make(chan error, len(images))

	klog.V(1).Info("pulling all images in parallel")
	for _, img := range images {
		image := img
		go func() {
			if ifNotPresent {
				exists, err := imageExistsFunc(image)
				if err != nil {
					errChan <- errors.WithMessagef(err, "failed to check if image %s exists", image)
					return
				}
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
func (runtime *CRIRuntime) ImageExists(image string) (bool, error) {
	err := runtime.crictl("inspecti", image).Run()
	return err == nil, nil
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
		return "", errors.Errorf("Found multiple CRI endpoints on the host. Please define which one do you wish "+
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
	args := []string{"-D=false", "info", "-o", "go-template", "--template", "{{.config.sandboxImage}}"}
	out, err := runtime.crictl(args...).CombinedOutput()
	if err != nil {
		return "", errors.Wrapf(err, "output: %s, error", string(out))
	}

	sandboxImage := strings.TrimSpace(string(out))
	if len(sandboxImage) > 0 {
		return sandboxImage, nil
	}

	return "", errors.Errorf("the detected sandbox image is empty")
}
