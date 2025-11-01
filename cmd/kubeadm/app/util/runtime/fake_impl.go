/*
Copyright 2024 The Kubernetes Authors.

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
	"context"
	"time"

	cri "k8s.io/cri-api/pkg/apis"
	v1 "k8s.io/cri-api/pkg/apis/runtime/v1"
)

// FakeImpl is a fake implementation of the impl interface.
type FakeImpl struct {
	runtimeConfigReturns struct {
		res *v1.RuntimeConfigResponse
		err error
	}
	imageStatusReturns struct {
		res *v1.ImageStatusResponse
		err error
	}
	listPodSandboxReturns struct {
		res []*v1.PodSandbox
		err error
	}
	newRemoteImageServiceReturns struct {
		res cri.ImageManagerService
		err error
	}
	newRemoteRuntimeServiceReturns struct {
		res cri.RuntimeService
		err error
	}
	pullImageReturns struct {
		res string
		err error
	}
	removePodSandboxReturns struct {
		res error
	}
	statusReturns struct {
		res *v1.StatusResponse
		err error
	}
	stopPodSandboxReturns struct {
		res error
	}
}

// ImageStatus returns the status of the image.
func (fake *FakeImpl) ImageStatus(context.Context, cri.ImageManagerService, *v1.ImageSpec, bool) (*v1.ImageStatusResponse, error) {
	fakeReturns := fake.imageStatusReturns
	return fakeReturns.res, fakeReturns.err
}

// ImageStatusReturns sets the return values for the ImageStatus method.
func (fake *FakeImpl) ImageStatusReturns(res *v1.ImageStatusResponse, err error) {
	fake.imageStatusReturns = struct {
		res *v1.ImageStatusResponse
		err error
	}{res, err}
}

// ListPodSandbox returns the list of pod sandboxes.
func (fake *FakeImpl) ListPodSandbox(context.Context, cri.RuntimeService, *v1.PodSandboxFilter) ([]*v1.PodSandbox, error) {
	fakeReturns := fake.listPodSandboxReturns
	return fakeReturns.res, fakeReturns.err
}

// ListPodSandboxReturns sets the return values for the ListPodSandbox method.
func (fake *FakeImpl) ListPodSandboxReturns(res []*v1.PodSandbox, err error) {
	fake.listPodSandboxReturns = struct {
		res []*v1.PodSandbox
		err error
	}{res, err}
}

// NewRemoteImageService returns the new remote image service.
func (fake *FakeImpl) NewRemoteImageService(string, time.Duration) (cri.ImageManagerService, error) {
	fakeReturns := fake.newRemoteImageServiceReturns
	return fakeReturns.res, fakeReturns.err
}

// NewRemoteImageServiceReturns sets the return values for the NewRemoteImageService method.
func (fake *FakeImpl) NewRemoteImageServiceReturns(res cri.ImageManagerService, err error) {
	fake.newRemoteImageServiceReturns = struct {
		res cri.ImageManagerService
		err error
	}{res, err}
}

// NewRemoteRuntimeService returns the new remote runtime service.
func (fake *FakeImpl) NewRemoteRuntimeService(string, time.Duration) (cri.RuntimeService, error) {
	fakeReturns := fake.newRemoteRuntimeServiceReturns
	return fakeReturns.res, fakeReturns.err
}

// NewRemoteRuntimeServiceReturns sets the return values for the NewRemoteRuntimeService method.
func (fake *FakeImpl) NewRemoteRuntimeServiceReturns(res cri.RuntimeService, err error) {
	fake.newRemoteRuntimeServiceReturns = struct {
		res cri.RuntimeService
		err error
	}{res, err}
}

// PullImage returns the pull image.
func (fake *FakeImpl) PullImage(context.Context, cri.ImageManagerService, *v1.ImageSpec, *v1.AuthConfig, *v1.PodSandboxConfig) (string, error) {
	fakeReturns := fake.pullImageReturns
	return fakeReturns.res, fakeReturns.err
}

// PullImageReturns sets the return values for the PullImage method.
func (fake *FakeImpl) PullImageReturns(res string, err error) {
	fake.pullImageReturns = struct {
		res string
		err error
	}{res, err}
}

// RemovePodSandbox removes the pod sandbox.
func (fake *FakeImpl) RemovePodSandbox(context.Context, cri.RuntimeService, string) error {
	fakeReturns := fake.removePodSandboxReturns
	return fakeReturns.res
}

// RemovePodSandboxReturns sets the return values for the RemovePodSandbox method.
func (fake *FakeImpl) RemovePodSandboxReturns(res error) {
	fake.removePodSandboxReturns = struct {
		res error
	}{res}
}

// RuntimeConfig returns the runtime config.
func (fake *FakeImpl) RuntimeConfig(context.Context, cri.RuntimeService) (*v1.RuntimeConfigResponse, error) {
	fakeReturns := fake.runtimeConfigReturns
	return fakeReturns.res, fakeReturns.err
}

// RuntimeConfigReturns sets the return values for the RuntimeConfig method.
func (fake *FakeImpl) RuntimeConfigReturns(res *v1.RuntimeConfigResponse, err error) {
	fake.runtimeConfigReturns = struct {
		res *v1.RuntimeConfigResponse
		err error
	}{res, err}
}

// Status returns the status of the runtime.
func (fake *FakeImpl) Status(context.Context, cri.RuntimeService, bool) (*v1.StatusResponse, error) {
	fakeReturns := fake.statusReturns
	return fakeReturns.res, fakeReturns.err
}

// StatusReturns sets the return values for the Status method.
func (fake *FakeImpl) StatusReturns(res *v1.StatusResponse, err error) {
	fake.statusReturns = struct {
		res *v1.StatusResponse
		err error
	}{res, err}
}

// StopPodSandbox stops the pod sandbox.
func (fake *FakeImpl) StopPodSandbox(context.Context, cri.RuntimeService, string) error {
	fakeReturns := fake.stopPodSandboxReturns
	return fakeReturns.res
}

// StopPodSandboxReturns sets the return values for the StopPodSandbox method.
func (fake *FakeImpl) StopPodSandboxReturns(res error) {
	fake.stopPodSandboxReturns = struct {
		res error
	}{res}
}
