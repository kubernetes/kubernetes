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

type fakeImpl struct {
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

func (fake *fakeImpl) ImageStatus(context.Context, cri.ImageManagerService, *v1.ImageSpec, bool) (*v1.ImageStatusResponse, error) {
	fakeReturns := fake.imageStatusReturns
	return fakeReturns.res, fakeReturns.err
}

func (fake *fakeImpl) ImageStatusReturns(res *v1.ImageStatusResponse, err error) {
	fake.imageStatusReturns = struct {
		res *v1.ImageStatusResponse
		err error
	}{res, err}
}

func (fake *fakeImpl) ListPodSandbox(context.Context, cri.RuntimeService, *v1.PodSandboxFilter) ([]*v1.PodSandbox, error) {
	fakeReturns := fake.listPodSandboxReturns
	return fakeReturns.res, fakeReturns.err
}

func (fake *fakeImpl) ListPodSandboxReturns(res []*v1.PodSandbox, err error) {
	fake.listPodSandboxReturns = struct {
		res []*v1.PodSandbox
		err error
	}{res, err}
}

func (fake *fakeImpl) NewRemoteImageService(string, time.Duration) (cri.ImageManagerService, error) {
	fakeReturns := fake.newRemoteImageServiceReturns
	return fakeReturns.res, fakeReturns.err
}

func (fake *fakeImpl) NewRemoteImageServiceReturns(res cri.ImageManagerService, err error) {
	fake.newRemoteImageServiceReturns = struct {
		res cri.ImageManagerService
		err error
	}{res, err}
}

func (fake *fakeImpl) NewRemoteRuntimeService(string, time.Duration) (cri.RuntimeService, error) {
	fakeReturns := fake.newRemoteRuntimeServiceReturns
	return fakeReturns.res, fakeReturns.err
}

func (fake *fakeImpl) NewRemoteRuntimeServiceReturns(res cri.RuntimeService, err error) {
	fake.newRemoteRuntimeServiceReturns = struct {
		res cri.RuntimeService
		err error
	}{res, err}
}

func (fake *fakeImpl) PullImage(context.Context, cri.ImageManagerService, *v1.ImageSpec, *v1.AuthConfig, *v1.PodSandboxConfig) (string, error) {
	fakeReturns := fake.pullImageReturns
	return fakeReturns.res, fakeReturns.err
}

func (fake *fakeImpl) PullImageReturns(res string, err error) {
	fake.pullImageReturns = struct {
		res string
		err error
	}{res, err}
}

func (fake *fakeImpl) RemovePodSandbox(context.Context, cri.RuntimeService, string) error {
	fakeReturns := fake.removePodSandboxReturns
	return fakeReturns.res
}

func (fake *fakeImpl) RemovePodSandboxReturns(res error) {
	fake.removePodSandboxReturns = struct {
		res error
	}{res}
}

func (fake *fakeImpl) Status(context.Context, cri.RuntimeService, bool) (*v1.StatusResponse, error) {
	fakeReturns := fake.statusReturns
	return fakeReturns.res, fakeReturns.err
}

func (fake *fakeImpl) StatusReturns(res *v1.StatusResponse, err error) {
	fake.statusReturns = struct {
		res *v1.StatusResponse
		err error
	}{res, err}
}

func (fake *fakeImpl) StopPodSandbox(context.Context, cri.RuntimeService, string) error {
	fakeReturns := fake.stopPodSandboxReturns
	return fakeReturns.res
}

func (fake *fakeImpl) StopPodSandboxReturns(res error) {
	fake.stopPodSandboxReturns = struct {
		res error
	}{res}
}
