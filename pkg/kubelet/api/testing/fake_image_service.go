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

package testing

import (
	"fmt"

	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

type fakeImageService struct {
}

func NewFakeImageService() internalApi.ImageManagerService {
	return &fakeImageService{}
}

func (r *fakeImageService) ListImages(filter *runtimeApi.ImageFilter) ([]*runtimeApi.Image, error) {
	return nil, fmt.Errorf("not implemented")
}

func (r *fakeImageService) ImageStatus(image *runtimeApi.ImageSpec) (*runtimeApi.Image, error) {
	return nil, fmt.Errorf("not implemented")
}

func (r *fakeImageService) PullImage(image *runtimeApi.ImageSpec, auth *runtimeApi.AuthConfig) error {
	return fmt.Errorf("not implemented")
}

func (r *fakeImageService) RemoveImage(image *runtimeApi.ImageSpec) error {
	return fmt.Errorf("not implemented")
}
