/*
Copyright 2017 The Kubernetes Authors.

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

package gpu

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/v1"
)

type gpuManagerStub struct{}

func (gms *gpuManagerStub) Start() error {
	return nil
}

func (gms *gpuManagerStub) Capacity() v1.ResourceList {
	return nil
}

func (gms *gpuManagerStub) AllocateGPU(_ *v1.Pod, _ *v1.Container) ([]string, error) {
	return nil, fmt.Errorf("GPUs are not supported")
}

func NewGPUManagerStub() GPUManager {
	return &gpuManagerStub{}
}
