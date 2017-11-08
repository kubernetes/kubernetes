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

package persistentvolume

import (
	"k8s.io/api/core/v1"
)

type FakeVolumeBinderConfig struct {
	AllBound     bool
	FindFoundPVs bool
	FindErr      error
	AssumeErr    error
	BindErr      error
}

// NewTopologyAwareVolumeBinder sets up all the caches needed for the scheduler to make
// topology-aware volume binding decisions.
func NewFakeVolumeBinder(config *FakeVolumeBinderConfig) *FakeVolumeBinder {
	return &FakeVolumeBinder{
		config: config,
	}
}

type FakeVolumeBinder struct {
	config       *FakeVolumeBinderConfig
	AssumeCalled bool
	BindCalled   bool
}

func (b *FakeVolumeBinder) FindPodVolumes(pod *v1.Pod, nodeName string) (needsBinding, foundPVs bool, err error) {
	return !b.config.AllBound, b.config.FindFoundPVs, b.config.FindErr
}

func (b *FakeVolumeBinder) AssumePodVolumes(assumedPod *v1.Pod, nodeName string) (bool, error) {
	b.AssumeCalled = true
	return !b.config.AllBound, b.config.AssumeErr
}

func (b *FakeVolumeBinder) BindPodVolumes(assumedPod *v1.Pod) (bool, error) {
	b.BindCalled = true
	return !b.config.AllBound, b.config.BindErr
}

func (b *FakeVolumeBinder) InitTmpData(pod *v1.Pod) {
	return
}
