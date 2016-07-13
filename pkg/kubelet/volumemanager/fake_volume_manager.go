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

package volumemanager

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

func NewFakeVolumeManager() *FakeVolumeManager {
	return &FakeVolumeManager{}
}

type FakeVolumeManager struct{}

func (f *FakeVolumeManager) Run(stopCh <-chan struct{}) {
}

func (f *FakeVolumeManager) WaitForAttachAndMount(pod *api.Pod) error {
	return nil
}

func (f *FakeVolumeManager) GetMountedVolumesForPod(podName types.UniquePodName) container.VolumeMap {
	return container.VolumeMap{}
}

func (f *FakeVolumeManager) GetExtraSupplementalGroupsForPod(pod *api.Pod) []int64 {
	return nil
}

func (f *FakeVolumeManager) GetVolumesInUse() []api.UniqueVolumeName {
	return nil
}

func (f *FakeVolumeManager) MarkVolumesAsReportedInUse(volumesReportedAsInUse []api.UniqueVolumeName) {
}

func (f *FakeVolumeManager) VolumeIsAttached(volumeName api.UniqueVolumeName) bool {
	return false
}
