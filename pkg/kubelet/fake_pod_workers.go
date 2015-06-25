/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
)

// fakePodWorkers runs sync pod function in serial, so we can have
// deterministic behaviour in testing.
type fakePodWorkers struct {
	syncPodFn    syncPodFnType
	runtimeCache kubecontainer.RuntimeCache
	t            TestingInterface
}

func (f *fakePodWorkers) UpdatePod(pod *api.Pod, mirrorPod *api.Pod, updateComplete func()) {
	pods, err := f.runtimeCache.GetPods()
	if err != nil {
		f.t.Errorf("Unexpected error: %v", err)
	}
	if err := f.syncPodFn(pod, mirrorPod, kubecontainer.Pods(pods).FindPodByID(pod.UID), SyncPodUpdate); err != nil {
		f.t.Errorf("Unexpected error: %v", err)
	}
}

func (f *fakePodWorkers) ForgetNonExistingPodWorkers(desiredPods map[types.UID]empty) {}

type TestingInterface interface {
	Errorf(format string, args ...interface{})
}
