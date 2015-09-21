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
	"sync"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
)

// volumeManager manages the volumes for the pods running on the kubelet.
// Currently it only does book keeping, but it can be expanded to
// take care of the volumePlugins.
type volumeManager struct {
	lock       sync.RWMutex
	volumeMaps map[types.UID]kubecontainer.VolumeMap
}

func newVolumeManager() *volumeManager {
	vm := &volumeManager{}
	vm.volumeMaps = make(map[types.UID]kubecontainer.VolumeMap)
	return vm
}

// SetVolumes sets the volume map for a pod.
// TODO(yifan): Currently we assume the volume is already mounted, so we only do a book keeping here.
func (vm *volumeManager) SetVolumes(podUID types.UID, podVolumes kubecontainer.VolumeMap) {
	vm.lock.Lock()
	defer vm.lock.Unlock()
	vm.volumeMaps[podUID] = podVolumes
}

// GetVolumes returns the volume map which are already mounted on the host machine
// for a pod.
func (vm *volumeManager) GetVolumes(podUID types.UID) (kubecontainer.VolumeMap, bool) {
	vm.lock.RLock()
	defer vm.lock.RUnlock()
	vol, ok := vm.volumeMaps[podUID]
	return vol, ok
}

// DeleteVolumes removes the reference to a volume map for a pod.
func (vm *volumeManager) DeleteVolumes(podUID types.UID) {
	vm.lock.Lock()
	defer vm.lock.Unlock()
	delete(vm.volumeMaps, podUID)
}
