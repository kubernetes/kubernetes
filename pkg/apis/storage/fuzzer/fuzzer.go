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

package fuzzer

import (
	"fmt"

	fuzz "github.com/google/gofuzz"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/storage"
)

// Funcs returns the fuzzer functions for the storage api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *storage.StorageClass, c fuzz.Continue) {
			c.FuzzNoCustom(obj) // fuzz self without calling this function again
			reclamationPolicies := []api.PersistentVolumeReclaimPolicy{api.PersistentVolumeReclaimDelete, api.PersistentVolumeReclaimRetain}
			obj.ReclaimPolicy = &reclamationPolicies[c.Rand.Intn(len(reclamationPolicies))]
			bindingModes := []storage.VolumeBindingMode{storage.VolumeBindingImmediate, storage.VolumeBindingWaitForFirstConsumer}
			obj.VolumeBindingMode = &bindingModes[c.Rand.Intn(len(bindingModes))]
		},
		func(obj *storage.CSIDriver, c fuzz.Continue) {
			c.FuzzNoCustom(obj) // fuzz self without calling this function again

			// Custom fuzzing for volume modes.
			switch c.Rand.Intn(7) {
			case 0:
				obj.Spec.VolumeLifecycleModes = nil
			case 1:
				obj.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{}
			case 2:
				// Invalid mode.
				obj.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
					storage.VolumeLifecycleMode(fmt.Sprintf("%d", c.Rand.Int31())),
				}
			case 3:
				obj.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
					storage.VolumeLifecyclePersistent,
				}
			case 4:
				obj.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
					storage.VolumeLifecycleEphemeral,
				}
			case 5:
				obj.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
					storage.VolumeLifecyclePersistent,
					storage.VolumeLifecycleEphemeral,
				}
			case 6:
				obj.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
					storage.VolumeLifecycleEphemeral,
					storage.VolumeLifecyclePersistent,
				}
			}

			// match defaulting
			if obj.Spec.AttachRequired == nil {
				obj.Spec.AttachRequired = new(bool)
				*(obj.Spec.AttachRequired) = true
			}
			if obj.Spec.PodInfoOnMount == nil {
				obj.Spec.PodInfoOnMount = new(bool)
				*(obj.Spec.PodInfoOnMount) = false
			}
			if obj.Spec.StorageCapacity == nil {
				obj.Spec.StorageCapacity = new(bool)
				*(obj.Spec.StorageCapacity) = false
			}
			if obj.Spec.FSGroupPolicy == nil {
				obj.Spec.FSGroupPolicy = new(storage.FSGroupPolicy)
				*obj.Spec.FSGroupPolicy = storage.ReadWriteOnceWithFSTypeFSGroupPolicy
			}
			if obj.Spec.RequiresRepublish == nil {
				obj.Spec.RequiresRepublish = new(bool)
				*(obj.Spec.RequiresRepublish) = false
			}
			if len(obj.Spec.VolumeLifecycleModes) == 0 {
				obj.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
					storage.VolumeLifecyclePersistent,
				}
			}
			if obj.Spec.SELinuxMount == nil {
				obj.Spec.SELinuxMount = new(bool)
				*(obj.Spec.SELinuxMount) = false
			}
		},
	}
}
