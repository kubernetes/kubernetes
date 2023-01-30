/*
Copyright 2023 The Kubernetes Authors.

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

package config

import (
	"encoding/json"
	"hash"

	v1 "k8s.io/api/core/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
)

// deepHashPod writes specified pod to hash using the json library
// ensuring the hash does not change when a pointer changes or adds a new field.
func deepHashPod(hasher hash.Hash, pod *api.Pod) {
	var podToWrite v1.Pod
	corev1.Convert_core_Pod_To_v1_Pod(pod, &podToWrite, nil)
	hasher.Reset()
	json.NewEncoder(hasher).Encode(podToWrite)
}
