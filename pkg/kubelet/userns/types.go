/*
Copyright 2025 The Kubernetes Authors.

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

package userns

import "k8s.io/apimachinery/pkg/types"

// Here go types that are common for all supported OS (windows, linux).

type userNsPodsManager interface {
	HandlerSupportsUserNamespaces(runtimeHandler string) (bool, error)
	GetPodDir(podUID types.UID) string
	ListPodsFromDisk() ([]types.UID, error)
	GetKubeletMappings() (uint32, uint32, error)
	GetMaxPods() int
	GetUserNamespacesIDsPerPod() uint32
}
