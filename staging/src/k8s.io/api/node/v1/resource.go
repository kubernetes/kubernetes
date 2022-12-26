/*
Copyright 2022 The Kubernetes Authors.

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

package v1

import (
	corev1 "k8s.io/api/core/v1"
	resource "k8s.io/apimachinery/pkg/api/resource"
)

// Swap, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
// Supported if the feature gate NodeSwap is enabled
const ResourceSwap corev1.ResourceName = "swap"

// Swap returns the Swap limit if specified.
func Swap(rl corev1.ResourceList) *resource.Quantity {
	return rl.Name(ResourceSwap, resource.BinarySI)
}
