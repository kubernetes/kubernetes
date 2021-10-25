//go:build linux
// +build linux

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

package nodestatus

import (
	"github.com/google/cadvisor/machine"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
)

func getSwapCapacity() resource.Quantity {
	swapCapacity, err := machine.GetMachineSwapCapacity()
	if err != nil {
		klog.ErrorS(err, "Failed to get swap capacity from cadvisor")
		return *resource.NewQuantity(
			int64(0),
			resource.BinarySI)
	}
	return *resource.NewQuantity(
		int64(swapCapacity),
		resource.BinarySI)
}
