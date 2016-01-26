/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package predicates

import "fmt"

var (
	ErrExceededMaxPodNumber    = newInsufficientResourceError("PodCount")
	ErrInsufficientFreeCPU     = newInsufficientResourceError("CPU")
	ErrInsufficientFreeMemory  = newInsufficientResourceError("Memory")
	ErrInsufficientFreeDevices = newInsufficientResourceError("Devices")
)

// InsufficientResourceError is an error type that indicates what kind of resource limit is
// hit and caused the unfitting failure.
type InsufficientResourceError struct {
	// ResourceName tells the name of the resource that is insufficient
	ResourceName string
}

func newInsufficientResourceError(resourceName string) *InsufficientResourceError {
	return &InsufficientResourceError{resourceName}
}

func (e *InsufficientResourceError) Error() string {
	return fmt.Sprintf("Node didn't have enough resource: %s", e.ResourceName)
}
