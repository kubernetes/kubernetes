/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package container

import (
	"fmt"
	"time"
)

// Specified a policy for garbage collecting containers.
type ContainerGCPolicy struct {
	// Minimum age at which a container can be garbage collected, zero for no limit.
	MinAge time.Duration

	// Max number of dead containers any single pod (UID, container name) pair is
	// allowed to have, less than zero for no limit.
	MaxPerPodContainer int

	// Max number of total dead containers, less than zero for no limit.
	MaxContainers int
}

// Manages garbage collection of dead containers.
//
// Implementation is thread-compatible.
type ContainerGC interface {
	// Garbage collect containers.
	GarbageCollect() error
}

// TODO(vmarmol): Preferentially remove pod infra containers.
type realContainerGC struct {
	// Container runtime
	runtime Runtime

	// Policy for garbage collection.
	policy ContainerGCPolicy
}

// New ContainerGC instance with the specified policy.
func NewContainerGC(runtime Runtime, policy ContainerGCPolicy) (ContainerGC, error) {
	if policy.MinAge < 0 {
		return nil, fmt.Errorf("invalid minimum garbage collection age: %v", policy.MinAge)
	}

	return &realContainerGC{
		runtime: runtime,
		policy:  policy,
	}, nil
}

func (cgc *realContainerGC) GarbageCollect() error {
	return cgc.runtime.GarbageCollect(cgc.policy)
}
