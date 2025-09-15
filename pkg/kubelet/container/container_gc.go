/*
Copyright 2014 The Kubernetes Authors.

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
	"context"
	"fmt"
	"time"

	"k8s.io/klog/v2"
)

// GCPolicy specifies a policy for garbage collecting containers.
type GCPolicy struct {
	// Minimum age at which a container can be garbage collected, zero for no limit.
	MinAge time.Duration

	// Max number of dead containers any single pod (UID, container name) pair is
	// allowed to have, less than zero for no limit.
	MaxPerPodContainer int

	// Max number of total dead containers, less than zero for no limit.
	MaxContainers int
}

// GC manages garbage collection of dead containers.
//
// Implementation is thread-compatible.
type GC interface {
	// Garbage collect containers.
	GarbageCollect(ctx context.Context) error
	// Deletes all unused containers, including containers belonging to pods that are terminated but not deleted
	DeleteAllUnusedContainers(ctx context.Context) error
}

// SourcesReadyProvider knows how to determine if configuration sources are ready
type SourcesReadyProvider interface {
	// AllReady returns true if the currently configured sources have all been seen.
	AllReady() bool
}

// TODO(vmarmol): Preferentially remove pod infra containers.
type realContainerGC struct {
	// Container runtime
	runtime Runtime

	// Policy for garbage collection.
	policy GCPolicy

	// sourcesReadyProvider provides the readiness of kubelet configuration sources.
	sourcesReadyProvider SourcesReadyProvider
}

// NewContainerGC creates a new instance of GC with the specified policy.
func NewContainerGC(runtime Runtime, policy GCPolicy, sourcesReadyProvider SourcesReadyProvider) (GC, error) {
	if policy.MinAge < 0 {
		return nil, fmt.Errorf("invalid minimum garbage collection age: %v", policy.MinAge)
	}

	return &realContainerGC{
		runtime:              runtime,
		policy:               policy,
		sourcesReadyProvider: sourcesReadyProvider,
	}, nil
}

func (cgc *realContainerGC) GarbageCollect(ctx context.Context) error {
	return cgc.runtime.GarbageCollect(ctx, cgc.policy, cgc.sourcesReadyProvider.AllReady(), false)
}

func (cgc *realContainerGC) DeleteAllUnusedContainers(ctx context.Context) error {
	klog.InfoS("Attempting to delete unused containers")
	return cgc.runtime.GarbageCollect(ctx, cgc.policy, cgc.sourcesReadyProvider.AllReady(), true)
}
