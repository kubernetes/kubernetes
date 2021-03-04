/*
Copyright 2019 The Kubernetes Authors.

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

package stateful

import (
	"context"
	"fmt"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// MultipointExample is an example plugin that is executed at multiple extension points.
// This plugin is stateful. It receives arguments at initialization (NewMultipointPlugin)
// and changes its state when it is executed.
type MultipointExample struct {
	executionPoints []string
	mu              sync.RWMutex
}

var _ framework.ReservePlugin = &MultipointExample{}
var _ framework.PreBindPlugin = &MultipointExample{}

// Name is the name of the plug used in Registry and configurations.
const Name = "multipoint-plugin-example"

// Name returns name of the plugin. It is used in logs, etc.
func (mp *MultipointExample) Name() string {
	return Name
}

// Reserve is the function invoked by the framework at "reserve" extension
// point. In this trivial example, the Reserve method allocates an array of
// strings.
func (mp *MultipointExample) Reserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	// Reserve is not called concurrently, and so we don't need to lock.
	mp.executionPoints = append(mp.executionPoints, "reserve")
	return nil
}

// Unreserve is the function invoked by the framework when any error happens
// during "reserve" extension point or later. In this example, the Unreserve
// method loses its reference to the string slice, allowing it to be garbage
// collected, and thereby "unallocating" the reserved resources.
func (mp *MultipointExample) Unreserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
	// Unlike Reserve, the Unreserve method may be called concurrently since
	// there is no guarantee that there will only one unreserve operation at any
	// given point in time (for example, during the binding cycle).
	mp.mu.Lock()
	defer mp.mu.Unlock()
	mp.executionPoints = nil
}

// PreBind is the function invoked by the framework at "prebind" extension
// point.
func (mp *MultipointExample) PreBind(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	// PreBind could be called concurrently for different pods.
	mp.mu.Lock()
	defer mp.mu.Unlock()
	mp.executionPoints = append(mp.executionPoints, "pre-bind")
	if pod == nil {
		return framework.NewStatus(framework.Error, "pod must not be nil")
	}
	return nil
}

// New initializes a new plugin and returns it.
func New(config *runtime.Unknown, _ framework.Handle) (framework.Plugin, error) {
	if config == nil {
		klog.Error("MultipointExample configuration cannot be empty")
		return nil, fmt.Errorf("MultipointExample configuration cannot be empty")
	}
	mp := MultipointExample{}
	return &mp, nil
}
