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

package multipoint

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// CommunicatingPlugin is an example of a plugin that implements two
// extension points. It communicates through state with another function.
type CommunicatingPlugin struct{}

var _ framework.ReservePlugin = CommunicatingPlugin{}
var _ framework.PreBindPlugin = CommunicatingPlugin{}

// Name is the name of the plugin used in Registry and configurations.
const Name = "multipoint-communicating-plugin"

// Name returns name of the plugin. It is used in logs, etc.
func (mc CommunicatingPlugin) Name() string {
	return Name
}

type stateData struct {
	data string
}

func (s *stateData) Clone() framework.StateData {
	copy := &stateData{
		data: s.data,
	}
	return copy
}

// Reserve is the function invoked by the framework at "reserve" extension point.
func (mc CommunicatingPlugin) Reserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	if pod == nil {
		return framework.NewStatus(framework.Error, "pod cannot be nil")
	}
	if pod.Name == "my-test-pod" {
		state.Write(framework.StateKey(pod.Name), &stateData{data: "never bind"})
	}
	return nil
}

// Unreserve is the function invoked by the framework when any error happens
// during "reserve" extension point or later.
func (mc CommunicatingPlugin) Unreserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
	if pod.Name == "my-test-pod" {
		// The pod is at the end of its lifecycle -- let's clean up the allocated
		// resources. In this case, our clean up is simply deleting the key written
		// in the Reserve operation.
		state.Delete(framework.StateKey(pod.Name))
	}
}

// PreBind is the function invoked by the framework at "prebind" extension point.
func (mc CommunicatingPlugin) PreBind(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	if pod == nil {
		return framework.NewStatus(framework.Error, "pod cannot be nil")
	}
	if v, e := state.Read(framework.StateKey(pod.Name)); e == nil {
		if value, ok := v.(*stateData); ok && value.data == "never bind" {
			return framework.NewStatus(framework.Unschedulable, "pod is not permitted")
		}
	}
	return nil
}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ *runtime.Unknown, _ framework.Handle) (framework.Plugin, error) {
	return &CommunicatingPlugin{}, nil
}
