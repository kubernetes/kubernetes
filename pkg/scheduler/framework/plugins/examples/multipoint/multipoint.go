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
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// CommunicatingPlugin is an example of a plugin that implements two
// extension points. It communicates through pluginContext with another function.
type CommunicatingPlugin struct{}

var _ = framework.ReservePlugin(CommunicatingPlugin{})
var _ = framework.PrebindPlugin(CommunicatingPlugin{})

// Name is the name of the plug used in Registry and configurations.
const Name = "multipoint-communicating-plugin"

// Name returns name of the plugin. It is used in logs, etc.
func (mc CommunicatingPlugin) Name() string {
	return Name
}

// Reserve is the functions invoked by the framework at "reserve" extension point.
func (mc CommunicatingPlugin) Reserve(pc *framework.PluginContext, pod *v1.Pod, nodeName string) *framework.Status {
	if pod == nil {
		return framework.NewStatus(framework.Error, "pod cannot be nil")
	}
	if pod.Name == "my-test-pod" {
		pc.Lock()
		pc.Write(framework.ContextKey(pod.Name), "never bind")
		pc.Unlock()
	}
	return nil
}

// Prebind is the functions invoked by the framework at "prebind" extension point.
func (mc CommunicatingPlugin) Prebind(pc *framework.PluginContext, pod *v1.Pod, nodeName string) *framework.Status {
	if pod == nil {
		return framework.NewStatus(framework.Error, "pod cannot be nil")
	}
	pc.RLock()
	defer pc.RUnlock()
	if v, e := pc.Read(framework.ContextKey(pod.Name)); e == nil && v == "never bind" {
		return framework.NewStatus(framework.Unschedulable, "pod is not permitted")
	}
	return nil
}

// New initializes a new plugin and returns it.
func New(_ *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	return &CommunicatingPlugin{}, nil
}
