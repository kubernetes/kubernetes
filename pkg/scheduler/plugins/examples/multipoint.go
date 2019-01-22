/*
Copyright 2018 The Kubernetes Authors.

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

package examples

import (
	"fmt"

	"k8s.io/api/core/v1"
	plugins "k8s.io/kubernetes/pkg/scheduler/plugins/v1alpha1"
)

// MultipointCommunicatingPlugin is an example of a plugin that implements two
// extension points. It communicates through pluginContext with another function.
type MultipointCommunicatingPlugin struct{}

var _ = plugins.ReservePlugin(MultipointCommunicatingPlugin{})

// Name returns name of the plugin. It is used in logs, etc.
func (mc MultipointCommunicatingPlugin) Name() string {
	return "multipoint-communicating-plugin"
}

// Reserve is the functions invoked by the framework at "reserve" extension point.
func (mc MultipointCommunicatingPlugin) Reserve(ps plugins.PluginSet, pod *v1.Pod, nodeName string) error {
	if pod == nil {
		return fmt.Errorf("pod cannot be nil")
	}
	if pod.Name == "my-test-pod" {
		ps.Data().Ctx.SyncWrite(plugins.ContextKey(pod.Name), "never bind")
	}
	return nil
}

// Prebind is the functions invoked by the framework at "prebind" extension point.
func (mc MultipointCommunicatingPlugin) Prebind(ps plugins.PluginSet, pod *v1.Pod, nodeName string) (bool, error) {
	if pod == nil {
		return false, fmt.Errorf("pod cannot be nil")
	}
	if v, e := ps.Data().Ctx.SyncRead(plugins.ContextKey(pod.Name)); e == nil && v == "never bind" {
		return false, nil
	}
	return true, nil
}

// NewMultipointCommunicatingPlugin initializes a new plugin and returns it.
func NewMultipointCommunicatingPlugin() *MultipointCommunicatingPlugin {
	return &MultipointCommunicatingPlugin{}
}
