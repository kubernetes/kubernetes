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

package v1alpha1

import (
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
)

// framework is the component responsible for initializing and running scheduler
// plugins.
type framework struct {
	registry         Registry
	nodeInfoSnapshot *cache.NodeInfoSnapshot
	plugins          map[string]Plugin // a map of initialized plugins. Plugin name:plugin instance.
	reservePlugins   []ReservePlugin
	prebindPlugins   []PrebindPlugin
}

var _ = Framework(&framework{})

// NewFramework initializes plugins given the configuration and the registry.
func NewFramework(r Registry, _ *runtime.Unknown) (Framework, error) {
	f := &framework{
		registry:         r,
		nodeInfoSnapshot: cache.NewNodeInfoSnapshot(),
		plugins:          make(map[string]Plugin),
	}

	// TODO: The framework needs to read the scheduler config and initialize only
	// needed plugins. In this initial version of the code, we initialize all.
	for name, factory := range r {
		// TODO: 'nil' should be replaced by plugin config.
		p, err := factory(nil, f)
		if err != nil {
			return nil, fmt.Errorf("error initializing plugin %v: %v", name, err)
		}
		f.plugins[name] = p

		// TODO: For now, we assume any plugins that implements an extension
		// point wants to be called at that extension point. We should change this
		// later and add these plugins based on the configuration.
		if rp, ok := p.(ReservePlugin); ok {
			f.reservePlugins = append(f.reservePlugins, rp)
		}
		if pp, ok := p.(PrebindPlugin); ok {
			f.prebindPlugins = append(f.prebindPlugins, pp)
		}
	}
	return f, nil
}

// RunPrebindPlugins runs the set of configured prebind plugins. It returns a
// failure (bool) if any of the plugins returns an error. It also returns an
// error containing the rejection message or the error occurred in the plugin.
func (f *framework) RunPrebindPlugins(
	pc *PluginContext, pod *v1.Pod, nodeName string) *Status {
	for _, pl := range f.prebindPlugins {
		status := pl.Prebind(pc, pod, nodeName)
		if !status.IsSuccess() {
			if status.Code() == Unschedulable {
				msg := fmt.Sprintf("rejected by %v at prebind: %v", pl.Name(), status.Message())
				klog.V(4).Infof(msg)
				return NewStatus(status.Code(), msg)
			}
			msg := fmt.Sprintf("error while running %v prebind plugin for pod %v: %v", pl.Name(), pod.Name, status.Message())
			klog.Error(msg)
			return NewStatus(Error, msg)
		}
	}
	return nil
}

// RunReservePlugins runs the set of configured reserve plugins. If any of these
// plugins returns an error, it does not continue running the remaining ones and
// returns the error. In such case, pod will not be scheduled.
func (f *framework) RunReservePlugins(
	pc *PluginContext, pod *v1.Pod, nodeName string) *Status {
	for _, pl := range f.reservePlugins {
		status := pl.Reserve(pc, pod, nodeName)
		if !status.IsSuccess() {
			msg := fmt.Sprintf("error while running %v reserve plugin for pod %v: %v", pl.Name(), pod.Name, status.Message())
			klog.Error(msg)
			return NewStatus(Error, msg)
		}
	}
	return nil
}

// NodeInfoSnapshot returns the latest NodeInfo snapshot. The snapshot
// is taken at the beginning of a scheduling cycle and remains unchanged until a
// pod finishes "Reserve". There is no guarantee that the information remains
// unchanged after "Reserve".
func (f *framework) NodeInfoSnapshot() *cache.NodeInfoSnapshot {
	return f.nodeInfoSnapshot
}
