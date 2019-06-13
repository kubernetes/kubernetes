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
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
)

// framework is the component responsible for initializing and running scheduler
// plugins.
type framework struct {
	registry         Registry
	nodeInfoSnapshot *cache.NodeInfoSnapshot
	waitingPods      *waitingPodsMap
	plugins          map[string]Plugin // a map of initialized plugins. Plugin name:plugin instance.
	queueSortPlugins []QueueSortPlugin
	reservePlugins   []ReservePlugin
	prebindPlugins   []PrebindPlugin
	postbindPlugins  []PostbindPlugin
	unreservePlugins []UnreservePlugin
	permitPlugins    []PermitPlugin
}

const (
	// Specifies the maximum timeout a permit plugin can return.
	maxTimeout time.Duration = 15 * time.Minute
)

var _ = Framework(&framework{})

// NewFramework initializes plugins given the configuration and the registry.
func NewFramework(r Registry, plugins *config.Plugins, args []config.PluginConfig) (Framework, error) {
	f := &framework{
		registry:         r,
		nodeInfoSnapshot: cache.NewNodeInfoSnapshot(),
		plugins:          make(map[string]Plugin),
		waitingPods:      newWaitingPodsMap(),
	}
	if plugins == nil {
		return f, nil
	}

	// get needed plugins from config
	pg := pluginsNeeded(plugins)
	if len(pg) == 0 {
		return f, nil
	}

	pluginConfig := pluginNameToConfig(args)
	for name, factory := range r {
		// initialize only needed plugins
		if _, ok := pg[name]; !ok {
			continue
		}

		// find the config args of a plugin
		pc := pluginConfig[name]

		p, err := factory(pc, f)
		if err != nil {
			return nil, fmt.Errorf("error initializing plugin %v: %v", name, err)
		}
		f.plugins[name] = p
	}

	if plugins.Reserve != nil {
		for _, r := range plugins.Reserve.Enabled {
			if pg, ok := f.plugins[r.Name]; ok {
				p, ok := pg.(ReservePlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %v does not extend reserve plugin", r.Name)
				}
				f.reservePlugins = append(f.reservePlugins, p)
			} else {
				return nil, fmt.Errorf("reserve plugin %v does not exist", r.Name)
			}
		}
	}

	if plugins.PreBind != nil {
		for _, pb := range plugins.PreBind.Enabled {
			if pg, ok := f.plugins[pb.Name]; ok {
				p, ok := pg.(PrebindPlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %v does not extend prebind plugin", pb.Name)
				}
				f.prebindPlugins = append(f.prebindPlugins, p)
			} else {
				return nil, fmt.Errorf("prebind plugin %v does not exist", pb.Name)
			}
		}
	}

	if plugins.PostBind != nil {
		for _, pb := range plugins.PostBind.Enabled {
			if pg, ok := f.plugins[pb.Name]; ok {
				p, ok := pg.(PostbindPlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %v does not extend postbind plugin", pb.Name)
				}
				f.postbindPlugins = append(f.postbindPlugins, p)
			} else {
				return nil, fmt.Errorf("postbind plugin %v does not exist", pb.Name)
			}
		}
	}

	if plugins.Unreserve != nil {
		for _, ur := range plugins.Unreserve.Enabled {
			if pg, ok := f.plugins[ur.Name]; ok {
				p, ok := pg.(UnreservePlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %v does not extend unreserve plugin", ur.Name)
				}
				f.unreservePlugins = append(f.unreservePlugins, p)
			} else {
				return nil, fmt.Errorf("unreserve plugin %v does not exist", ur.Name)
			}
		}
	}

	if plugins.Permit != nil {
		for _, pr := range plugins.Permit.Enabled {
			if pg, ok := f.plugins[pr.Name]; ok {
				p, ok := pg.(PermitPlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %v does not extend permit plugin", pr.Name)
				}
				f.permitPlugins = append(f.permitPlugins, p)
			} else {
				return nil, fmt.Errorf("permit plugin %v does not exist", pr.Name)
			}
		}
	}

	if plugins.QueueSort != nil {
		for _, qs := range plugins.QueueSort.Enabled {
			if pg, ok := f.plugins[qs.Name]; ok {
				p, ok := pg.(QueueSortPlugin)
				if !ok {
					return nil, fmt.Errorf("plugin %v does not extend queue sort plugin", qs.Name)
				}
				f.queueSortPlugins = append(f.queueSortPlugins, p)
				if len(f.queueSortPlugins) > 1 {
					return nil, fmt.Errorf("only one queue sort plugin can be enabled")
				}
			} else {
				return nil, fmt.Errorf("queue sort plugin %v does not exist", qs.Name)
			}
		}
	}

	return f, nil
}

// QueueSortFunc returns the function to sort pods in scheduling queue
func (f *framework) QueueSortFunc() LessFunc {
	if len(f.queueSortPlugins) == 0 {
		return nil
	}

	// Only one QueueSort plugin can be enabled.
	return f.queueSortPlugins[0].Less
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

// RunPostbindPlugins runs the set of configured postbind plugins.
func (f *framework) RunPostbindPlugins(
	pc *PluginContext, pod *v1.Pod, nodeName string) {
	for _, pl := range f.postbindPlugins {
		pl.Postbind(pc, pod, nodeName)
	}
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

// RunUnreservePlugins runs the set of configured unreserve plugins.
func (f *framework) RunUnreservePlugins(
	pc *PluginContext, pod *v1.Pod, nodeName string) {
	for _, pl := range f.unreservePlugins {
		pl.Unreserve(pc, pod, nodeName)
	}
}

// RunPermitPlugins runs the set of configured permit plugins. If any of these
// plugins returns a status other than "Success" or "Wait", it does not continue
// running the remaining plugins and returns an error. Otherwise, if any of the
// plugins returns "Wait", then this function will block for the timeout period
// returned by the plugin, if the time expires, then it will return an error.
// Note that if multiple plugins asked to wait, then we wait for the minimum
// timeout duration.
func (f *framework) RunPermitPlugins(
	pc *PluginContext, pod *v1.Pod, nodeName string) *Status {
	timeout := maxTimeout
	statusCode := Success
	for _, pl := range f.permitPlugins {
		status, d := pl.Permit(pc, pod, nodeName)
		if !status.IsSuccess() {
			if status.Code() == Unschedulable {
				msg := fmt.Sprintf("rejected by %v at permit: %v", pl.Name(), status.Message())
				klog.V(4).Infof(msg)
				return NewStatus(status.Code(), msg)
			}
			if status.Code() == Wait {
				// Use the minimum timeout duration.
				if timeout > d {
					timeout = d
				}
				statusCode = Wait
			} else {
				msg := fmt.Sprintf("error while running %v permit plugin for pod %v: %v", pl.Name(), pod.Name, status.Message())
				klog.Error(msg)
				return NewStatus(Error, msg)
			}
		}
	}

	// We now wait for the minimum duration if at least one plugin asked to
	// wait (and no plugin rejected the pod)
	if statusCode == Wait {
		w := newWaitingPod(pod)
		f.waitingPods.add(w)
		defer f.waitingPods.remove(pod.UID)
		timer := time.NewTimer(timeout)
		klog.V(4).Infof("waiting for %v for pod %v at permit", timeout, pod.Name)
		select {
		case <-timer.C:
			msg := fmt.Sprintf("pod %v rejected due to timeout after waiting %v at permit", pod.Name, timeout)
			klog.V(4).Infof(msg)
			return NewStatus(Unschedulable, msg)
		case s := <-w.s:
			if !s.IsSuccess() {
				if s.Code() == Unschedulable {
					msg := fmt.Sprintf("rejected while waiting at permit: %v", s.Message())
					klog.V(4).Infof(msg)
					return NewStatus(s.Code(), msg)
				}
				msg := fmt.Sprintf("error received while waiting at permit for pod %v: %v", pod.Name, s.Message())
				klog.Error(msg)
				return NewStatus(Error, msg)
			}
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

// IterateOverWaitingPods acquires a read lock and iterates over the WaitingPods map.
func (f *framework) IterateOverWaitingPods(callback func(WaitingPod)) {
	f.waitingPods.iterate(callback)
}

// GetWaitingPod returns a reference to a WaitingPod given its UID.
func (f *framework) GetWaitingPod(uid types.UID) WaitingPod {
	return f.waitingPods.get(uid)
}

func pluginNameToConfig(args []config.PluginConfig) map[string]*runtime.Unknown {
	pc := make(map[string]*runtime.Unknown, 0)
	for _, p := range args {
		pc[p.Name] = &p.Args
	}
	return pc
}

func pluginsNeeded(plugins *config.Plugins) map[string]struct{} {
	pgMap := make(map[string]struct{}, 0)

	if plugins == nil {
		return pgMap
	}

	find := func(pgs *config.PluginSet) {
		if pgs == nil {
			return
		}
		for _, pg := range pgs.Enabled {
			pgMap[pg.Name] = struct{}{}
		}
	}
	find(plugins.QueueSort)
	find(plugins.PreFilter)
	find(plugins.Filter)
	find(plugins.PostFilter)
	find(plugins.Score)
	find(plugins.NormalizeScore)
	find(plugins.Reserve)
	find(plugins.Permit)
	find(plugins.PreBind)
	find(plugins.Bind)
	find(plugins.PostBind)
	find(plugins.Unreserve)

	return pgMap
}
