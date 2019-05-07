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
	unreservePlugins []UnreservePlugin
	permitPlugins    []PermitPlugin
}

const (
	// Specifies the maximum timeout a permit plugin can return.
	maxTimeout time.Duration = 15 * time.Minute
)

var _ = Framework(&framework{})

// NewFramework initializes plugins given the configuration and the registry.
func NewFramework(r Registry, _ *runtime.Unknown) (Framework, error) {
	f := &framework{
		registry:         r,
		nodeInfoSnapshot: cache.NewNodeInfoSnapshot(),
		plugins:          make(map[string]Plugin),
		waitingPods:      newWaitingPodsMap(),
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
		if qsp, ok := p.(QueueSortPlugin); ok {
			f.queueSortPlugins = append(f.queueSortPlugins, qsp)
		}

		if rp, ok := p.(ReservePlugin); ok {
			f.reservePlugins = append(f.reservePlugins, rp)
		}
		if pp, ok := p.(PrebindPlugin); ok {
			f.prebindPlugins = append(f.prebindPlugins, pp)
		}
		if up, ok := p.(UnreservePlugin); ok {
			f.unreservePlugins = append(f.unreservePlugins, up)
		}
		if pr, ok := p.(PermitPlugin); ok {
			f.permitPlugins = append(f.permitPlugins, pr)
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
