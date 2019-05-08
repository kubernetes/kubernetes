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

// This file defines the scheduling framework plugin interfaces.

package v1alpha1

import (
	"errors"

	"k8s.io/api/core/v1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
)

// Code is the Status code/type which is returned from plugins.
type Code int

// These are predefined codes used in a Status.
const (
	// Success means that plugin ran correctly and found pod schedulable.
	// NOTE: A nil status is also considered as "Success".
	Success Code = 0
	// Error is used for internal plugin errors, unexpected input, etc.
	Error Code = 1
	// Unschedulable is used when a plugin finds a pod unschedulable.
	// The accompanying status message should explain why the pod is unschedulable.
	Unschedulable Code = 2
)

// Status indicates the result of running a plugin. It consists of a code and a
// message. When the status code is not `Success`, the status message should
// explain why.
// NOTE: A nil Status is also considered as Success.
type Status struct {
	code    Code
	message string
}

// Code returns code of the Status.
func (s *Status) Code() Code {
	if s == nil {
		return Success
	}
	return s.code
}

// Message returns message of the Status.
func (s *Status) Message() string {
	return s.message
}

// IsSuccess returns true if and only if "Status" is nil or Code is "Success".
func (s *Status) IsSuccess() bool {
	if s == nil || s.code == Success {
		return true
	}
	return false
}

// AsError returns an "error" object with the same message as that of the Status.
func (s *Status) AsError() error {
	if s.IsSuccess() {
		return nil
	}
	return errors.New(s.message)
}

// NewStatus makes a Status out of the given arguments and returns its pointer.
func NewStatus(code Code, msg string) *Status {
	return &Status{
		code:    code,
		message: msg,
	}
}

// Plugin is the parent type for all the scheduling framework plugins.
type Plugin interface {
	Name() string
}

// ReservePlugin is an interface for Reserve plugins. These plugins are called
// at the reservation point. These are meant to update the state of the plugin.
// This concept used to be called 'assume' in the original scheduler.
// These plugins should return only Success or Error in Status.code. However,
// the scheduler accepts other valid codes as well. Anything other than Success
// will lead to rejection of the pod.
type ReservePlugin interface {
	Plugin
	// Reserve is called by the scheduling framework when the scheduler cache is
	// updated.
	Reserve(pc *PluginContext, p *v1.Pod, nodeName string) *Status
}

// PrebindPlugin is an interface that must be implemented by "prebind" plugins.
// These plugins are called before a pod being scheduled
type PrebindPlugin interface {
	Plugin
	// Prebind is called before binding a pod. All prebind plugins must return
	// success or the pod will be rejected and won't be sent for binding.
	Prebind(pc *PluginContext, p *v1.Pod, nodeName string) *Status
}

// UnreservePlugin is an interface for Unreserve plugins. This is an informational
// extension point. If a pod was reserved and then rejected in a later phase, then
// un-reserve plugins will be notified. Un-reserve plugins should clean up state
// associated with the reserved Pod.
type UnreservePlugin interface {
	Plugin
	// Unreserve is called by the scheduling framework when a reserved pod was
	// rejected in a later phase.
	Unreserve(pc *PluginContext, p *v1.Pod, nodeName string)
}

// Framework manages the set of plugins in use by the scheduling framework.
// Configured plugins are called at specified points in a scheduling context.
type Framework interface {
	FrameworkHandle
	// RunPrebindPlugins runs the set of configured prebind plugins. It returns
	// *Status and its code is set to non-success if any of the plugins returns
	// anything but Success. If the Status code is "Unschedulable", it is
	// considered as a scheduling check failure, otherwise, it is considered as an
	// internal error. In either case the pod is not going to be bound.
	RunPrebindPlugins(pc *PluginContext, pod *v1.Pod, nodeName string) *Status

	// RunReservePlugins runs the set of configured reserve plugins. If any of these
	// plugins returns an error, it does not continue running the remaining ones and
	// returns the error. In such case, pod will not be scheduled.
	RunReservePlugins(pc *PluginContext, pod *v1.Pod, nodeName string) *Status

	// RunUnreservePlugins runs the set of configured unreserve plugins.
	RunUnreservePlugins(pc *PluginContext, pod *v1.Pod, nodeName string)
}

// FrameworkHandle provides data and some tools that plugins can use. It is
// passed to the plugin factories at the time of plugin initialization. Plugins
// must store and use this handle to call framework functions.
type FrameworkHandle interface {
	// NodeInfoSnapshot return the latest NodeInfo snapshot. The snapshot
	// is taken at the beginning of a scheduling cycle and remains unchanged until
	// a pod finishes "Reserve" point. There is no guarantee that the information
	// remains unchanged in the binding phase of scheduling.
	NodeInfoSnapshot() *internalcache.NodeInfoSnapshot
}
