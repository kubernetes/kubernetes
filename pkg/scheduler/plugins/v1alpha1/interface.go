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

// This file defines the scheduling framework plugin interfaces.

package v1alpha1

import (
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
)

// PluginData carries information that plugins may need.
type PluginData struct {
	Ctx            *PluginContext
	SchedulerCache *cache.Cache
	// We may want to add the scheduling queue here too.
}

// Plugin is the parent type for all the scheduling framework plugins.
type Plugin interface {
	Name() string
}

// ReservePlugin is an interface for Reserve plugins. These plugins are called
// at the reservation point, AKA "assume". These are meant to updated the state
// of the plugin. They do not return any value (other than error).
type ReservePlugin interface {
	Plugin
	// Reserve is called by the scheduling framework when the scheduler cache is
	// updated.
	Reserve(ps PluginSet, p *v1.Pod, nodeName string) error
}

// PrebindPlugin is an interface that must be implemented by "prebind" plugins.
// These plugins are called before a pod being scheduled
type PrebindPlugin interface {
	Plugin
	// Prebind is called before binding a pod. All prebind plugins must return
	// or the pod will not be sent for binding.
	Prebind(ps PluginSet, p *v1.Pod, nodeName string) (bool, error)
}

// PluginSet registers plugins used by the scheduling framework.
// The plugins registered are called at specified points in an scheduling cycle.
type PluginSet interface {
	Data() *PluginData
	ReservePlugins() []ReservePlugin
	PrebindPlugins() []PrebindPlugin
}
