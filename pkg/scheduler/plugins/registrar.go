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

package plugins

import (
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
	plugins "k8s.io/kubernetes/pkg/scheduler/plugins/v1alpha1"
)

// DefaultPluginSet is the default plugin registrar used by the default scheduler.
type DefaultPluginSet struct {
	data           *plugins.PluginData
	reservePlugins []plugins.ReservePlugin
	prebindPlugins []plugins.PrebindPlugin
}

var _ = plugins.PluginSet(&DefaultPluginSet{})

// ReservePlugins returns a slice of default reserve plugins.
func (r *DefaultPluginSet) ReservePlugins() []plugins.ReservePlugin {
	return r.reservePlugins
}

// PrebindPlugins returns a slice of default prebind plugins.
func (r *DefaultPluginSet) PrebindPlugins() []plugins.PrebindPlugin {
	return r.prebindPlugins
}

// Data returns a pointer to PluginData.
func (r *DefaultPluginSet) Data() *plugins.PluginData {
	return r.data
}

// NewDefaultPluginSet initializes default plugin set and returns its pointer.
func NewDefaultPluginSet(ctx *plugins.PluginContext, schedulerCache *cache.Cache) *DefaultPluginSet {
	defaultRegistrar := DefaultPluginSet{
		data: &plugins.PluginData{
			Ctx:            ctx,
			SchedulerCache: schedulerCache,
		},
	}
	defaultRegistrar.registerReservePlugins()
	defaultRegistrar.registerPrebindPlugins()
	return &defaultRegistrar
}

func (r *DefaultPluginSet) registerReservePlugins() {
	r.reservePlugins = []plugins.ReservePlugin{
		// Init functions of all reserve plugins go here. They are called in the
		// same order that they are registered.
		// Example:
		// examples.NewStatefulMultipointExample(map[int]string{1: "test1", 2: "test2"}),
	}
}

func (r *DefaultPluginSet) registerPrebindPlugins() {
	r.prebindPlugins = []plugins.PrebindPlugin{
		// Init functions of all prebind plugins go here. They are called in the
		// same order that they are registered.
		// Example:
		// examples.NewStatelessPrebindExample(),
	}
}
