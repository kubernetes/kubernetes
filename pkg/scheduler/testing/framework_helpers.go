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

package testing

import (
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// RegisterFilterPluginFunc is a function signature used in method RegisterFilterPlugin()
// to register a Filter Plugin to a given registry.
type RegisterFilterPluginFunc func(reg *framework.Registry, plugins *schedulerapi.Plugins, pluginConfigs []schedulerapi.PluginConfig)

// RegisterFilterPlugin returns a function to register a Filter Plugin to a given registry.
func RegisterFilterPlugin(pluginName string, pluginNewFunc framework.PluginFactory) RegisterFilterPluginFunc {
	return func(reg *framework.Registry, plugins *schedulerapi.Plugins, pluginConfigs []schedulerapi.PluginConfig) {
		reg.Register(pluginName, pluginNewFunc)
		plugins.Filter.Enabled = append(plugins.Filter.Enabled, schedulerapi.Plugin{Name: pluginName})
		//lint:ignore SA4006 this value of pluginConfigs is never used.
		//lint:ignore SA4010 this result of append is never used.
		pluginConfigs = append(pluginConfigs, schedulerapi.PluginConfig{Name: pluginName})
	}
}

// RegisterScorePluginFunc is a function signature used in method RegisterScorePlugin()
// to register a Score Plugin to a given registry.
type RegisterScorePluginFunc func(reg *framework.Registry, plugins *schedulerapi.Plugins, pluginConfigs []schedulerapi.PluginConfig)

// RegisterScorePlugin returns a function to register a Score Plugin to a given registry.
func RegisterScorePlugin(pluginName string, pluginNewFunc framework.PluginFactory, weight int32) RegisterScorePluginFunc {
	return func(reg *framework.Registry, plugins *schedulerapi.Plugins, pluginConfigs []schedulerapi.PluginConfig) {
		reg.Register(pluginName, pluginNewFunc)
		plugins.Score.Enabled = append(plugins.Score.Enabled, schedulerapi.Plugin{Name: pluginName, Weight: weight})
		//lint:ignore SA4006 this value of pluginConfigs is never used.
		//lint:ignore SA4010 this result of append is never used.
		pluginConfigs = append(pluginConfigs, schedulerapi.PluginConfig{Name: pluginName})
	}
}
