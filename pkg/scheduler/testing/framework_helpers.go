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

// RegisterPluginFunc is a function signature used in method RegisterFilterPlugin()
// to register a Filter Plugin to a given registry.
type RegisterPluginFunc func(reg *framework.Registry, plugins *schedulerapi.Plugins, pluginConfigs []schedulerapi.PluginConfig)

// RegisterFilterPlugin returns a function to register a Filter Plugin to a given registry.
func RegisterFilterPlugin(pluginName string, pluginNewFunc framework.PluginFactory) RegisterPluginFunc {
	return RegisterPluginAsExtensions(pluginName, 1, pluginNewFunc, "Filter")
}

// RegisterScorePlugin returns a function to register a Score Plugin to a given registry.
func RegisterScorePlugin(pluginName string, pluginNewFunc framework.PluginFactory, weight int32) RegisterPluginFunc {
	return RegisterPluginAsExtensions(pluginName, weight, pluginNewFunc, "Score")
}

// RegisterPluginAsExtensions returns a function to register a Plugin as given extensionPoints to a given registry.
func RegisterPluginAsExtensions(pluginName string, weight int32, pluginNewFunc framework.PluginFactory, extensions ...string) RegisterPluginFunc {
	return func(reg *framework.Registry, plugins *schedulerapi.Plugins, pluginConfigs []schedulerapi.PluginConfig) {
		reg.Register(pluginName, pluginNewFunc)
		for _, extension := range extensions {
			pluginSet := getPluginSetByExtension(plugins, extension)
			if pluginSet == nil {
				continue
			}
			pluginSet.Enabled = append(pluginSet.Enabled, schedulerapi.Plugin{Name: pluginName, Weight: weight})
		}
		//lint:ignore SA4006 this value of pluginConfigs is never used.
		//lint:ignore SA4010 this result of append is never used.
		pluginConfigs = append(pluginConfigs, schedulerapi.PluginConfig{Name: pluginName})
	}
}

func getPluginSetByExtension(plugins *schedulerapi.Plugins, extension string) *schedulerapi.PluginSet {
	switch extension {
	case "Filter":
		return plugins.Filter
	case "PreFilter":
		return plugins.PreFilter
	case "PostFilter":
		return plugins.PostFilter
	case "Score":
		return plugins.Score
	case "Bind":
		return plugins.Bind
	case "Reserve":
		return plugins.Reserve
	case "Permit":
		return plugins.Permit
	default:
		return nil
	}
}
