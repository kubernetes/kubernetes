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
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-scheduler/config/v1beta2"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
)

var configDecoder = scheme.Codecs.UniversalDecoder()

// NewFramework creates a Framework from the register functions and options.
func NewFramework(fns []RegisterPluginFunc, profileName string, stopCh <-chan struct{}, opts ...runtime.Option) (framework.Framework, error) {
	registry := runtime.Registry{}
	profile := &schedulerapi.KubeSchedulerProfile{
		SchedulerName: profileName,
		Plugins:       &schedulerapi.Plugins{},
	}
	for _, f := range fns {
		f(&registry, profile)
	}
	return runtime.NewFramework(registry, profile, stopCh, opts...)
}

// RegisterPluginFunc is a function signature used in method RegisterFilterPlugin()
// to register a Filter Plugin to a given registry.
type RegisterPluginFunc func(reg *runtime.Registry, profile *schedulerapi.KubeSchedulerProfile)

// RegisterQueueSortPlugin returns a function to register a QueueSort Plugin to a given registry.
func RegisterQueueSortPlugin(pluginName string, pluginNewFunc runtime.PluginFactory) RegisterPluginFunc {
	return RegisterPluginAsExtensions(pluginName, pluginNewFunc, "QueueSort")
}

// RegisterPreFilterPlugin returns a function to register a PreFilter Plugin to a given registry.
func RegisterPreFilterPlugin(pluginName string, pluginNewFunc runtime.PluginFactory) RegisterPluginFunc {
	return RegisterPluginAsExtensions(pluginName, pluginNewFunc, "PreFilter")
}

// RegisterFilterPlugin returns a function to register a Filter Plugin to a given registry.
func RegisterFilterPlugin(pluginName string, pluginNewFunc runtime.PluginFactory) RegisterPluginFunc {
	return RegisterPluginAsExtensions(pluginName, pluginNewFunc, "Filter")
}

// RegisterReservePlugin returns a function to register a Reserve Plugin to a given registry.
func RegisterReservePlugin(pluginName string, pluginNewFunc runtime.PluginFactory) RegisterPluginFunc {
	return RegisterPluginAsExtensions(pluginName, pluginNewFunc, "Reserve")
}

// RegisterPermitPlugin returns a function to register a Permit Plugin to a given registry.
func RegisterPermitPlugin(pluginName string, pluginNewFunc runtime.PluginFactory) RegisterPluginFunc {
	return RegisterPluginAsExtensions(pluginName, pluginNewFunc, "Permit")
}

// RegisterPreBindPlugin returns a function to register a PreBind Plugin to a given registry.
func RegisterPreBindPlugin(pluginName string, pluginNewFunc runtime.PluginFactory) RegisterPluginFunc {
	return RegisterPluginAsExtensions(pluginName, pluginNewFunc, "PreBind")
}

// RegisterScorePlugin returns a function to register a Score Plugin to a given registry.
func RegisterScorePlugin(pluginName string, pluginNewFunc runtime.PluginFactory, weight int32) RegisterPluginFunc {
	return RegisterPluginAsExtensionsWithWeight(pluginName, weight, pluginNewFunc, "Score")
}

// RegisterPreScorePlugin returns a function to register a Score Plugin to a given registry.
func RegisterPreScorePlugin(pluginName string, pluginNewFunc runtime.PluginFactory) RegisterPluginFunc {
	return RegisterPluginAsExtensions(pluginName, pluginNewFunc, "PreScore")
}

// RegisterBindPlugin returns a function to register a Bind Plugin to a given registry.
func RegisterBindPlugin(pluginName string, pluginNewFunc runtime.PluginFactory) RegisterPluginFunc {
	return RegisterPluginAsExtensions(pluginName, pluginNewFunc, "Bind")
}

// RegisterPluginAsExtensions returns a function to register a Plugin as given extensionPoints to a given registry.
func RegisterPluginAsExtensions(pluginName string, pluginNewFunc runtime.PluginFactory, extensions ...string) RegisterPluginFunc {
	return RegisterPluginAsExtensionsWithWeight(pluginName, 1, pluginNewFunc, extensions...)
}

// RegisterPluginAsExtensionsWithWeight returns a function to register a Plugin as given extensionPoints with weight to a given registry.
func RegisterPluginAsExtensionsWithWeight(pluginName string, weight int32, pluginNewFunc runtime.PluginFactory, extensions ...string) RegisterPluginFunc {
	return func(reg *runtime.Registry, profile *schedulerapi.KubeSchedulerProfile) {
		reg.Register(pluginName, pluginNewFunc)
		for _, extension := range extensions {
			ps := getPluginSetByExtension(profile.Plugins, extension)
			if ps == nil {
				continue
			}
			ps.Enabled = append(ps.Enabled, schedulerapi.Plugin{Name: pluginName, Weight: weight})
		}
		// Use defaults from latest config API version.
		var gvk schema.GroupVersionKind
		gvk = v1beta2.SchemeGroupVersion.WithKind(pluginName + "Args")
		if args, _, err := configDecoder.Decode(nil, &gvk, nil); err == nil {
			profile.PluginConfig = append(profile.PluginConfig, schedulerapi.PluginConfig{
				Name: pluginName,
				Args: args,
			})
		}
	}
}

func getPluginSetByExtension(plugins *schedulerapi.Plugins, extension string) *schedulerapi.PluginSet {
	switch extension {
	case "QueueSort":
		return &plugins.QueueSort
	case "Filter":
		return &plugins.Filter
	case "PreFilter":
		return &plugins.PreFilter
	case "PreScore":
		return &plugins.PreScore
	case "Score":
		return &plugins.Score
	case "Bind":
		return &plugins.Bind
	case "Reserve":
		return &plugins.Reserve
	case "Permit":
		return &plugins.Permit
	case "PreBind":
		return &plugins.PreBind
	case "PostBind":
		return &plugins.PostBind
	default:
		return nil
	}
}
