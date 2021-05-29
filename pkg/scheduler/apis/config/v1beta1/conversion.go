/*
Copyright 2020 The Kubernetes Authors.

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

package v1beta1

import (
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/kube-scheduler/config/v1beta1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/utils/pointer"
)

var (
	// pluginArgConversionScheme is a scheme with internal and v1beta1 registered,
	// used for defaulting/converting typed PluginConfig Args.
	// Access via getPluginArgConversionScheme()
	pluginArgConversionScheme     *runtime.Scheme
	initPluginArgConversionScheme sync.Once
)

func getPluginArgConversionScheme() *runtime.Scheme {
	initPluginArgConversionScheme.Do(func() {
		// set up the scheme used for plugin arg conversion
		pluginArgConversionScheme = runtime.NewScheme()
		utilruntime.Must(AddToScheme(pluginArgConversionScheme))
		utilruntime.Must(config.AddToScheme(pluginArgConversionScheme))
	})
	return pluginArgConversionScheme
}

func Convert_v1beta1_Plugins_To_config_Plugins(in *v1beta1.Plugins, out *config.Plugins, s conversion.Scope) error {
	if err := Convert_v1beta1_PluginSet_To_config_PluginSet(in.QueueSort, &out.QueueSort, s); err != nil {
		return err
	}
	if err := Convert_v1beta1_PluginSet_To_config_PluginSet(in.PreFilter, &out.PreFilter, s); err != nil {
		return err
	}
	if err := Convert_v1beta1_PluginSet_To_config_PluginSet(in.Filter, &out.Filter, s); err != nil {
		return err
	}
	if err := Convert_v1beta1_PluginSet_To_config_PluginSet(in.PostFilter, &out.PostFilter, s); err != nil {
		return err
	}
	if err := Convert_v1beta1_PluginSet_To_config_PluginSet(in.PreScore, &out.PreScore, s); err != nil {
		return err
	}
	if err := Convert_v1beta1_PluginSet_To_config_PluginSet(in.Score, &out.Score, s); err != nil {
		return err
	}
	if err := Convert_v1beta1_PluginSet_To_config_PluginSet(in.Reserve, &out.Reserve, s); err != nil {
		return err
	}
	if err := Convert_v1beta1_PluginSet_To_config_PluginSet(in.Permit, &out.Permit, s); err != nil {
		return err
	}
	if err := Convert_v1beta1_PluginSet_To_config_PluginSet(in.PreBind, &out.PreBind, s); err != nil {
		return err
	}
	if err := Convert_v1beta1_PluginSet_To_config_PluginSet(in.Bind, &out.Bind, s); err != nil {
		return err
	}
	if err := Convert_v1beta1_PluginSet_To_config_PluginSet(in.PostBind, &out.PostBind, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1beta1_PluginSet_To_config_PluginSet(in *v1beta1.PluginSet, out *config.PluginSet, s conversion.Scope) error {
	if in == nil {
		return nil
	}
	return autoConvert_v1beta1_PluginSet_To_config_PluginSet(in, out, s)
}

func Convert_config_Plugins_To_v1beta1_Plugins(in *config.Plugins, out *v1beta1.Plugins, s conversion.Scope) error {
	out.QueueSort = new(v1beta1.PluginSet)
	if err := Convert_config_PluginSet_To_v1beta1_PluginSet(&in.QueueSort, out.QueueSort, s); err != nil {
		return err
	}
	out.PreFilter = new(v1beta1.PluginSet)
	if err := Convert_config_PluginSet_To_v1beta1_PluginSet(&in.PreFilter, out.PreFilter, s); err != nil {
		return err
	}
	out.Filter = new(v1beta1.PluginSet)
	if err := Convert_config_PluginSet_To_v1beta1_PluginSet(&in.Filter, out.Filter, s); err != nil {
		return err
	}
	out.PostFilter = new(v1beta1.PluginSet)
	if err := Convert_config_PluginSet_To_v1beta1_PluginSet(&in.PostFilter, out.PostFilter, s); err != nil {
		return err
	}
	out.PreScore = new(v1beta1.PluginSet)
	if err := Convert_config_PluginSet_To_v1beta1_PluginSet(&in.PreScore, out.PreScore, s); err != nil {
		return err
	}
	out.Score = new(v1beta1.PluginSet)
	if err := Convert_config_PluginSet_To_v1beta1_PluginSet(&in.Score, out.Score, s); err != nil {
		return err
	}
	out.Reserve = new(v1beta1.PluginSet)
	if err := Convert_config_PluginSet_To_v1beta1_PluginSet(&in.Reserve, out.Reserve, s); err != nil {
		return err
	}
	out.Permit = new(v1beta1.PluginSet)
	if err := Convert_config_PluginSet_To_v1beta1_PluginSet(&in.Permit, out.Permit, s); err != nil {
		return err
	}
	out.PreBind = new(v1beta1.PluginSet)
	if err := Convert_config_PluginSet_To_v1beta1_PluginSet(&in.PreBind, out.PreBind, s); err != nil {
		return err
	}
	out.Bind = new(v1beta1.PluginSet)
	if err := Convert_config_PluginSet_To_v1beta1_PluginSet(&in.Bind, out.Bind, s); err != nil {
		return err
	}
	out.PostBind = new(v1beta1.PluginSet)
	if err := Convert_config_PluginSet_To_v1beta1_PluginSet(&in.PostBind, out.PostBind, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1beta1_KubeSchedulerConfiguration_To_config_KubeSchedulerConfiguration(in *v1beta1.KubeSchedulerConfiguration, out *config.KubeSchedulerConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1beta1_KubeSchedulerConfiguration_To_config_KubeSchedulerConfiguration(in, out, s); err != nil {
		return err
	}
	out.AlgorithmSource.Provider = pointer.StringPtr(v1beta1.SchedulerDefaultProviderName)
	return convertToInternalPluginConfigArgs(out)
}

// convertToInternalPluginConfigArgs converts PluginConfig#Args into internal
// types using a scheme, after applying defaults.
func convertToInternalPluginConfigArgs(out *config.KubeSchedulerConfiguration) error {
	scheme := getPluginArgConversionScheme()
	for i := range out.Profiles {
		for j := range out.Profiles[i].PluginConfig {
			args := out.Profiles[i].PluginConfig[j].Args
			if args == nil {
				continue
			}
			if _, isUnknown := args.(*runtime.Unknown); isUnknown {
				continue
			}
			scheme.Default(args)
			internalArgs, err := scheme.ConvertToVersion(args, config.SchemeGroupVersion)
			if err != nil {
				return fmt.Errorf("converting .Profiles[%d].PluginConfig[%d].Args into internal type: %w", i, j, err)
			}
			out.Profiles[i].PluginConfig[j].Args = internalArgs
		}
	}
	return nil
}

func Convert_config_KubeSchedulerConfiguration_To_v1beta1_KubeSchedulerConfiguration(in *config.KubeSchedulerConfiguration, out *v1beta1.KubeSchedulerConfiguration, s conversion.Scope) error {
	if err := autoConvert_config_KubeSchedulerConfiguration_To_v1beta1_KubeSchedulerConfiguration(in, out, s); err != nil {
		return err
	}
	return convertToExternalPluginConfigArgs(out)
}

// convertToExternalPluginConfigArgs converts PluginConfig#Args into
// external (versioned) types using a scheme.
func convertToExternalPluginConfigArgs(out *v1beta1.KubeSchedulerConfiguration) error {
	scheme := getPluginArgConversionScheme()
	for i := range out.Profiles {
		for j := range out.Profiles[i].PluginConfig {
			args := out.Profiles[i].PluginConfig[j].Args
			if args.Object == nil {
				continue
			}
			if _, isUnknown := args.Object.(*runtime.Unknown); isUnknown {
				continue
			}
			externalArgs, err := scheme.ConvertToVersion(args.Object, SchemeGroupVersion)
			if err != nil {
				return err
			}
			out.Profiles[i].PluginConfig[j].Args.Object = externalArgs
		}
	}
	return nil
}
