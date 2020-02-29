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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kube-scheduler/config/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
)

// Convert_v1alpha1_KubeSchedulerConfiguration_To_config_KubeSchedulerConfiguration converts to the internal.
func Convert_v1alpha1_KubeSchedulerConfiguration_To_config_KubeSchedulerConfiguration(in *v1alpha1.KubeSchedulerConfiguration, out *config.KubeSchedulerConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha1_KubeSchedulerConfiguration_To_config_KubeSchedulerConfiguration(in, out, s); err != nil {
		return err
	}
	var profile config.KubeSchedulerProfile
	if err := metav1.Convert_Pointer_string_To_string(&in.SchedulerName, &profile.SchedulerName, s); err != nil {
		return err
	}
	if in.Plugins != nil {
		profile.Plugins = &config.Plugins{}
		if err := Convert_v1alpha1_Plugins_To_config_Plugins(in.Plugins, profile.Plugins, s); err != nil {
			return err
		}
	} else {
		profile.Plugins = nil
	}
	if in.PluginConfig != nil {
		profile.PluginConfig = make([]config.PluginConfig, len(in.PluginConfig))
		for i := range in.PluginConfig {
			if err := Convert_v1alpha1_PluginConfig_To_config_PluginConfig(&in.PluginConfig[i], &profile.PluginConfig[i], s); err != nil {
				return err
			}
		}
	}
	if in.HardPodAffinitySymmetricWeight != nil {
		args := interpodaffinity.Args{HardPodAffinityWeight: in.HardPodAffinitySymmetricWeight}
		plCfg := plugins.NewPluginConfig(interpodaffinity.Name, args)
		profile.PluginConfig = append(profile.PluginConfig, plCfg)
	}
	out.Profiles = []config.KubeSchedulerProfile{profile}
	return nil
}

func Convert_config_KubeSchedulerConfiguration_To_v1alpha1_KubeSchedulerConfiguration(in *config.KubeSchedulerConfiguration, out *v1alpha1.KubeSchedulerConfiguration, s conversion.Scope) error {
	// Conversion from internal to v1alpha1 is not relevant for kube-scheduler.
	return autoConvert_config_KubeSchedulerConfiguration_To_v1alpha1_KubeSchedulerConfiguration(in, out, s)
}

// Convert_v1alpha1_KubeSchedulerLeaderElectionConfiguration_To_config_KubeSchedulerLeaderElectionConfiguration handles deprecated parameters for leader election.
func Convert_v1alpha1_KubeSchedulerLeaderElectionConfiguration_To_config_KubeSchedulerLeaderElectionConfiguration(in *v1alpha1.KubeSchedulerLeaderElectionConfiguration, out *config.KubeSchedulerLeaderElectionConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha1_KubeSchedulerLeaderElectionConfiguration_To_config_KubeSchedulerLeaderElectionConfiguration(in, out, s); err != nil {
		return err
	}
	if len(in.ResourceNamespace) > 0 && len(in.LockObjectNamespace) == 0 {
		out.ResourceNamespace = in.ResourceNamespace
	} else if len(in.ResourceNamespace) == 0 && len(in.LockObjectNamespace) > 0 {
		out.ResourceNamespace = in.LockObjectNamespace
	} else if len(in.ResourceNamespace) > 0 && len(in.LockObjectNamespace) > 0 {
		if in.ResourceNamespace == in.LockObjectNamespace {
			out.ResourceNamespace = in.ResourceNamespace
		} else {
			return fmt.Errorf("ResourceNamespace and LockObjectNamespace are both set and do not match, ResourceNamespace: %s, LockObjectNamespace: %s", in.ResourceNamespace, in.LockObjectNamespace)
		}
	}

	if len(in.ResourceName) > 0 && len(in.LockObjectName) == 0 {
		out.ResourceName = in.ResourceName
	} else if len(in.ResourceName) == 0 && len(in.LockObjectName) > 0 {
		out.ResourceName = in.LockObjectName
	} else if len(in.ResourceName) > 0 && len(in.LockObjectName) > 0 {
		if in.ResourceName == in.LockObjectName {
			out.ResourceName = in.ResourceName
		} else {
			return fmt.Errorf("ResourceName and LockObjectName are both set and do not match, ResourceName: %s, LockObjectName: %s", in.ResourceName, in.LockObjectName)
		}
	}
	return nil
}

// Convert_config_KubeSchedulerLeaderElectionConfiguration_To_v1alpha1_KubeSchedulerLeaderElectionConfiguration handles deprecated parameters for leader election.
func Convert_config_KubeSchedulerLeaderElectionConfiguration_To_v1alpha1_KubeSchedulerLeaderElectionConfiguration(in *config.KubeSchedulerLeaderElectionConfiguration, out *v1alpha1.KubeSchedulerLeaderElectionConfiguration, s conversion.Scope) error {
	if err := autoConvert_config_KubeSchedulerLeaderElectionConfiguration_To_v1alpha1_KubeSchedulerLeaderElectionConfiguration(in, out, s); err != nil {
		return err
	}
	out.ResourceNamespace = in.ResourceNamespace
	out.LockObjectNamespace = in.ResourceNamespace
	out.ResourceName = in.ResourceName
	out.LockObjectName = in.ResourceName
	return nil
}

func Convert_v1alpha1_Plugins_To_config_Plugins(in *v1alpha1.Plugins, out *config.Plugins, s conversion.Scope) error {
	if err := autoConvert_v1alpha1_Plugins_To_config_Plugins(in, out, s); err != nil {
		return err
	}

	if in.PostFilter != nil {
		postFilter, preScore := &in.PostFilter, &out.PreScore
		*preScore = new(config.PluginSet)
		if err := Convert_v1alpha1_PluginSet_To_config_PluginSet(*postFilter, *preScore, s); err != nil {
			return err
		}
	} else {
		out.PreScore = nil
	}

	return nil
}

func Convert_config_Plugins_To_v1alpha1_Plugins(in *config.Plugins, out *v1alpha1.Plugins, s conversion.Scope) error {
	if err := autoConvert_config_Plugins_To_v1alpha1_Plugins(in, out, s); err != nil {
		return err
	}

	if in.PreScore != nil {
		preScore, postFilter := &in.PreScore, &out.PostFilter
		*postFilter = new(v1alpha1.PluginSet)
		if err := Convert_config_PluginSet_To_v1alpha1_PluginSet(*preScore, *postFilter, s); err != nil {
			return err
		}
	} else {
		out.PostFilter = nil
	}

	return nil
}
