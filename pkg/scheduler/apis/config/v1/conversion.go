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

package v1

import (
	"k8s.io/apimachinery/pkg/conversion"
	v1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

func Convert_v1_LegacyExtender_To_config_Extender(in *v1.LegacyExtender, out *config.Extender, s conversion.Scope) error {
	out.URLPrefix = in.URLPrefix
	out.FilterVerb = in.FilterVerb
	out.PreemptVerb = in.PreemptVerb
	out.PrioritizeVerb = in.PrioritizeVerb
	out.Weight = in.Weight
	out.BindVerb = in.BindVerb
	out.EnableHTTPS = in.EnableHTTPS
	out.HTTPTimeout.Duration = in.HTTPTimeout
	out.NodeCacheCapable = in.NodeCacheCapable
	out.Ignorable = in.Ignorable

	if in.TLSConfig != nil {
		out.TLSConfig = &config.ExtenderTLSConfig{}
		if err := Convert_v1_ExtenderTLSConfig_To_config_ExtenderTLSConfig(in.TLSConfig, out.TLSConfig, s); err != nil {
			return err
		}
	} else {
		out.TLSConfig = nil
	}

	if in.ManagedResources != nil {
		out.ManagedResources = make([]config.ExtenderManagedResource, len(in.ManagedResources))
		for i, res := range in.ManagedResources {
			err := Convert_v1_ExtenderManagedResource_To_config_ExtenderManagedResource(&res, &out.ManagedResources[i], s)
			if err != nil {
				return err
			}
		}
	} else {
		out.ManagedResources = nil
	}

	return nil
}

func Convert_config_Extender_To_v1_LegacyExtender(in *config.Extender, out *v1.LegacyExtender, s conversion.Scope) error {
	out.URLPrefix = in.URLPrefix
	out.FilterVerb = in.FilterVerb
	out.PreemptVerb = in.PreemptVerb
	out.PrioritizeVerb = in.PrioritizeVerb
	out.Weight = in.Weight
	out.BindVerb = in.BindVerb
	out.EnableHTTPS = in.EnableHTTPS
	out.HTTPTimeout = in.HTTPTimeout.Duration
	out.NodeCacheCapable = in.NodeCacheCapable
	out.Ignorable = in.Ignorable

	if in.TLSConfig != nil {
		out.TLSConfig = &v1.ExtenderTLSConfig{}
		if err := Convert_config_ExtenderTLSConfig_To_v1_ExtenderTLSConfig(in.TLSConfig, out.TLSConfig, s); err != nil {
			return err
		}
	} else {
		out.TLSConfig = nil
	}

	if in.ManagedResources != nil {
		out.ManagedResources = make([]v1.ExtenderManagedResource, len(in.ManagedResources))
		for i, res := range in.ManagedResources {
			err := Convert_config_ExtenderManagedResource_To_v1_ExtenderManagedResource(&res, &out.ManagedResources[i], s)
			if err != nil {
				return err
			}
		}
	} else {
		out.ManagedResources = nil
	}

	return nil
}
