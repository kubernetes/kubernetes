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

package v1beta3

import (
	"sort"

	"github.com/pkg/errors"

	conversion "k8s.io/apimachinery/pkg/conversion"
	kubeadm "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func Convert_kubeadm_InitConfiguration_To_v1beta3_InitConfiguration(in *kubeadm.InitConfiguration, out *InitConfiguration, s conversion.Scope) error {
	return autoConvert_kubeadm_InitConfiguration_To_v1beta3_InitConfiguration(in, out, s)
}

func Convert_v1beta3_InitConfiguration_To_kubeadm_InitConfiguration(in *InitConfiguration, out *kubeadm.InitConfiguration, s conversion.Scope) error {
	err := autoConvert_v1beta3_InitConfiguration_To_kubeadm_InitConfiguration(in, out, s)
	if err != nil {
		return err
	}

	// Ignore InitConfiguration.ObjectMeta as it has no meaning and no representation in kubeadm.InitConfiguration

	// Keep the fuzzer test happy by setting out.ClusterConfiguration to defaults
	clusterCfg := &ClusterConfiguration{}
	SetDefaults_ClusterConfiguration(clusterCfg)
	return Convert_v1beta3_ClusterConfiguration_To_kubeadm_ClusterConfiguration(clusterCfg, &out.ClusterConfiguration, s)
}

// Convert_v1beta3_JoinConfiguration_To_kubeadm_JoinConfiguration is required to ignore the local ObjectMeta (missing from the internal type)
func Convert_v1beta3_JoinConfiguration_To_kubeadm_JoinConfiguration(in *JoinConfiguration, out *kubeadm.JoinConfiguration, s conversion.Scope) error {
	return autoConvert_v1beta3_JoinConfiguration_To_kubeadm_JoinConfiguration(in, out, s)
}

// Convert_kubeadm_ClusterConfiguration_To_v1beta3_ClusterConfiguration is required to convert the AddOns from the map in the internal type
func Convert_kubeadm_ClusterConfiguration_To_v1beta3_ClusterConfiguration(in *kubeadm.ClusterConfiguration, out *ClusterConfiguration, s conversion.Scope) error {
	if err := autoConvert_kubeadm_ClusterConfiguration_To_v1beta3_ClusterConfiguration(in, out, s); err != nil {
		return err
	}

	// we want this to be deterministic, so we need to sort the map keys
	addonKeys := []string{}
	for key := range in.AddOns {
		addonKeys = append(addonKeys, key)
	}
	sort.Strings(addonKeys)

	// Needed so we can correctly convert from empty in.AddOns to empty out.AddOns
	if in.AddOns != nil {
		out.AddOns = []AddOn{}
	}

	for _, key := range addonKeys {
		v1beta3AddOn := AddOn{}
		kubeadmAddOn := in.AddOns[key]
		if err := Convert_kubeadm_AddOn_To_v1beta3_AddOn(&kubeadmAddOn, &v1beta3AddOn, s); err != nil {
			return err
		}
		out.AddOns = append(out.AddOns, v1beta3AddOn)
	}

	return nil
}

// Convert_v1beta3_ClusterConfiguration_To_kubeadm_ClusterConfiguration is required because of slice to map conversion
func Convert_v1beta3_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in *ClusterConfiguration, out *kubeadm.ClusterConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1beta3_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in, out, s); err != nil {
		return err
	}

	// Convert the v1beta3 addons slice to an internal map
	if in.AddOns != nil {
		out.AddOns = map[string]kubeadm.AddOn{}
		for _, addon := range in.AddOns {
			convertedAddOn := kubeadm.AddOn{}
			if err := Convert_v1beta3_AddOn_To_kubeadm_AddOn(&addon, &convertedAddOn, s); err != nil {
				return err
			}
			if _, ok := out.AddOns[convertedAddOn.Kind]; ok {
				return errors.Errorf("addon %q is specified twice", convertedAddOn.Kind)
			}
			out.AddOns[convertedAddOn.Kind] = convertedAddOn
		}
	}

	return nil
}
