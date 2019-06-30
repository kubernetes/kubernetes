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
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	configv1alpha1 "k8s.io/component-base/config/v1alpha1"
	v1alpha1 "k8s.io/kube-proxy/config/v1alpha1"
	"k8s.io/kubernetes/pkg/proxy/apis/config"
	"unsafe"
)

func Convert_v1alpha1_KubeProxyConfiguration_To_config_KubeProxyConfiguration(in *v1alpha1.KubeProxyConfiguration, out *config.KubeProxyConfiguration, s conversion.Scope) error {
	out.FeatureGates = *(*map[string]bool)(unsafe.Pointer(&in.FeatureGates))
	if in.BindAddress == "" {
		out.BindAddress = make([]string, 0)
	} else {
		out.BindAddress = []string{in.BindAddress}
	}
	out.HealthzBindAddress = in.HealthzBindAddress
	out.MetricsBindAddress = in.MetricsBindAddress
	out.EnableProfiling = in.EnableProfiling
	if in.ClusterCIDR == "" {
		out.ClusterCIDR = make([]string, 0)
	} else {
		out.ClusterCIDR = []string{in.ClusterCIDR}
	}
	out.HostnameOverride = in.HostnameOverride
	if err := configv1alpha1.Convert_v1alpha1_ClientConnectionConfiguration_To_config_ClientConnectionConfiguration(&in.ClientConnection, &out.ClientConnection, s); err != nil {
		return err
	}
	if err := Convert_v1alpha1_KubeProxyIPTablesConfiguration_To_config_KubeProxyIPTablesConfiguration(&in.IPTables, &out.IPTables, s); err != nil {
		return err
	}
	if err := Convert_v1alpha1_KubeProxyIPVSConfiguration_To_config_KubeProxyIPVSConfiguration(&in.IPVS, &out.IPVS, s); err != nil {
		return err
	}
	out.OOMScoreAdj = (*int32)(unsafe.Pointer(in.OOMScoreAdj))
	out.Mode = config.ProxyMode(in.Mode)
	out.PortRange = in.PortRange
	out.UDPIdleTimeout = in.UDPIdleTimeout
	if err := Convert_v1alpha1_KubeProxyConntrackConfiguration_To_config_KubeProxyConntrackConfiguration(&in.Conntrack, &out.Conntrack, s); err != nil {
		return err
	}
	out.ConfigSyncPeriod = in.ConfigSyncPeriod
	out.NodePortAddresses = *(*[]string)(unsafe.Pointer(&in.NodePortAddresses))
	if err := Convert_v1alpha1_KubeProxyWinkernelConfiguration_To_config_KubeProxyWinkernelConfiguration(&in.Winkernel, &out.Winkernel, s); err != nil {
		return err
	}
	return nil
}

func Convert_config_KubeProxyConfiguration_To_v1alpha1_KubeProxyConfiguration(in *config.KubeProxyConfiguration, out *v1alpha1.KubeProxyConfiguration, s conversion.Scope) error {
	out.FeatureGates = *(*map[string]bool)(unsafe.Pointer(&in.FeatureGates))
	if err := runtime.Convert_Slice_string_To_string(&in.BindAddress, &out.BindAddress, s); err != nil {
		return err
	}

	if err := runtime.Convert_Slice_string_To_string(&in.BindAddress, &out.BindAddress, s); err != nil {
		return err
	}
	if err := runtime.Convert_Slice_string_To_string(&in.ClusterCIDR, &out.ClusterCIDR, s); err != nil {
		return err
	}

	out.HealthzBindAddress = in.HealthzBindAddress
	out.MetricsBindAddress = in.MetricsBindAddress
	out.EnableProfiling = in.EnableProfiling
	out.HostnameOverride = in.HostnameOverride

	if err := configv1alpha1.Convert_config_ClientConnectionConfiguration_To_v1alpha1_ClientConnectionConfiguration(&in.ClientConnection, &out.ClientConnection, s); err != nil {
		return err
	}
	if err := Convert_config_KubeProxyIPTablesConfiguration_To_v1alpha1_KubeProxyIPTablesConfiguration(&in.IPTables, &out.IPTables, s); err != nil {
		return err
	}
	if err := Convert_config_KubeProxyIPVSConfiguration_To_v1alpha1_KubeProxyIPVSConfiguration(&in.IPVS, &out.IPVS, s); err != nil {
		return err
	}
	out.OOMScoreAdj = (*int32)(unsafe.Pointer(in.OOMScoreAdj))
	out.Mode = v1alpha1.ProxyMode(in.Mode)
	out.PortRange = in.PortRange
	out.UDPIdleTimeout = in.UDPIdleTimeout
	if err := Convert_config_KubeProxyConntrackConfiguration_To_v1alpha1_KubeProxyConntrackConfiguration(&in.Conntrack, &out.Conntrack, s); err != nil {
		return err
	}
	out.ConfigSyncPeriod = in.ConfigSyncPeriod
	out.NodePortAddresses = *(*[]string)(unsafe.Pointer(&in.NodePortAddresses))
	if err := Convert_config_KubeProxyWinkernelConfiguration_To_v1alpha1_KubeProxyWinkernelConfiguration(&in.Winkernel, &out.Winkernel, s); err != nil {
		return err
	}
	return nil
}
