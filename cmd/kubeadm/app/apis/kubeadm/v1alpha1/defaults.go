/*
Copyright 2016 The Kubernetes Authors.

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
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/pkg/runtime"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	RegisterDefaults(scheme)
	return scheme.AddDefaultingFuncs(
		SetDefaults_ClusterInfo,
		SetDefaults_MasterConfiguration,
		SetDefaults_NodeConfiguration,
	)
}

func SetDefaults_ClusterInfo(obj *ClusterInfo) {
	// defaults for ClusterInfo not needed yet
}

func SetDefaults_MasterConfiguration(obj *MasterConfiguration) {
	if obj.KubernetesVersion == "" {
		obj.KubernetesVersion = kubeadmapi.DefaultKubernetesVersion
	}

	if obj.API.BindPort == 0 {
		obj.API.BindPort = kubeadmapi.DefaultAPIBindPort
	}

	if obj.Discovery.BindPort == 0 {
		obj.Discovery.BindPort = kubeadmapi.DefaultDiscoveryBindPort
	}

	if obj.Networking.ServiceSubnet == "" {
		obj.Networking.ServiceSubnet = kubeadmapi.DefaultServicesSubnet
	}

	if obj.Networking.DNSDomain == "" {
		obj.Networking.DNSDomain = kubeadmapi.DefaultServiceDNSDomain
	}
}

func SetDefaults_NodeConfiguration(obj *NodeConfiguration) {
	if obj.APIPort == 0 {
		obj.APIPort = kubeadmapi.DefaultAPIBindPort
	}

	if obj.DiscoveryPort == 0 {
		obj.DiscoveryPort = kubeadmapi.DefaultDiscoveryBindPort
	}
}
