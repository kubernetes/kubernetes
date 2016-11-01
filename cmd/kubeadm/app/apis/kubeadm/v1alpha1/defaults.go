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
	"k8s.io/kubernetes/pkg/runtime"
)

const (
	DefaultServiceDNSDomain  = "cluster.local"
	DefaultServicesSubnet    = "10.96.0.0/12"
	DefaultKubernetesVersion = "v1.4.4"
	DefaultAPIBindPort       = 6443
	DefaultDiscoveryBindPort = 9898
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	RegisterDefaults(scheme)
	return scheme.AddDefaultingFuncs(
		SetDefaults_MasterConfiguration,
		SetDefaults_NodeConfiguration,
	)
}

func SetDefaults_MasterConfiguration(obj *MasterConfiguration) {
	if obj.KubernetesVersion == "" {
		obj.KubernetesVersion = DefaultKubernetesVersion
	}

	if obj.API.BindPort == 0 {
		obj.API.BindPort = DefaultAPIBindPort
	}

	if obj.Discovery.BindPort == 0 {
		obj.Discovery.BindPort = DefaultDiscoveryBindPort
	}

	if obj.Networking.ServiceSubnet == "" {
		obj.Networking.ServiceSubnet = DefaultServicesSubnet
	}

	if obj.Networking.DNSDomain == "" {
		obj.Networking.DNSDomain = DefaultServiceDNSDomain
	}
}

func SetDefaults_NodeConfiguration(obj *NodeConfiguration) {
	if obj.APIPort == 0 {
		obj.APIPort = DefaultAPIBindPort
	}

	if obj.DiscoveryPort == 0 {
		obj.DiscoveryPort = DefaultDiscoveryBindPort
	}
}
