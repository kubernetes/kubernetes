/*
Copyright 2017 The Kubernetes Authors.

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
	"net/url"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

const (
	DefaultServiceDNSDomain   = "cluster.local"
	DefaultServicesSubnet     = "10.96.0.0/12"
	DefaultKubernetesVersion  = "stable-1.7"
	DefaultAPIBindPort        = 6443
	DefaultAuthorizationModes = "Node,RBAC"
	DefaultCACertPath         = "/etc/kubernetes/pki/ca.crt"
	DefaultCertificatesDir    = "/etc/kubernetes/pki"
	DefaultEtcdDataDir        = "/var/lib/etcd"
	DefaultImageRepository    = "gcr.io/google_containers"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_MasterConfiguration(obj *MasterConfiguration) {
	SetDefaults_APIServer(&obj.APIServer)
	SetDefaults_ControllerManager(&obj.ControllerManager)
	SetDefaults_Scheduler(&obj.Scheduler)
	SetDefaults_Networking(&obj.Networking)
	SetDefaults_BootstrapToken(&obj.BootstrapToken)
	SetDefaults_Etcd(&obj.Etcd)
	SetDefaults_MasterPaths(&obj.MasterPaths)

	if obj.KubernetesVersion == "" {
		obj.KubernetesVersion = DefaultKubernetesVersion
	}

	if obj.ImageRepository == "" {
		obj.ImageRepository = DefaultImageRepository
	}
}

func SetDefaults_APIServer(obj *APIServer) {
	if obj.BindPort == 0 {
		obj.BindPort = DefaultAPIBindPort
	}

	if len(obj.AuthorizationModes) == 0 {
		obj.AuthorizationModes = strings.Split(DefaultAuthorizationModes, ",")
	}

	if obj.ExtraArgs == nil {
		obj.ExtraArgs = map[string]string{}
	}
}

func SetDefaults_ControllerManager(obj *ControllerManager) {
	if obj.ExtraArgs == nil {
		obj.ExtraArgs = map[string]string{}
	}
}

func SetDefaults_Scheduler(obj *Scheduler) {
	if obj.ExtraArgs == nil {
		obj.ExtraArgs = map[string]string{}
	}
}

func SetDefaults_Networking(obj *Networking) {
	if obj.ServiceSubnet == "" {
		obj.ServiceSubnet = DefaultServicesSubnet
	}

	if obj.DNSDomain == "" {
		obj.DNSDomain = DefaultServiceDNSDomain
	}
}

func SetDefaults_BootstrapToken(obj *BootstrapToken) {
	if obj.Node == nil {
		obj.Node = NodeBootstrapToken{
			Enabled: true,
			Token: "", // will be auto-generated in a later stage
			TokenTTL: constants.DefaultTokenDuration,
		}
	}
}

func SetDefaults_Etcd(obj *Etcd) {
	if obj.Local == nil && obj.External == nil {
		obj.Local = LocalEtcd{}
	}

	if obj.Local != nil {
		if obj.Local.DataDir == "" {
			obj.Local.DataDir = DefaultEtcdDataDir
		}

		if obj.Local.ExtraArgs == nil {
			obj.Local.ExtraArgs = map[string]string{}
		}
	}
}

func SetDefaults_MasterPaths(obj *MasterPaths) {
	if obj.CertificatesDir == "" {
		obj.CertificatesDir = DefaultCertificatesDir
	}
}

func SetDefaults_NodeConfiguration(obj *NodeConfiguration) {
	if obj.CACertPath == "" {
		obj.CACertPath = DefaultCACertPath
	}
	if len(obj.TLSBootstrapToken) == 0 {
		obj.TLSBootstrapToken = obj.Token
	}
	if len(obj.DiscoveryToken) == 0 && len(obj.DiscoveryFile) == 0 {
		obj.DiscoveryToken = obj.Token
	}
	// Make sure file URLs become paths
	if len(obj.DiscoveryFile) != 0 {
		u, err := url.Parse(obj.DiscoveryFile)
		if err == nil && u.Scheme == "file" {
			obj.DiscoveryFile = u.Path
		}
	}
}
