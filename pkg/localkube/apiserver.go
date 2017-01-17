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

package localkube

import (
	"strings"

	apiserver "k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"

	"k8s.io/kubernetes/pkg/storage/storagebackend"
)

func (lk LocalkubeServer) NewAPIServer() Server {
	return NewSimpleServer("apiserver", serverInterval, StartAPIServer(lk))
}

func StartAPIServer(lk LocalkubeServer) func() error {
	s := options.NewServerRunOptions()

	s.SecureServing.ServingOptions.BindAddress = lk.APIServerAddress
	s.SecureServing.ServingOptions.BindPort = lk.APIServerPort

	s.InsecureServing.BindAddress = lk.APIServerInsecureAddress
	s.InsecureServing.BindPort = lk.APIServerInsecurePort

	s.Authentication.RequestHeader.ClientCAFile = lk.GetCAPublicKeyCertPath()
	s.SecureServing.ServerCert.CertKey.CertFile = lk.GetPublicKeyCertPath()
	s.SecureServing.ServerCert.CertKey.KeyFile = lk.GetPrivateKeyCertPath()

	s.GenericServerRunOptions.AdmissionControl = "NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,ResourceQuota"
	// use localkube etcd
	s.Etcd.StorageConfig = storagebackend.Config{ServerList: KubeEtcdClientURLs, Type: storagebackend.StorageTypeETCD2}

	// set Service IP range
	s.ServiceClusterIPRange = lk.ServiceClusterIPRange

	// defaults from apiserver command
	s.GenericServerRunOptions.EnableProfiling = true
	s.GenericServerRunOptions.EnableWatchCache = true
	s.GenericServerRunOptions.MinRequestTimeout = 1800

	s.AllowPrivileged = true

	s.GenericServerRunOptions.RuntimeConfig = lk.RuntimeConfig

	lk.SetExtraConfigForComponent("apiserver", &s)

	return func() error {
		return apiserver.Run(s)
	}
}

// notFoundErr returns true if the passed error is an API server object not found error
func notFoundErr(err error) bool {
	if err == nil {
		return false
	}
	return strings.HasSuffix(err.Error(), "not found")
}
