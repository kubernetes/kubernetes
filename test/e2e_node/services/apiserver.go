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

package services

import (
	"fmt"
	"net"

	apiserver "k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
)

const (
	clusterIPRange          = "10.0.0.1/24"
	apiserverClientURL      = "http://localhost:8080"
	apiserverHealthCheckURL = apiserverClientURL + "/healthz"
)

// APIServer is a server which manages apiserver.
type APIServer struct{}

// NewAPIServer creates an apiserver.
func NewAPIServer() *APIServer {
	return &APIServer{}
}

// Start starts the apiserver, returns when apiserver is ready.
func (a *APIServer) Start() error {
	config := options.NewServerRunOptions()
	config.Etcd.StorageConfig.ServerList = []string{getEtcdClientURL()}
	// TODO: Current setup of etcd in e2e-node tests doesn't support etcd v3
	// protocol. We should migrate it to use the same infrastructure as all
	// other tests (pkg/storage/etcd/testing).
	config.Etcd.StorageConfig.Type = "etcd2"
	_, ipnet, err := net.ParseCIDR(clusterIPRange)
	if err != nil {
		return err
	}
	config.ServiceClusterIPRange = *ipnet
	config.AllowPrivileged = true
	errCh := make(chan error)
	go func() {
		defer close(errCh)
		stopCh := make(chan struct{})
		defer close(stopCh)
		err := apiserver.Run(config, stopCh)
		if err != nil {
			errCh <- fmt.Errorf("run apiserver error: %v", err)
		}
	}()

	err = readinessCheck("apiserver", []string{apiserverHealthCheckURL}, errCh)
	if err != nil {
		return err
	}
	return nil
}

// Stop stops the apiserver. Currently, there is no way to stop the apiserver.
// The function is here only for completion.
func (a *APIServer) Stop() error {
	return nil
}

const apiserverName = "apiserver"

func (a *APIServer) Name() string {
	return apiserverName
}

func getAPIServerClientURL() string {
	return apiserverClientURL
}

func getAPIServerHealthCheckURL() string {
	return apiserverHealthCheckURL
}
