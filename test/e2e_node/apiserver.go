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

package e2e_node

import (
	"net"

	apiserver "k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"

	"github.com/golang/glog"
)

const (
	clusterIPRange          = "10.0.0.1/24"
	apiserverClientURL      = "http://localhost:8080"
	apiserverHealthCheckURL = apiserverClientURL + "/healthz"
)

// APIServer is a server which manages apiserver.
type APIServer struct {
	stopCh chan struct{}
}

// NewAPIServer creates an apiserver.
func NewAPIServer() *APIServer {
	return &APIServer{stopCh: make(chan struct{})}
}

// Start starts the apiserver, returns when apiserver is ready.
func (a *APIServer) Start() error {
	var err error
	config := options.NewAPIServer()
	config.StorageConfig.ServerList = []string{getEtcdClientURL()}
	_, ipnet, err := net.ParseCIDR(clusterIPRange)
	if err != nil {
		return err
	}
	config.ServiceClusterIPRange = *ipnet
	config.AllowPrivileged = true
	go a.run(config)
	err = readinessCheck([]string{apiserverHealthCheckURL})
	if err != nil {
		return err
	}
	return nil
}

// run starts the apiserver and wait until stop channel is closed.
func (a *APIServer) run(config *options.APIServer) {
	go func() {
		err := apiserver.Run(config)
		if err != nil {
			glog.Fatalf("run apiserver error: %v", err)
		}
	}()
	<-a.stopCh
}

// Stop closes the stop channel to stop the apiserver.
func (a *APIServer) Stop() error {
	close(a.stopCh)
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
