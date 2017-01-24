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

package cmd

import (
	"fmt"
	"os"
	"os/signal"

	"github.com/coreos/pkg/capnslog"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/localkube"
	"k8s.io/kubernetes/pkg/version"
)

// The main instance of the current localkube server that is started
var Server *localkube.LocalkubeServer

func StartLocalkube() {

	if Server.ShowVersion {
		fmt.Println("localkube version:", version.Get().String())
		os.Exit(0)
	}

	// Get the etcd logger for the api repo
	apiRepoLogger := capnslog.MustRepoLogger("github.com/coreos/etcd/etcdserver/api")
	// Set the logging level to NOTICE as there is an INFO lvl log statement that runs every few seconds -> log spam
	apiRepoLogger.SetRepoLogLevel(capnslog.NOTICE)

	// TODO: Require root

	SetupServer(Server)
	Server.StartAll()

	defer Server.StopAll()

	interruptChan := make(chan os.Signal, 1)
	signal.Notify(interruptChan, os.Interrupt)

	<-interruptChan
	fmt.Println("Shutting down...")
}

func SetupServer(s *localkube.LocalkubeServer) {
	if s.ShouldGenerateCerts {
		if err := s.GenerateCerts(); err != nil {
			fmt.Println("Failed to create certificates!")
			panic(err)
		}
	}

	// Setup capabilities. This can only be done once per binary.
	allSources, _ := types.GetValidatedSources([]string{types.AllSource})
	c := capabilities.Capabilities{
		AllowPrivileged: true,
		PrivilegedSources: capabilities.PrivilegedSources{
			HostNetworkSources: allSources,
			HostIPCSources:     allSources,
			HostPIDSources:     allSources,
		},
	}
	capabilities.Initialize(c)

	// setup etcd
	etcd, err := s.NewEtcd(localkube.KubeEtcdClientURLs, localkube.KubeEtcdPeerURLs, "kubeetcd", s.GetEtcdDataDirectory())
	if err != nil {
		panic(err)
	}
	s.AddServer(etcd)

	// setup apiserver
	apiserver := s.NewAPIServer()
	s.AddServer(apiserver)

	// setup controller-manager
	controllerManager := s.NewControllerManagerServer()
	s.AddServer(controllerManager)

	// setup scheduler
	scheduler := s.NewSchedulerServer()
	s.AddServer(scheduler)

	// setup kubelet
	kubelet := s.NewKubeletServer()
	s.AddServer(kubelet)

	// setup proxy
	proxy := s.NewProxyServer()
	s.AddServer(proxy)
}
