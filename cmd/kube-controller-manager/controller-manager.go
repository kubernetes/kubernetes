/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// The controller manager is responsible for monitoring replication
// controllers, and creating corresponding pods to achieve the desired
// state.  It uses the API to listen for new controllers and to create/delete
// pods.
package main

import (
	"fmt"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/client/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/clientcmd/api"
	"os"
	"runtime"
	"time"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/version/verflag"
)

func init() {
	healthz.DefaultHealthz()
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	s := app.NewCMServer()
	s.AddFlags(pflag.CommandLine)

	//We need a kubeconfig in order to use the locking API, so we create it here.
	kubeconfig, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		&clientcmd.ClientConfigLoadingRules{ExplicitPath: s.Kubeconfig},
		&clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: s.Master}}).ClientConfig()
	if err != nil {
		glog.Infof("Exiting, couldn't create kube configuration with parameters cfg=%v and master=%v ", s.Kubeconfig, s.Master)
		os.Exit(1)
	}
	kubeClient, err := client.New(kubeconfig)

	util.InitFlags()
	util.InitLogs()
	defer util.FlushLogs()

	leaseUserInfo := controller.LeaseUser{
		Running:      false,
		LeasesGained: 0,
		LeasesLost:   0,
	}

	//Functions to start and stop this daemon.
	startRC := func() bool {
		leaseUserInfo.Running = true
		if err := s.Run(pflag.CommandLine.Args()); err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			glog.Infof("EXITING NOW ! Killed")
			os.Exit(1)
		}
		return true
	}

	endRC := func() bool {
		glog.Infof("Hard-exiting the replication controller process now !")

		//Even though we're exiting, we should set this flag before dying just for completeness.
		leaseUserInfo.Running = false
		os.Exit(0)

		//this should never be reached
		return true
	}

	//Acquire a lock before starting.
	//TODO some of these will change now that implementing robs lock.
	//we can delete some params...
	mcfg := controller.Config{
		Key:           "cm-LEASE",
		LeaseUserInfo: &leaseUserInfo,
		LeaseGained:   startRC,
		LeaseLost:     endRC,
		Cli:           kubeClient}

	//This starts a thread that continues running.
	controller.RunLease(&mcfg)

	for true {
		glog.Infof("CM is running, active = %v", mcfg.LeaseUserInfo.Running)
		time.Sleep(5 * time.Second)
	}

	verflag.PrintAndExitIfRequested()
}
