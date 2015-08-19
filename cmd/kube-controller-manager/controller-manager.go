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
	"github.com/golang/glog"
	"github.com/spf13/pflag"
	"k8s.io/kubernetes/cmd/kube-controller-manager/app"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/tools/ha"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/version/verflag"
	"os"
	"runtime"
	"time"
)

func init() {
	healthz.DefaultHealthz()
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	s := app.NewCMServer()
	s.AddFlags(pflag.CommandLine)

	haconfig := ha.Config{}
	haconfig.AddFlags(pflag.CommandLine)

	util.InitFlags()
	util.InitLogs()
	defer util.FlushLogs()

	//Functions to start and stop this daemon.
	startCM := func(leaseUserInfo *ha.LeaseUser) bool {
		leaseUserInfo.Running = true
		if err := s.Run(pflag.CommandLine.Args()); err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			glog.Infof("EXITING NOW ! Killed")
			os.Exit(1)
		}
		return true
	}

	endCM := func(leaseUserInfo *ha.LeaseUser) bool {
		glog.Infof("Hard-exiting the replication controller process now !")

		//Even though we're exiting, we should set this flag before dying just for completeness.
		leaseUserInfo.Running = false
		os.Exit(0)

		//this should never be reached
		return true
	}

	if haconfig.Key == "" {
		haconfig.Key = "ha.controllermanager.lock"
	}
	//This starts a thread that continues running.
	ha.RunHA(s.Kubeconfig, s.Master, startCM, endCM, &haconfig)

	for true {
		glog.Infof("CM lease loop is running...")
		time.Sleep(5 * time.Second)
	}

	verflag.PrintAndExitIfRequested()
}
