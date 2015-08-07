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

var RUNNING = false

func isRunning() bool {
	return RUNNING
}

func endRC() bool {
	//glog.Infof("Hard-exiting the replication controller process now !")

	//Even though we're exiting, we should set this flag before dying just for completeness.
	RUNNING = false
	os.Exit(0)

	//this should never be reached
	return false
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	s := app.NewCMServer()
	s.AddFlags(pflag.CommandLine)

	util.InitFlags()
	util.InitLogs()
	defer util.FlushLogs()

	//HA Function, this function will be called by the lease manager itself.
	startRC := func() bool {
		RUNNING = true
		if err := s.Run(pflag.CommandLine.Args()); err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			glog.Infof("EXITING NOW ! Killed")
			os.Exit(1)
		}
		return true
	}

	//Acquire a lock before starting.
	mcfg := controller.Config{
		EtcdServers: "http://localhost:4001",
		Key:         "rcLEASE",
		Running:     isRunning,
		Lease:       startRC,
		Unlease:     endRC}

	//This starts a thread that continues running.
	controller.RunLease(&mcfg)

	for true {
		glog.Infof("RC is running, active = %v", RUNNING)
		time.Sleep(5 * time.Second)
	}

	verflag.PrintAndExitIfRequested()
}
