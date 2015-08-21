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

package main

import (
	"fmt"
	"github.com/golang/glog"
	"github.com/spf13/pflag"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/tools/ha"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/version/verflag"
	"k8s.io/kubernetes/plugin/cmd/kube-scheduler/app"
	"os"
	"runtime"
	"time"
)

func init() {
	healthz.DefaultHealthz()
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	s := app.NewSchedulerServer()
	s.AddFlags(pflag.CommandLine)

	haconfig := ha.Config{}
	haconfig.AddFlags(pflag.CommandLine)

	util.InitFlags()
	util.InitLogs()
	defer util.FlushLogs()

	verflag.PrintAndExitIfRequested()

	startSched := func(leaseUserInfo *ha.LeaseUser) bool {
		leaseUserInfo.Running = true
		glog.Infof("Starting kube scheduler. %v", leaseUserInfo)
		if err := s.Run(pflag.CommandLine.Args()); err != nil {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			glog.Infof("EXITING NOW ! Killed")
			os.Exit(1)
		}
		return true
	}

	endSched := func(leaseUserInfo *ha.LeaseUser) bool {
		glog.Infof("Hard-exiting the scheduler process now!")
		leaseUserInfo.Running = true
		os.Exit(0)
		return true
	}

	if haconfig.Key == "" {
		haconfig.Key = "ha.scheduler.lock"
	}

	ha.RunHA(s.Kubeconfig, s.Master, startSched, endSched, &haconfig)

	for true {
		glog.Infof("Scheduler lease loop is running...")
		time.Sleep(5 * time.Second)
	}

}
