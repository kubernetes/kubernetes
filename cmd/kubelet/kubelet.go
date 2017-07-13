/*
Copyright 2014 The Kubernetes Authors.

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

// The kubelet binary is responsible for maintaining a set of containers on a particular host VM.
// It syncs data from both configuration file(s) as well as from a quorum of etcd servers.
// It then queries Docker to see what is currently running.  It synchronizes the configuration data,
// with the running set of containers by starting or stopping Docker containers.
package main

import (
	"fmt"
	"os"

	"github.com/spf13/pflag"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/flag"
	"k8s.io/apiserver/pkg/util/logs"
	"k8s.io/kubernetes/cmd/kubelet/app"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	_ "k8s.io/kubernetes/pkg/client/metrics/prometheus" // for client metric registration
	"k8s.io/kubernetes/pkg/features"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig"
	_ "k8s.io/kubernetes/pkg/version/prometheus" // for version metric registration
	"k8s.io/kubernetes/pkg/version/verflag"
)

func die(err error) {
	fmt.Fprintf(os.Stderr, "error: %v\n", err)
	os.Exit(1)
}

func main() {
	// construct KubeletFlags object and register command line flags mapping
	kubeletFlags := options.NewKubeletFlags()
	kubeletFlags.AddFlags(pflag.CommandLine)

	// construct KubeletConfiguration object and register command line flags mapping
	defaultConfig, err := options.NewKubeletConfiguration()
	if err != nil {
		die(err)
	}
	options.AddKubeletConfigFlags(pflag.CommandLine, defaultConfig)

	// parse the command line flags into the respective objects
	flag.InitFlags()

	// initialize logging and defer flush
	logs.InitLogs()
	defer logs.FlushLogs()

	// short-circuit on verflag
	verflag.PrintAndExitIfRequested()

	// TODO(mtaufen): won't need this this once dynamic config is GA
	// set feature gates so we can check if dynamic config is enabled
	if err := utilfeature.DefaultFeatureGate.Set(defaultConfig.FeatureGates); err != nil {
		die(err)
	}
	// validate the initial KubeletFlags, to make sure the dynamic-config-related flags aren't used unless the feature gate is on
	if err := options.ValidateKubeletFlags(kubeletFlags); err != nil {
		die(err)
	}
	// if dynamic kubelet config is enabled, bootstrap the kubelet config controller
	var kubeletConfig *kubeletconfiginternal.KubeletConfiguration
	var kubeletConfigController *kubeletconfig.Controller
	if utilfeature.DefaultFeatureGate.Enabled(features.DynamicKubeletConfig) {
		var err error
		kubeletConfig, kubeletConfigController, err = app.BootstrapKubeletConfigController(kubeletFlags, defaultConfig)
		if err != nil {
			die(err)
		}
	} else if kubeletConfig == nil {
		kubeletConfig = defaultConfig
	}

	// construct a KubeletServer from kubeletFlags and kubeletConfig
	kubeletServer := &options.KubeletServer{KubeletFlags: *kubeletFlags, KubeletConfiguration: *kubeletConfig}

	// use kubeletServer to construct the default KubeletDeps
	kubeletDeps, err := app.UnsecuredDependencies(kubeletServer)

	// add the kubelet config controller to kubeletDeps
	kubeletDeps.KubeletConfigController = kubeletConfigController

	// start the experimental docker shim, if enabled
	if kubeletFlags.ExperimentalDockershim {
		if err := app.RunDockershim(kubeletConfig, &kubeletFlags.ContainerRuntimeOptions); err != nil {
			die(err)
		}
	}

	// run the kubelet
	if err := app.Run(kubeletServer, kubeletDeps); err != nil {
		die(err)
	}
}
