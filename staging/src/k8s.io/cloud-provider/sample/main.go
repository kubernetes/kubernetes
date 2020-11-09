/*
Copyright 2020 The Kubernetes Authors.

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

// This file should be written by each cloud provider.
// This is an minimal example. For more details, please refer to k8s.io/kubernetes/cmd/cloud-controller-manager/controller-manager.go

package sample

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/spf13/pflag"
	"k8s.io/cloud-provider"
	"k8s.io/cloud-provider/app"
	"k8s.io/cloud-provider/options"
	"k8s.io/component-base/cli/flag"
	"k8s.io/component-base/logs"
	_ "k8s.io/component-base/metrics/prometheus/clientgo" // load all the prometheus client-go plugins
	_ "k8s.io/component-base/metrics/prometheus/version"  // for version metric registration
	"k8s.io/klog/v2"
)

const (
	// cloudProviderName shows an sample of using hard coded parameter
	cloudProviderName = "SampleCloudProviderName"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// cloudProviderConfigFile shows an sample of parse config file from flag option
	var flagset *pflag.FlagSet = pflag.NewFlagSet("flagSet", pflag.ContinueOnError)
	var cloudProviderConfigFile *string = flagset.String("cloud-provider-configfile", "", "This is the sample input for cloud provider config file")
	pflag.CommandLine.ParseErrorsWhitelist.UnknownFlags = true
	_ = pflag.CommandLine.Parse(os.Args[1:])

	// this is an example of allow-listing specific controller loops
	controllerList := []string{"cloud-node", "cloud-node-lifecycle", "service", "route"}

	s, err := options.NewCloudControllerManagerOptions()
	if err != nil {
		klog.Fatalf("unable to initialize command options: %v", err)
	}
	c, err := s.Config(controllerList, app.ControllersDisabledByDefault.List())
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	cloud, err := cloudprovider.InitCloudProvider(cloudProviderName, *cloudProviderConfigFile)
	if err != nil {
		klog.Fatalf("Cloud provider could not be initialized: %v", err)
	}
	if cloud == nil {
		klog.Fatalf("cloud provider is nil")
	}

	if !cloud.HasClusterID() {
		if c.ComponentConfig.KubeCloudShared.AllowUntaggedCloud {
			klog.Warning("detected a cluster without a ClusterID.  A ClusterID will be required in the future.  Please tag your cluster to avoid any future issues")
		} else {
			klog.Fatalf("no ClusterID found.  A ClusterID is required for the cloud provider to function properly.  This check can be bypassed by setting the allow-untagged-cloud option")
		}
	}

	// Initialize the cloud provider with a reference to the clientBuilder
	cloud.Initialize(c.ClientBuilder, make(chan struct{}))
	// Set the informer on the user cloud object
	if informerUserCloud, ok := cloud.(cloudprovider.InformerUser); ok {
		informerUserCloud.SetInformers(c.SharedInformers)
	}

	controllerInitializers := app.DefaultControllerInitializers(c.Complete(), cloud)
	command := app.NewCloudControllerManagerCommand(s, c, controllerInitializers)

	// TODO: once we switch everything over to Cobra commands, we can go back to calling
	// utilflag.InitFlags() (by removing its pflag.Parse() call). For now, we have to set the
	// normalize func and add the go flag set by hand.
	// Here is an sample
	pflag.CommandLine.SetNormalizeFunc(flag.WordSepNormalizeFunc)
	// utilflag.InitFlags()
	logs.InitLogs()
	defer logs.FlushLogs()

	// the flags could be set before execute
	command.Flags().VisitAll(func(flag *pflag.Flag) {
		if flag.Name == "cloud-provider" {
			flag.Value.Set("SampleCloudProviderFlagValue")
			return
		}
	})
	if err := command.Execute(); err != nil {
		os.Exit(1)
	}
}
