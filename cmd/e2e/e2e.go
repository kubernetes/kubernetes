/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"os"
	goruntime "runtime"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/test/e2e"
	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
)

var (
	context     = &e2e.TestContextType{}
	cloudConfig = &context.CloudConfig

	orderseed = flag.Int64("orderseed", 0, "If non-zero, seed of random test shuffle order. (Otherwise random.)")
	reportDir = flag.String("report-dir", "", "Path to the directory where the JUnit XML reports should be saved. Default is empty, which doesn't generate these reports.")
	times     = flag.Int("times", 1, "Number of times each test is eligible to be run. Individual order is determined by shuffling --times instances of each test using --orderseed (like a multi-deck shoe of cards).")
	testList  util.StringList
)

func init() {
	flag.VarP(&testList, "test", "t", "Test to execute (may be repeated or comma separated list of tests.) Defaults to running all tests.")

	flag.StringVar(&context.KubeConfig, clientcmd.RecommendedConfigPathFlag, "", "Path to kubeconfig containing embeded authinfo.")
	flag.StringVar(&context.KubeContext, clientcmd.FlagContext, "", "kubeconfig context to use/override. If unset, will use value from 'current-context'")
	flag.StringVar(&context.AuthConfig, "auth-config", "", "Path to the auth info file.")
	flag.StringVar(&context.CertDir, "cert-dir", "", "Path to the directory containing the certs. Default is empty, which doesn't use certs.")
	flag.StringVar(&context.Host, "host", "", "The host, or apiserver, to connect to")
	flag.StringVar(&context.RepoRoot, "repo-root", "./", "Root directory of kubernetes repository, for finding test files. Default assumes working directory is repository root")
	flag.StringVar(&context.Provider, "provider", "", "The name of the Kubernetes provider (gce, gke, local, vagrant, etc.)")

	// TODO: Flags per provider?  Rename gce_project/gce_zone?
	flag.StringVar(&cloudConfig.MasterName, "kube-master", "", "Name of the kubernetes master. Only required if provider is gce or gke")
	flag.StringVar(&cloudConfig.ProjectID, "gce-project", "", "The GCE project being used, if applicable")
	flag.StringVar(&cloudConfig.Zone, "gce-zone", "", "GCE zone being used, if applicable")
}

func main() {
	util.InitFlags()
	goruntime.GOMAXPROCS(goruntime.NumCPU())
	if context.Provider == "" {
		glog.Info("The --provider flag is not set.  Treating as a conformance test.  Some tests may not be run.")
		os.Exit(1)
	}
	if *times <= 0 {
		glog.Error("Invalid --times (negative or no testing requested)!")
		os.Exit(1)
	}

	if context.Provider == "aws" {
		awsConfig := "[Global]\n"
		if cloudConfig.Zone == "" {
			glog.Error("--gce-zone must be specified for AWS")
			os.Exit(1)
		}
		awsConfig += fmt.Sprintf("Zone=%s\n", cloudConfig.Zone)

		var err error
		cloudConfig.Provider, err = cloudprovider.GetCloudProvider(context.Provider, strings.NewReader(awsConfig))
		if err != nil {
			glog.Error("Error building AWS provider: ", err)
			os.Exit(1)
		}
	}

	e2e.RunE2ETests(context, *orderseed, *times, *reportDir, testList)
}
