/*
Copyright 2015 Google Inc. All rights reserved.

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
	"os"
	goruntime "runtime"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/test/e2e"
	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
)

var (
	kubeConfig = flag.String(clientcmd.RecommendedConfigPathFlag, "", "Path to kubeconfig containing embeded authinfo. Will use cluster/user info from 'current-context'")
	authConfig = flag.String("auth_config", "", "Path to the auth info file.")
	certDir    = flag.String("cert_dir", "", "Path to the directory containing the certs. Default is empty, which doesn't use certs.")
	gceProject = flag.String("gce_project", "", "The GCE project being used, if applicable")
	gceZone    = flag.String("gce_zone", "", "GCE zone being used, if applicable")
	host       = flag.String("host", "", "The host to connect to")
	masterName = flag.String("kube_master", "", "Name of the kubernetes master. Only required if provider is gce or gke")
	provider   = flag.String("provider", "", "The name of the Kubernetes provider")
	orderseed  = flag.Int64("orderseed", 0, "If non-zero, seed of random test shuffle order. (Otherwise random.)")
	repoRoot   = flag.String("repo_root", "./", "Root directory of kubernetes repository, for finding test files. Default assumes working directory is repository root")
	reportDir  = flag.String("report_dir", "", "Path to the directory where the JUnit XML reports should be saved. Default is empty, which doesn't generate these reports.")
	times      = flag.Int("times", 1, "Number of times each test is eligible to be run. Individual order is determined by shuffling --times instances of each test using --orderseed (like a multi-deck shoe of cards).")
	testList   util.StringList
)

func init() {
	flag.VarP(&testList, "test", "t", "Test to execute (may be repeated or comma separated list of tests.) Defaults to running all tests.")
}

func main() {
	util.InitFlags()
	goruntime.GOMAXPROCS(goruntime.NumCPU())
	if *provider == "" {
		glog.Info("The --provider flag is not set.  Treating as a conformance test.  Some tests may not be run.")
		os.Exit(1)
	}
	if *times <= 0 {
		glog.Error("Invalid --times (negative or no testing requested)!")
		os.Exit(1)
	}
	gceConfig := &e2e.GCEConfig{
		ProjectID:  *gceProject,
		Zone:       *gceZone,
		MasterName: *masterName,
	}
	e2e.RunE2ETests(*kubeConfig, *authConfig, *certDir, *host, *repoRoot, *provider, gceConfig, *orderseed, *times, *reportDir, testList)
}
