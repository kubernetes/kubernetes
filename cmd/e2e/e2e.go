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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/test/e2e"
	"github.com/golang/glog"
	flag "github.com/spf13/pflag"
)

var (
	authConfig = flag.String("auth_config", os.Getenv("HOME")+"/.kubernetes_auth", "Path to the auth info file.")
	certDir    = flag.String("cert_dir", "", "Path to the directory containing the certs. Default is empty, which doesn't use certs.")
	host       = flag.String("host", "", "The host to connect to")
	repoRoot   = flag.String("repo_root", "./", "Root directory of kubernetes repository, for finding test files. Default assumes working directory is repository root")
	provider   = flag.String("provider", "", "The name of the Kubernetes provider")
	orderseed  = flag.Int64("orderseed", 0, "If non-zero, seed of random test shuffle order. (Otherwise random.)")
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
		glog.Error("e2e needs the have the --provider flag set")
		os.Exit(1)
	}
	if *times <= 0 {
		glog.Error("Invalid --times (negative or no testing requested)!")
		os.Exit(1)
	}
	e2e.RunE2ETests(*authConfig, *certDir, *host, *repoRoot, *provider, *orderseed, *times, testList)
}
