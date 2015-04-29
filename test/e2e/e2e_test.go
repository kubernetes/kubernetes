/*
Copyright 2014 Google Inc. All rights reserved.

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

package e2e

import (
	"flag"
	"fmt"
	"os"
	"path"
	goruntime "runtime"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
	"github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/gomega"
)

type testResult bool

var (
	cloudConfig = &testContext.CloudConfig

	reportDir = flag.String("report-dir", "", "Path to the directory where the JUnit XML reports should be saved. Default is empty, which doesn't generate these reports.")
)

func init() {
	// Turn on verbose by default to get spec names
	config.DefaultReporterConfig.Verbose = true

	// Turn on EmitSpecProgress to get spec progress (especially on interrupt)
	config.GinkgoConfig.EmitSpecProgress = true

	// Randomize specs as well as suites
	config.GinkgoConfig.RandomizeAllSpecs = true

	flag.StringVar(&testContext.KubeConfig, clientcmd.RecommendedConfigPathFlag, "", "Path to kubeconfig containing embeded authinfo.")
	flag.StringVar(&testContext.KubeContext, clientcmd.FlagContext, "", "kubeconfig context to use/override. If unset, will use value from 'current-context'")
	flag.StringVar(&testContext.AuthConfig, "auth-config", "", "Path to the auth info file.")
	flag.StringVar(&testContext.CertDir, "cert-dir", "", "Path to the directory containing the certs. Default is empty, which doesn't use certs.")
	flag.StringVar(&testContext.Host, "host", "", "The host, or apiserver, to connect to")
	flag.StringVar(&testContext.RepoRoot, "repo-root", "../../", "Root directory of kubernetes repository, for finding test files.")
	flag.StringVar(&testContext.Provider, "provider", "", "The name of the Kubernetes provider (gce, gke, local, vagrant, etc.)")

	// TODO: Flags per provider?  Rename gce-project/gce-zone?
	flag.StringVar(&cloudConfig.MasterName, "kube-master", "", "Name of the kubernetes master. Only required if provider is gce or gke")
	flag.StringVar(&cloudConfig.ProjectID, "gce-project", "", "The GCE project being used, if applicable")
	flag.StringVar(&cloudConfig.Zone, "gce-zone", "", "GCE zone being used, if applicable")
}

func (t *testResult) Fail() { *t = false }

func TestE2E(t *testing.T) {
	defer util.FlushLogs()

	// Disable density test unless it's explicitly requested.
	if config.GinkgoConfig.FocusString == "" && config.GinkgoConfig.SkipString == "" {
		config.GinkgoConfig.SkipString = "Skipped"
	}

	gomega.RegisterFailHandler(ginkgo.Fail)
	// Run tests through the Ginkgo runner with output to console + JUnit for Jenkins
	var r []ginkgo.Reporter
	if *reportDir != "" {
		r = append(r, reporters.NewJUnitReporter(path.Join(*reportDir, fmt.Sprintf("junit_%02d.xml", config.GinkgoConfig.ParallelNode))))
	}
	ginkgo.RunSpecsWithDefaultAndCustomReporters(t, "Kubernetes e2e suite", r)
}

func TestMain(m *testing.M) {
	flag.Parse()
	util.ReallyCrash = true
	util.InitLogs()
	goruntime.GOMAXPROCS(goruntime.NumCPU())

	// TODO: possibly clean up or refactor this functionality.
	if testContext.Provider == "" {
		glog.Info("The --provider flag is not set.  Treating as a conformance test.  Some tests may not be run.")
		os.Exit(1)
	}

	if testContext.Provider == "aws" {
		awsConfig := "[Global]\n"
		if cloudConfig.Zone == "" {
			glog.Error("gce-zone must be specified for AWS")
			os.Exit(1)
		}
		awsConfig += fmt.Sprintf("Zone=%s\n", cloudConfig.Zone)

		var err error
		cloudConfig.Provider, err = cloudprovider.GetCloudProvider(testContext.Provider, strings.NewReader(awsConfig))
		if err != nil {
			glog.Error("Error building AWS provider: ", err)
			os.Exit(1)
		}
	}

	os.Exit(m.Run())
}
