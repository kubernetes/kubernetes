/*
Copyright 2018 The Kubernetes Authors.

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

package kubeadm

import (
	"flag"
	"fmt"
	"os"
	"testing"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/test/e2e/framework"
	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"

	// reconfigure framework
	_ "k8s.io/kubernetes/test/e2e/framework/debug/init"
	_ "k8s.io/kubernetes/test/e2e/framework/metrics/init"
	_ "k8s.io/kubernetes/test/e2e/framework/node/init"
	_ "k8s.io/kubernetes/test/utils/format"
)

func TestMain(m *testing.M) {
	// Copy go flags in TestMain, to ensure go test flags are registered (no longer available in init() as of go1.13)
	e2econfig.CopyFlags(e2econfig.Flags, flag.CommandLine)
	framework.RegisterCommonFlags(flag.CommandLine)
	framework.RegisterClusterFlags(flag.CommandLine)
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()
	if pflag.CommandLine.NArg() > 0 {
		fmt.Fprintf(os.Stderr, "unknown additional command line arguments: %s", pflag.CommandLine.Args())
		os.Exit(1)
	}
	framework.AfterReadingAllFlags(&framework.TestContext)
	os.Exit(m.Run())
}

func TestE2E(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	reportDir := framework.TestContext.ReportDir
	if reportDir != "" {
		// Create the directory if it doesn't already exists
		// NOTE: junit report can be simply created by executing your tests with the new --junit-report flags instead.
		if err := os.MkdirAll(reportDir, 0755); err != nil {
			t.Fatalf("Failed creating report directory: %v", err)
		}
	}
	suiteConfig, reporterConfig := framework.CreateGinkgoConfig()
	ginkgo.RunSpecs(t, "E2EKubeadm suite", suiteConfig, reporterConfig)
}
