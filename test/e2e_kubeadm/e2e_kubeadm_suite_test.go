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
	"path/filepath"
	"testing"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"github.com/spf13/pflag"

	"github.com/onsi/ginkgo/config"
	morereporters "github.com/onsi/ginkgo/reporters"
	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"
)

func TestMain(m *testing.M) {
	// Copy go flags in TestMain, to ensure go test flags are registered (no longer available in init() as of go1.13)
	e2econfig.CopyFlags(e2econfig.Flags, flag.CommandLine)
	e2econfig.RegisterCommonFlags(flag.CommandLine)
	e2econfig.RegisterClusterFlags(flag.CommandLine)
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()
	e2econfig.AfterReadingAllFlags(&e2econfig.TestContext)
	os.Exit(m.Run())
}

func TestE2E(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	reporters := []ginkgo.Reporter{}
	reportDir := e2econfig.TestContext.ReportDir
	if reportDir != "" {
		// Create the directory if it doesn't already exists
		if err := os.MkdirAll(reportDir, 0755); err != nil {
			t.Fatalf("Failed creating report directory: %v", err)
		} else {
			// Configure a junit reporter to write to the directory
			junitFile := fmt.Sprintf("junit_%s_%02d.xml", e2econfig.TestContext.ReportPrefix, config.GinkgoConfig.ParallelNode)
			junitPath := filepath.Join(reportDir, junitFile)
			reporters = append(reporters, morereporters.NewJUnitReporter(junitPath))
		}
	}
	ginkgo.RunSpecsWithDefaultAndCustomReporters(t, "E2EKubeadm suite", reporters)
}
