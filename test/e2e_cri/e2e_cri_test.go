/*
Copyright 2016 The Kubernetes Authors.

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

// To run tests in this suite
// NOTE: This test suite requires password-less sudo capabilities to run the kubelet and kube-apiserver.
package e2e_cri

import (
	"fmt"
	"math/rand"
	"os"
	"path"
	"testing"
	"time"

	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	more_reporters "github.com/onsi/ginkgo/reporters"
	. "github.com/onsi/gomega"
)

func init() {
	framework.RegisterCommonFlags()
	framework.RegisterCRIFlags()
}

// TestE2eCRI checks configuration parameters (specified through flags) and then runs
// E2E tests using the Ginkgo runner.
// If a "report directory" is specified, one or more JUnit test reports will be
// generated in this directory, and cluster logs will also be saved.
// This function is called on each Ginkgo node in parallel mode.
func TestE2eCRI(t *testing.T) {
	rand.Seed(time.Now().UTC().UnixNano())
	RegisterFailHandler(Fail)
	reporters := []Reporter{}
	reportDir := framework.TestContext.ReportDir
	if reportDir != "" {
		// Create the directory if it doesn't already exists
		if err := os.MkdirAll(reportDir, 0755); err != nil {
			glog.Errorf("Failed creating report directory: %v", err)
		} else {
			// Configure a junit reporter to write to the directory
			junitFile := fmt.Sprintf("junit_%s%02d.xml", framework.TestContext.ReportPrefix, config.GinkgoConfig.ParallelNode)
			junitPath := path.Join(reportDir, junitFile)
			reporters = append(reporters, more_reporters.NewJUnitReporter(junitPath))
		}
	}
	RunSpecsWithDefaultAndCustomReporters(t, "E2eCRI Suite", reporters)
}
