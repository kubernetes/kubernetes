/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"os"
	"path"
	"time"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	"k8s.io/kubernetes/test/e2e/framework/utils"
)

// CleanupSuite is the boilerplate that can be used after tests on ginkgo were run, on the SynchronizedAfterSuite step.
// Similar to SynchronizedBeforeSuite, we want to run some operations only once (such as collecting cluster logs).
// Here, the order of functions is reversed; first, the function which runs everywhere,
// and then the function that only runs on the first Ginkgo node.
func CleanupSuite() {
	// Run on all Ginkgo nodes
	utils.Logf("Running AfterSuite actions on all nodes")
	framework.RunCleanupActions()
}

// AfterSuiteActions are actions that are run on ginkgo's SynchronizedAfterSuite
func AfterSuiteActions() {
	// Run only Ginkgo on node 1
	utils.Logf("Running AfterSuite actions on node 1")
	if e2econfig.TestContext.ReportDir != "" {
		utils.CoreDump(e2econfig.TestContext.ReportDir)
	}
	if e2econfig.TestContext.GatherSuiteMetricsAfterTest {
		if err := gatherTestSuiteMetrics(); err != nil {
			utils.Logf("Error gathering metrics: %v", err)
		}
	}
	if e2econfig.TestContext.NodeKiller.Enabled {
		close(e2econfig.TestContext.NodeKiller.NodeKillerStopCh)
	}
}

func gatherTestSuiteMetrics() error {
	utils.Logf("Gathering metrics")
	config, err := utils.LoadConfig()
	if err != nil {
		return fmt.Errorf("error loading client config: %v", err)
	}
	c, err := clientset.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("error creating client: %v", err)
	}

	// Grab metrics for apiserver, scheduler, controller-manager, kubelet (for non-kubemark case) and cluster autoscaler (optionally).
	grabber, err := e2emetrics.NewMetricsGrabber(c, nil, config, !utils.ProviderIs("kubemark"), true, true, true, e2econfig.TestContext.IncludeClusterAutoscalerMetrics, false)
	if err != nil {
		return fmt.Errorf("failed to create MetricsGrabber: %v", err)
	}

	received, err := grabber.Grab()
	if err != nil {
		return fmt.Errorf("failed to grab metrics: %v", err)
	}

	metricsForE2E := (*e2emetrics.ComponentCollection)(&received)
	metricsJSON := metricsForE2E.PrintJSON()
	if e2econfig.TestContext.ReportDir != "" {
		filePath := path.Join(e2econfig.TestContext.ReportDir, "MetricsForE2ESuite_"+time.Now().Format(time.RFC3339)+".json")
		if err := os.WriteFile(filePath, []byte(metricsJSON), 0644); err != nil {
			return fmt.Errorf("error writing to %q: %v", filePath, err)
		}
	} else {
		utils.Logf("\n\nTest Suite Metrics:\n%s\n", metricsJSON)
	}

	return nil
}
