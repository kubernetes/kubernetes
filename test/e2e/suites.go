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
	"context"
	"fmt"
	"os"
	"path"
	"time"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
)

// AfterSuiteActions are actions that are run on ginkgo's SynchronizedAfterSuite
func AfterSuiteActions(ctx context.Context) {
	// Run only Ginkgo on node 1
	framework.Logf("Running AfterSuite actions on node 1")
	if framework.TestContext.ReportDir != "" {
		framework.CoreDump(framework.TestContext.ReportDir)
	}
	if framework.TestContext.GatherSuiteMetricsAfterTest {
		if err := gatherTestSuiteMetrics(ctx); err != nil {
			framework.Logf("Error gathering metrics: %v", err)
		}
	}
	if framework.TestContext.NodeKiller.NodeKillerStop != nil {
		framework.TestContext.NodeKiller.NodeKillerStop()
	}
}

func gatherTestSuiteMetrics(ctx context.Context) error {
	framework.Logf("Gathering metrics")
	config, err := framework.LoadConfig()
	if err != nil {
		return fmt.Errorf("error loading client config: %w", err)
	}
	c, err := clientset.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("error creating client: %w", err)
	}

	// Grab metrics for apiserver, scheduler, controller-manager, kubelet (for non-kubemark case) and cluster autoscaler (optionally).
	grabber, err := e2emetrics.NewMetricsGrabber(ctx, c, nil, config, !framework.ProviderIs("kubemark"), true, true, true, framework.TestContext.IncludeClusterAutoscalerMetrics, false)
	if err != nil {
		return fmt.Errorf("failed to create MetricsGrabber: %w", err)
	}

	received, err := grabber.Grab(ctx)
	if err != nil {
		return fmt.Errorf("failed to grab metrics: %w", err)
	}

	metricsForE2E := (*e2emetrics.ComponentCollection)(&received)
	metricsJSON := metricsForE2E.PrintJSON()
	if framework.TestContext.ReportDir != "" {
		filePath := path.Join(framework.TestContext.ReportDir, "MetricsForE2ESuite_"+time.Now().Format(time.RFC3339)+".json")
		if err := os.WriteFile(filePath, []byte(metricsJSON), 0644); err != nil {
			return fmt.Errorf("error writing to %q: %w", filePath, err)
		}
	} else {
		framework.Logf("\n\nTest Suite Metrics:\n%s\n", metricsJSON)
	}

	return nil
}
