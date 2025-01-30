/*
Copyright 2015 The Kubernetes Authors.

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

package metrics

import (
	"context"

	"github.com/onsi/ginkgo/v2"

	"k8s.io/kubernetes/test/e2e/framework"
)

func GrabBeforeEach(ctx context.Context, f *framework.Framework) (result *Collection) {
	gatherMetricsAfterTest := framework.TestContext.GatherMetricsAfterTest == "true" || framework.TestContext.GatherMetricsAfterTest == "master"
	if !gatherMetricsAfterTest || !framework.TestContext.IncludeClusterAutoscalerMetrics {
		return nil
	}

	ginkgo.By("Gathering metrics before test", func() {
		grabber, err := NewMetricsGrabber(ctx, f.ClientSet, f.KubemarkExternalClusterClientSet, f.ClientConfig(), !framework.ProviderIs("kubemark"), false, false, false, framework.TestContext.IncludeClusterAutoscalerMetrics, false)
		if err != nil {
			framework.Logf("Failed to create MetricsGrabber (skipping ClusterAutoscaler metrics gathering before test): %v", err)
			return
		}
		metrics, err := grabber.Grab(ctx)
		if err != nil {
			framework.Logf("MetricsGrabber failed to grab CA metrics before test (skipping metrics gathering): %v", err)
			return
		}
		framework.Logf("Gathered ClusterAutoscaler metrics before test")
		result = &metrics
	})

	return
}

func GrabAfterEach(ctx context.Context, f *framework.Framework, before *Collection) {
	if framework.TestContext.GatherMetricsAfterTest == "false" {
		return
	}

	ginkgo.By("Gathering metrics after test", func() {
		// Grab apiserver, scheduler, controller-manager metrics and (optionally) nodes' kubelet metrics.
		grabMetricsFromKubelets := framework.TestContext.GatherMetricsAfterTest != "master" && !framework.ProviderIs("kubemark")
		grabber, err := NewMetricsGrabber(ctx, f.ClientSet, f.KubemarkExternalClusterClientSet, f.ClientConfig(), grabMetricsFromKubelets, true, true, true, framework.TestContext.IncludeClusterAutoscalerMetrics, false)
		if err != nil {
			framework.Logf("Failed to create MetricsGrabber (skipping metrics gathering): %v", err)
			return
		}
		received, err := grabber.Grab(ctx)
		if err != nil {
			framework.Logf("MetricsGrabber failed to grab some of the metrics: %v", err)
			return
		}
		if before == nil {
			before = &Collection{}
		}
		(*ComponentCollection)(&received).ComputeClusterAutoscalerMetricsDelta(*before)
		f.TestSummaries = append(f.TestSummaries, (*ComponentCollection)(&received))
	})
}
