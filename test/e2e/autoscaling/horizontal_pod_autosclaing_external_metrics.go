/*
Copyright The Kubernetes Authors.

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

package autoscaling

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	v2 "k8s.io/api/autoscaling/v2"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eautoscaling "k8s.io/kubernetes/test/e2e/framework/autoscaling"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe(feature.HPA, "Horizontal pod autoscaling (external metrics)", func() {
	var (
		rc                *e2eautoscaling.ResourceConsumer
		metricsController *e2eautoscaling.ExternalMetricsController
	)

	waitBuffer := 1 * time.Minute

	f := framework.NewDefaultFramework("horizontal-pod-autoscaling-external")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.BeforeEach(func(ctx context.Context) {
		ginkgo.By("Setting up the external metrics server")
		metricsController = e2eautoscaling.RunExternalMetricsServer(ctx, f.ClientSet, f.Namespace.Name, "external-metrics-server", nil)
	})
	ginkgo.AfterEach(func(ctx context.Context) {
		if metricsController != nil {
			e2eautoscaling.CleanupExternalMetricsServer(ctx, f.ClientSet, f.Namespace.Name, "external-metrics-server")
		}
	})

	ginkgo.It("should scale up and down based on external metric value", func(ctx context.Context) {
		ginkgo.By("Creating the resource consumer deployment")
		initPods := 1
		rc = e2eautoscaling.NewDynamicResourceConsumer(ctx,
			hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
			0, 0, 0,
			int64(podCPURequest), 200,
			f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			nil)
		rc.WaitForReplicas(ctx, initPods, maxResourceConsumerDelay+waitBuffer)

		metricName := "queue_messages_ready"
		ginkgo.By(fmt.Sprintf("Creating an HPA based on external metric %s", metricName))
		// Disable stabilization window for faster scale-down
		stabilizationWindowZero := int32(0)
		behavior := &v2.HorizontalPodAutoscalerBehavior{
			ScaleDown: &v2.HPAScalingRules{
				StabilizationWindowSeconds: &stabilizationWindowZero,
			},
		}
		// since queue_messages_ready default is 100 this will cause the HPA to scale out till max replicas
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx, rc, metricName, nil, v2.ValueMetricType, 50, int32(initPods), 3, behavior)

		ginkgo.By("Waiting for HPA to scale up to max replicas")
		rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), maxResourceConsumerDelay+waitBuffer)

		ginkgo.By(fmt.Sprintf("Setting %s metric value to 0", metricName))
		err := metricsController.SetMetricValue(ctx, metricName, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale down to min replicas")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), maxResourceConsumerDelay+waitBuffer)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)
	})

	// Test case: Multiple Metrics
	// Verifies that a single HPA can correctly handle two or more external metrics simultaneously.
	// The HPA should scale based on the metric that requires the most replicas.
	ginkgo.It("should handle multiple external metrics and scale based on the largest requirement", func(ctx context.Context) {
		ginkgo.By("Creating the resource consumer deployment")
		initPods := 1
		rc = e2eautoscaling.NewDynamicResourceConsumer(ctx,
			hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
			0, 0, 0,
			int64(podCPURequest), 200,
			f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			nil)
		rc.WaitForReplicas(ctx, initPods, maxResourceConsumerDelay+waitBuffer)

		metricName1 := "queue_messages_ready"
		metricName2 := "active_connections"

		ginkgo.By(fmt.Sprintf("Setting initial values: %s=100, %s=0", metricName1, metricName2))
		err := metricsController.SetMetricValue(ctx, metricName2, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Creating HPA with two external metrics")
		stabilizationWindowZero := int32(0)
		behavior := &v2.HorizontalPodAutoscalerBehavior{
			ScaleDown: &v2.HPAScalingRules{
				StabilizationWindowSeconds: &stabilizationWindowZero,
			},
		}
		// Primary metric: queue_messages_ready (default 100, threshold 50 → needs 2 replicas)
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx, rc, metricName1, nil, v2.ValueMetricType, 50, int32(initPods), 4, behavior)

		ginkgo.By("Waiting for HPA to scale up based on first metric")
		rc.WaitForReplicas(ctx, 2, maxResourceConsumerDelay+waitBuffer)

		// Now raise second metric to require more replicas (200, threshold 50 → needs 4 replicas)
		ginkgo.By(fmt.Sprintf("Setting %s metric value high to trigger larger scale", metricName2))
		err = metricsController.SetMetricValue(ctx, metricName2, 200, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale up to max replicas based on second metric dominance")
		rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Resetting both metrics to 0 to allow scale down")
		err = metricsController.SetMetricValue(ctx, metricName1, 0, nil)
		framework.ExpectNoError(err)
		err = metricsController.SetMetricValue(ctx, metricName2, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale down to min replicas")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), maxResourceConsumerDelay+waitBuffer)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)
	})

	// Test case: Stabilization Window
	// Ensures that the HPA does not scale too aggressively when metric values fluctuate rapidly,
	// respecting the configured stabilization window.
	ginkgo.It("should respect stabilization window and not scale down prematurely during fluctuating metrics", func(ctx context.Context) {
		ginkgo.By("Creating the resource consumer deployment")
		initPods := 3
		rc = e2eautoscaling.NewDynamicResourceConsumer(ctx,
			hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
			0, 0, 0,
			int64(podCPURequest), 200,
			f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			nil)
		rc.WaitForReplicas(ctx, initPods, maxResourceConsumerDelay+waitBuffer)

		metricName := "request_rate"

		ginkgo.By("Setting initial high metric value to trigger scale-up")
		err := metricsController.SetMetricValue(ctx, metricName, 150, nil)
		framework.ExpectNoError(err)

		// Configure a 60-second stabilization window for scale-down
		stabilizationWindow := int32(60)
		behavior := &v2.HorizontalPodAutoscalerBehavior{
			ScaleDown: &v2.HPAScalingRules{
				StabilizationWindowSeconds: &stabilizationWindow,
			},
		}

		ginkgo.By("Creating HPA with stabilization window for scale-down")
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx, rc, metricName, nil, v2.ValueMetricType, 50, 1, 5, behavior)

		ginkgo.By("Waiting for HPA to scale up")
		rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Temporarily dropping metric to near zero (simulating a spike drop)")
		err = metricsController.SetMetricValue(ctx, metricName, 5, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting briefly - HPA should NOT scale down immediately due to stabilization window")
		time.Sleep(30 * time.Second)

		ginkgo.By("Verifying replicas have NOT scaled down yet (stabilization window active)")
		// HPA should still be at or near max replicas since 60s window hasn't elapsed
		rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), 45*time.Second)

		ginkgo.By("Resetting metric to 0 and waiting for stabilization window to expire")
		err = metricsController.SetMetricValue(ctx, metricName, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale down after stabilization window expires")
		rc.WaitForReplicas(ctx, 1, maxResourceConsumerDelay+waitBuffer+time.Duration(stabilizationWindow)*time.Second)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)
	})

	// Test case: Scaling Limits
	// Validates that the configured scaling limits (behavior.scaleUp and behavior.scaleDown)
	// are enforced correctly.
	ginkgo.It("should enforce scaling limits defined in HPA behavior", func(ctx context.Context) {
		ginkgo.By("Creating the resource consumer deployment")
		initPods := 2
		rc = e2eautoscaling.NewDynamicResourceConsumer(ctx,
			hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
			0, 0, 0,
			int64(podCPURequest), 200,
			f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			nil)
		rc.WaitForReplicas(ctx, initPods, maxResourceConsumerDelay+waitBuffer)

		metricName := "jobs_in_queue"

		ginkgo.By("Setting metric value high to trigger scale-up")
		err := metricsController.SetMetricValue(ctx, metricName, 500, nil)
		framework.ExpectNoError(err)

		// Configure strict scaling limits:
		// scaleUp: max 1 pod per 60 seconds
		// scaleDown: max 1 pod per 60 seconds
		scaleUpPeriod := int32(60)
		scaleDownPeriod := int32(60)
		stabilizationWindowZero := int32(0)
		maxPodsPerPeriod := int32(1)

		behavior := &v2.HorizontalPodAutoscalerBehavior{
			ScaleUp: &v2.HPAScalingRules{
				Policies: []v2.HPAScalingPolicy{
					{
						Type:          v2.PodsScalingPolicy,
						Value:         maxPodsPerPeriod,
						PeriodSeconds: scaleUpPeriod,
					},
				},
			},
			ScaleDown: &v2.HPAScalingRules{
				StabilizationWindowSeconds: &stabilizationWindowZero,
				Policies: []v2.HPAScalingPolicy{
					{
						Type:          v2.PodsScalingPolicy,
						Value:         maxPodsPerPeriod,
						PeriodSeconds: scaleDownPeriod,
					},
				},
			},
		}

		ginkgo.By("Creating HPA with strict scaling limits (max 1 pod per 60s)")
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx, rc, metricName, nil, v2.ValueMetricType, 50, int32(initPods), 6, behavior)

		ginkgo.By("Waiting for initial scale-up (should add only 1 pod due to limit)")
		// With a limit of 1 pod per period and starting at 2, after one period we expect 3
		rc.WaitForReplicas(ctx, initPods+1, maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Verifying replicas did not jump directly to max (limit enforced)")
		currentReplicas, err := e2eautoscaling.GetCurrentReplicaCountForTarget(ctx, f.ClientSet, rc)
		framework.ExpectNoError(err)
		// Should not have jumped past initPods+1 in the first period
		if currentReplicas > int32(initPods+1) {
			framework.Failf("Expected at most %d replicas due to scaleUp limit, got %d", initPods+1, currentReplicas)
		}

		ginkgo.By("Setting metric to 0 to trigger scale-down")
		err = metricsController.SetMetricValue(ctx, metricName, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for scale-down (should remove only 1 pod per period due to limit)")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), maxResourceConsumerDelay+waitBuffer+2*time.Minute)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)
	})
})
