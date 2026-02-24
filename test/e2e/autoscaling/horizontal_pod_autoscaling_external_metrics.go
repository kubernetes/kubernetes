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
	"github.com/onsi/gomega"
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

	ginkgo.It("should scale up and down based on external metric value", framework.WithSerial(), func(ctx context.Context) {
		ginkgo.By("Creating the resource consumer deployment")
		initPods := 1
		rc = e2eautoscaling.NewDynamicResourceConsumer(ctx,
			hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
			0, 0, 0,
			int64(podCPURequest), 200,
			f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			nil)
		ginkgo.DeferCleanup(rc.CleanUp)
		rc.WaitForReplicas(ctx, initPods, maxResourceConsumerDelay+waitBuffer)

		metricName := "queue_messages_ready"
		ginkgo.By(fmt.Sprintf("Creating an HPA based on external metric %s", metricName))
		// Disable stabilization window for faster scale-up and scale-down
		behavior := e2eautoscaling.HPABehaviorWithStabilizationWindows(0, 0)
		// since queue_messages_ready default is 100 this will cause the HPA to scale out till max replicas
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx, rc, metricName, nil, v2.ValueMetricType, 50, int32(initPods), 3, behavior)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)

		ginkgo.By("Waiting for HPA to scale up to max replicas")
		rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)

		ginkgo.By(fmt.Sprintf("Setting %s metric value to 0", metricName))
		err := metricsController.SetMetricValue(ctx, metricName, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale down to min replicas")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)
	})

	ginkgo.It("should scale based on multiple external metrics", framework.WithSerial(), func(ctx context.Context) {
		ginkgo.By("Creating the resource consumer deployment")
		initPods := 1
		rc = e2eautoscaling.NewDynamicResourceConsumer(ctx,
			hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
			0, 0, 0,
			int64(podCPURequest), 200,
			f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			nil)
		ginkgo.DeferCleanup(rc.CleanUp)
		rc.WaitForReplicas(ctx, initPods, maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Setting queue_messages_ready to 0")
		err := metricsController.SetMetricValue(ctx, "queue_messages_ready", 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Creating an HPA with two external metrics: one at 0, one requiring max replicas")
		metrics := []v2.MetricSpec{
			e2eautoscaling.CreateExternalMetricSpec("queue_messages_ready", nil, v2.ValueMetricType, 50),
			e2eautoscaling.CreateExternalMetricSpec("http_requests_total", nil, v2.ValueMetricType, 100),
		}
		// Disable stabilization window for faster scale-up and scale-down
		behavior := e2eautoscaling.HPABehaviorWithStabilizationWindows(0, 0)
		hpa := e2eautoscaling.CreateMultiMetricHPAWithBehavior(ctx, rc, metrics, int32(initPods), 5, behavior)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)

		ginkgo.By("Waiting for HPA to scale up to max replicas (http_requests_total dominates)")
		rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Setting http_requests_total to 0 to trigger scale down")
		err = metricsController.SetMetricValue(ctx, "http_requests_total", 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale down to min replicas")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)
	})

	ginkgo.It("should scale to intermediate replica count when dominant metric drops", framework.WithSerial(), func(ctx context.Context) {
		ginkgo.By("Creating the resource consumer deployment")
		initPods := 1
		rc = e2eautoscaling.NewDynamicResourceConsumer(ctx,
			hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
			0, 0, 0,
			int64(podCPURequest), 200,
			f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			nil)
		ginkgo.DeferCleanup(rc.CleanUp)
		rc.WaitForReplicas(ctx, initPods, maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Creating an HPA based on two external metrics with different desired replica counts")
		// queue_messages_ready default=100, target_average=50 -> wants ceil(100/50)=2 replicas
		// http_requests_total default=500, target_average=200 -> wants ceil(500/200)=3 replicas (intermediate, not max)
		// HPA takes the maximum desired replica count across all metrics -> 3
		metrics := []v2.MetricSpec{
			e2eautoscaling.CreateExternalMetricSpec("queue_messages_ready", nil, v2.AverageValueMetricType, 50),
			e2eautoscaling.CreateExternalMetricSpec("http_requests_total", nil, v2.AverageValueMetricType, 200),
		}
		// Disable stabilization window for faster scale-up and scale-down
		behavior := e2eautoscaling.HPABehaviorWithStabilizationWindows(0, 0)
		hpa := e2eautoscaling.CreateMultiMetricHPAWithBehavior(ctx, rc, metrics, int32(initPods), 5, behavior)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)

		ginkgo.By("Waiting for HPA to scale up to 3 replicas")
		rc.WaitForReplicas(ctx, 3, maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Setting http_requests_total to 100")
		// http_requests_total=100, target_average=200 -> wants ceil(100/200)=1, floored to minReplicas=1
		// queue_messages_ready=100 (default), target_average=50 -> wants ceil(100/50)=2 replicas
		// HPA takes MAX(1, 2) = 2 -> should scale down to 2, not to min
		err := metricsController.SetMetricValue(ctx, "http_requests_total", 100, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale down to 2 replicas (queue_messages_ready now dominates)")
		rc.WaitForReplicas(ctx, 2, maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Verifying HPA does not scale below 2 while queue_messages_ready still requires 2 replicas")
		rc.EnsureDesiredReplicasInRange(ctx, 2, 2, 2*hpaReconciliationInterval+waitBuffer, hpa.Name)

		ginkgo.By("Setting queue_messages_ready to 0 to allow scale down to min replicas")
		// Both metrics now 0 -> both want min replicas -> HPA scales down to min
		err = metricsController.SetMetricValue(ctx, "queue_messages_ready", 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale down to min replicas")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)
	})

	ginkgo.It("should respect the stabilization window for scale down", framework.WithSerial(), func(ctx context.Context) {
		ginkgo.By("Creating the resource consumer deployment")
		initPods := 1
		rc = e2eautoscaling.NewDynamicResourceConsumer(ctx,
			hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
			0, 0, 0,
			int64(podCPURequest), 200,
			f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			nil)
		ginkgo.DeferCleanup(rc.CleanUp)
		rc.WaitForReplicas(ctx, initPods, maxResourceConsumerDelay+waitBuffer)

		metricName := "queue_messages_ready"
		downScaleStabilization := 60 * time.Second
		ginkgo.By(fmt.Sprintf("Creating an HPA with %s downscale stabilization window", downScaleStabilization))
		// queue_messages_ready default=100, target=20 -> wants ceil(1×100/20)=5 replicas
		// Upscale stabilization is 0 so scale-up happens quickly; downscale stabilization is 60s
		behavior := e2eautoscaling.HPABehaviorWithStabilizationWindows(0, downScaleStabilization)
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx, rc, metricName, nil, v2.ValueMetricType, 20, int32(initPods), 5, behavior)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)

		ginkgo.By("Waiting for HPA to scale up to max replicas")
		rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)

		ginkgo.By(fmt.Sprintf("Setting %s metric to 0 to trigger scale down", metricName))
		err := metricsController.SetMetricValue(ctx, metricName, 0, nil)
		framework.ExpectNoError(err)
		waitStart := time.Now()

		ginkgo.By("Waiting for HPA to scale down to min replicas after stabilization window")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), downScaleStabilization+maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)
		timeWaited := time.Since(waitStart)

		ginkgo.By("Verifying scale down respected the stabilization window")
		framework.Logf("time waited for scale down: %s", timeWaited)
		gomega.Expect(timeWaited).To(gomega.BeNumerically(">", downScaleStabilization),
			"waited %s, wanted more than %s", timeWaited, downScaleStabilization)
		deadline := downScaleStabilization + maxHPAReactionTime + maxResourceConsumerDelay + waitBuffer
		gomega.Expect(timeWaited).To(gomega.BeNumerically("<", deadline),
			"waited %s, wanted less than %s", timeWaited, deadline)
	})

	ginkgo.It("should respect the scaling limits when scaling up", framework.WithSerial(), func(ctx context.Context) {
		ginkgo.By("Creating the resource consumer deployment")
		initPods := 1
		rc = e2eautoscaling.NewDynamicResourceConsumer(ctx,
			hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
			0, 0, 0,
			int64(podCPURequest), 200,
			f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			nil)
		ginkgo.DeferCleanup(rc.CleanUp)
		rc.WaitForReplicas(ctx, initPods, maxResourceConsumerDelay+waitBuffer)

		metricName := "queue_messages_ready"
		scaleLimitWindow := 1 * time.Minute
		podsLimitPerWindow := int32(1)
		ginkgo.By(fmt.Sprintf("Creating an HPA with scaleUp limited to %d pod(s) per %s", podsLimitPerWindow, scaleLimitWindow))
		// queue_messages_ready default=100, target=20 -> wants ceil(1×100/20)=5 replicas
		// But scaleUp is rate-limited to 1 pod per minute, so scaling is gradual
		behavior := e2eautoscaling.HPABehaviorWithScaleLimitedByNumberOfPods(
			e2eautoscaling.ScaleUpDirection, podsLimitPerWindow, int32(scaleLimitWindow/time.Second))
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx, rc, metricName, nil, v2.ValueMetricType, 20, int32(initPods), 5, behavior)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)

		// deadlineFor2: first scale is not rate-limited, so it should happen quickly.
		deadlineFor2 :=  waitBuffer + maxHPAReactionTime + maxResourceConsumerDelay
		// deadlineFor3: second scale must wait the full rate-limit window, plus HPA reaction and
		// resource consumer delay.
		deadlineFor3 := deadlineFor2 + scaleLimitWindow

		waitStart := time.Now()
		rc.WaitForReplicas(ctx, 2, maxHPAReactionTime+maxResourceConsumerDelay+scaleLimitWindow)
		timeWaitedFor2 := time.Since(waitStart)

		waitStart = time.Now()
		rc.WaitForReplicas(ctx, 3, maxHPAReactionTime+maxResourceConsumerDelay+scaleLimitWindow+waitBuffer)
		timeWaitedFor3 := time.Since(waitStart)

		ginkgo.By("Verifying time waited for scale up to 2 replicas (first scale can happen quickly)")
		framework.Logf("time waited for scale up to 2 replicas: %s", timeWaitedFor2)
		gomega.Expect(timeWaitedFor2).To(gomega.BeNumerically("<", deadlineFor2),
			"waited %s, wanted less than %s", timeWaitedFor2, deadlineFor2)

		ginkgo.By("Verifying time waited for scale up to 3 replicas (rate limit enforced)")
		framework.Logf("time waited for scale up to 3 replicas: %s", timeWaitedFor3)
		gomega.Expect(timeWaitedFor3).To(gomega.BeNumerically(">", scaleLimitWindow),
			"waited %s, wanted more than %s", timeWaitedFor3, scaleLimitWindow)
		gomega.Expect(timeWaitedFor3).To(gomega.BeNumerically("<", deadlineFor3),
			"waited %s, wanted less than %s", timeWaitedFor3, deadlineFor3)
	})
})
