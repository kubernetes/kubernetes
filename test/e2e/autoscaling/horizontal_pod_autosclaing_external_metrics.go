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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eautoscaling "k8s.io/kubernetes/test/e2e/framework/autoscaling"
	admissionapi "k8s.io/pod-security-admission/api"
)

// Requires WithSerial as test sets up external metrics server
var _ = SIGDescribe(feature.HPA, "Horizontal pod autoscaling (external metrics)", framework.WithSerial(), func() {
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
		ginkgo.DeferCleanup(rc.CleanUp)
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
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHPAWithBehavior, rc, hpa.Name)

		ginkgo.By("Waiting for HPA to scale up to max replicas")
		rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), maxResourceConsumerDelay+waitBuffer)

		ginkgo.By(fmt.Sprintf("Setting %s metric value to 0", metricName))
		err := metricsController.SetMetricValue(ctx, metricName, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale down to min replicas")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), maxResourceConsumerDelay+waitBuffer)
	})

	ginkgo.It("should scale based on the highest recommendation across multiple external metrics", func(ctx context.Context) {
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

		metricA := "orders_backlog"
		metricB := "priority_orders_backlog"
		targetAverageValue := int64(25)
		scaleDeadline := maxHPAReactionTime + maxResourceConsumerDelay + waitBuffer

		ginkgo.By("Creating two external metrics used by one HPA")
		err := metricsController.CreateMetric(ctx, metricA, 10, nil, false)
		framework.ExpectNoError(err)
		err = metricsController.CreateMetric(ctx, metricB, 10, nil, false)
		framework.ExpectNoError(err)

		ginkgo.By("Creating an HPA with both external metrics")
		hpa := e2eautoscaling.CreateMultiMetricHorizontalPodAutoscaler(
			ctx,
			rc,
			[]v2.MetricSpec{
				e2eautoscaling.CreateExternalMetricSpec(metricA, nil, v2.AverageValueMetricType, targetAverageValue),
				e2eautoscaling.CreateExternalMetricSpec(metricB, nil, v2.AverageValueMetricType, targetAverageValue),
			},
			int32(initPods),
			4,
		)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHPAWithBehavior, rc, hpa.Name)

		ginkgo.By(fmt.Sprintf("Increasing %s to drive scale up to 3 replicas", metricA))
		err = metricsController.SetMetricValue(ctx, metricA, 60, nil)
		framework.ExpectNoError(err)
		err = metricsController.SetMetricValue(ctx, metricB, 10, nil)
		framework.ExpectNoError(err)
		rc.WaitForReplicas(ctx, 3, scaleDeadline)

		ginkgo.By(fmt.Sprintf("Lowering %s and increasing %s to drive scale up to 4 replicas", metricA, metricB))
		err = metricsController.SetMetricValue(ctx, metricA, 10, nil)
		framework.ExpectNoError(err)
		err = metricsController.SetMetricValue(ctx, metricB, 100, nil)
		framework.ExpectNoError(err)
		rc.WaitForReplicas(ctx, 4, scaleDeadline)
	})

	ginkgo.It("should respect downscale stabilization window when external metric fluctuates", func(ctx context.Context) {
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

		metricName := "fluctuating_queue_depth"
		targetAverageValue := int64(30)
		highValue := int64(120)
		lowValue := int64(20)
		downscaleStabilization := 1 * time.Minute

		ginkgo.By(fmt.Sprintf("Creating external metric %s", metricName))
		err := metricsController.CreateMetric(ctx, metricName, highValue, nil, false)
		framework.ExpectNoError(err)

		ginkgo.By("Creating an HPA with a downscale stabilization window")
		behavior := e2eautoscaling.HPABehaviorWithStabilizationWindows(0*time.Second, downscaleStabilization)
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(
			ctx,
			rc,
			metricName,
			nil,
			v2.AverageValueMetricType,
			targetAverageValue,
			int32(initPods),
			4,
			behavior,
		)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHPAWithBehavior, rc, hpa.Name)

		ginkgo.By("Waiting for initial scale up to 4 replicas")
		scaleDeadline := maxHPAReactionTime + maxResourceConsumerDelay + waitBuffer
		rc.WaitForReplicas(ctx, 4, scaleDeadline)

		ginkgo.By("Fluctuating metric value quickly")
		err = metricsController.SetMetricValue(ctx, metricName, lowValue, nil)
		framework.ExpectNoError(err)
		time.Sleep(10 * time.Second)
		err = metricsController.SetMetricValue(ctx, metricName, highValue, nil)
		framework.ExpectNoError(err)
		time.Sleep(10 * time.Second)

		waitStart := time.Now()
		err = metricsController.SetMetricValue(ctx, metricName, lowValue, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying HPA does not scale down aggressively during stabilization window")
		rc.EnsureDesiredReplicasInRange(ctx, 4, 4, downscaleStabilization-20*time.Second, hpa.Name)

		ginkgo.By("Waiting for scale down after stabilization window has passed")
		scaleDownDeadline := downscaleStabilization + maxHPAReactionTime + maxResourceConsumerDelay + waitBuffer
		rc.WaitForReplicas(ctx, 1, scaleDownDeadline)
		timeWaited := time.Since(waitStart)

		ginkgo.By("Verifying scale down happened only after stabilization window")
		gomega.Expect(timeWaited).To(gomega.BeNumerically(">", downscaleStabilization), "waited %s, wanted more than %s", timeWaited, downscaleStabilization)
		gomega.Expect(timeWaited).To(gomega.BeNumerically("<", scaleDownDeadline), "waited %s, wanted less than %s", timeWaited, scaleDownDeadline)
	})

	ginkgo.It("should enforce configured scale up and scale down pod limits for external metrics", func(ctx context.Context) {
		ginkgo.By("Creating the resource consumer deployment")
		initPods := 2
		rc = e2eautoscaling.NewDynamicResourceConsumer(ctx,
			hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
			0, 0, 0,
			int64(podCPURequest), 200,
			f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			nil)
		ginkgo.DeferCleanup(rc.CleanUp)
		rc.WaitForReplicas(ctx, initPods, maxResourceConsumerDelay+waitBuffer)

		metricName := "rate_limited_queue_depth"
		targetAverageValue := int64(20)
		limitWindowLength := 1 * time.Minute
		scaleDeadline := limitWindowLength + maxHPAReactionTime + maxResourceConsumerDelay + waitBuffer
		podsLimitPerWindow := int32(1)

		ginkgo.By(fmt.Sprintf("Creating external metric %s", metricName))
		err := metricsController.CreateMetric(ctx, metricName, 80, nil, false)
		framework.ExpectNoError(err)

		ginkgo.By("Creating an HPA with scaleUp and scaleDown pod limits")
		scaleUpRule := e2eautoscaling.HPAScalingRuleWithScalingPolicy(v2.PodsScalingPolicy, podsLimitPerWindow, int32(limitWindowLength.Seconds()))
		scaleDownRule := e2eautoscaling.HPAScalingRuleWithScalingPolicy(v2.PodsScalingPolicy, podsLimitPerWindow, int32(limitWindowLength.Seconds()))
		behavior := e2eautoscaling.HPABehaviorWithScaleUpAndDownRules(scaleUpRule, scaleDownRule)
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(
			ctx,
			rc,
			metricName,
			nil,
			v2.AverageValueMetricType,
			targetAverageValue,
			int32(initPods),
			4,
			behavior,
		)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHPAWithBehavior, rc, hpa.Name)

		ginkgo.By("Triggering scale up to the maximum replicas")
		rc.WaitForReplicas(ctx, 3, scaleDeadline)
		waitStart := time.Now()
		rc.WaitForReplicas(ctx, 4, scaleDeadline)
		timeWaitedFor4 := time.Since(waitStart)
		gomega.Expect(timeWaitedFor4).To(gomega.BeNumerically(">", limitWindowLength), "waited %s, wanted more than %s", timeWaitedFor4, limitWindowLength)
		gomega.Expect(timeWaitedFor4).To(gomega.BeNumerically("<", scaleDeadline), "waited %s, wanted less than %s", timeWaitedFor4, scaleDeadline)

		ginkgo.By("Lowering metric to trigger scale down")
		err = metricsController.SetMetricValue(ctx, metricName, 40, nil)
		framework.ExpectNoError(err)
		rc.WaitForReplicas(ctx, 3, scaleDeadline)
		waitStart = time.Now()
		rc.WaitForReplicas(ctx, 2, scaleDeadline)
		timeWaitedFor2 := time.Since(waitStart)
		gomega.Expect(timeWaitedFor2).To(gomega.BeNumerically(">", limitWindowLength), "waited %s, wanted more than %s", timeWaitedFor2, limitWindowLength)
		gomega.Expect(timeWaitedFor2).To(gomega.BeNumerically("<", scaleDeadline), "waited %s, wanted less than %s", timeWaitedFor2, scaleDeadline)
	})
})

// Requires WithSerial as test sets up external metrics server
var _ = SIGDescribe(feature.HPA, framework.WithFeatureGate(features.HPAScaleToZero), framework.WithSerial(),
	"Horizontal pod autoscaling (scale to zero)", func() {
		var (
			rc                *e2eautoscaling.ResourceConsumer
			metricsController *e2eautoscaling.ExternalMetricsController
		)

		waitBuffer := 1 * time.Minute

		f := framework.NewDefaultFramework("horizontal-pod-autoscaling-scale-to-zero")
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

		ginkgo.It("should scale down to zero and back up based on external metric value", func(ctx context.Context) {
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

			ginkgo.By(fmt.Sprintf("Creating an HPA with minReplicas=0 based on external metric %s", metricName))
			stabilizationWindowZero := int32(0)
			behavior := &v2.HorizontalPodAutoscalerBehavior{
				ScaleDown: &v2.HPAScalingRules{
					StabilizationWindowSeconds: &stabilizationWindowZero,
				},
			}
			hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx,
				rc, metricName, nil, v2.ValueMetricType, 50,
				0, 3, behavior)
			ginkgo.DeferCleanup(e2eautoscaling.DeleteHPAWithBehavior, rc, hpa.Name)

			ginkgo.By(fmt.Sprintf("Setting %s metric value to 0 to trigger scale to zero", metricName))
			err := metricsController.SetMetricValue(ctx, metricName, 0, nil)
			framework.ExpectNoError(err)

			ginkgo.By("Waiting for HPA to scale down to zero replicas")
			rc.WaitForReplicas(ctx, 0, maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)

			ginkgo.By("Verifying the ScaledToZero condition is True")
			updatedHPA, err := f.ClientSet.AutoscalingV2().HorizontalPodAutoscalers(f.Namespace.Name).Get(ctx, hpa.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			var scaledToZeroCondition *v2.HorizontalPodAutoscalerCondition
			for i := range updatedHPA.Status.Conditions {
				if updatedHPA.Status.Conditions[i].Type == v2.ScaledToZero {
					scaledToZeroCondition = &updatedHPA.Status.Conditions[i]
					break
				}
			}
			gomega.Expect(scaledToZeroCondition).NotTo(gomega.BeNil(), "expected ScaledToZero condition to be present")
			gomega.Expect(scaledToZeroCondition.Status).To(gomega.Equal(v1.ConditionTrue), "expected ScaledToZero condition to be True")

			ginkgo.By(fmt.Sprintf("Setting %s metric value to 200 to trigger scale from zero", metricName))
			err = metricsController.SetMetricValue(ctx, metricName, 200, nil)
			framework.ExpectNoError(err)

			ginkgo.By("Waiting for HPA to scale back up from zero")
			rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)
		})
	})
