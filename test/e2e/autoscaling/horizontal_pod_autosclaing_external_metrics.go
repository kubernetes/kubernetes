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

	ginkgo.It("should scale based on highest value when multiple external metrics are configured", func(ctx context.Context) {
		ginkgo.By("Creating the resource consumer deployment")
		initPods := 1
		rc = e2eautoscaling.NewDynamicResourceConsumer(ctx,
			hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
			0, 0, 0,
			int64(podCPURequest), 200,
			f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			nil)
		rc.WaitForReplicas(ctx, initPods, maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Creating an HPA with multiple external metrics")
		stabilizationWindowZero := int32(0)
		behavior := &v2.HorizontalPodAutoscalerBehavior{
			ScaleDown: &v2.HPAScalingRules{
				StabilizationWindowSeconds: &stabilizationWindowZero,
			},
		}

		metric1Name := "queue_messages_ready"
		metric2Name := "request_latency_seconds"

		hpa := e2eautoscaling.CreateMultiMetricHorizontalPodAutoscaler(ctx, rc, metric1Name, metric2Name, nil, v2.ValueMetricType, 50, int32(initPods), 3, behavior)

		ginkgo.By("Setting first metric to trigger scale-up")
		err := metricsController.SetMetricValue(ctx, metric1Name, 200, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale up based on first metric")
		rc.WaitForReplicas(ctx, 3, maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Setting second metric higher to test metric selection")
		err = metricsController.SetMetricValue(ctx, metric2Name, 300, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale to max replicas based on higher metric")
		rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Setting both metrics to 0 to trigger scale-down")
		err = metricsController.SetMetricValue(ctx, metric1Name, 0, nil)
		framework.ExpectNoError(err)
		err = metricsController.SetMetricValue(ctx, metric2Name, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale down to min replicas")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), maxResourceConsumerDelay+waitBuffer)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)
	})

	ginkgo.It("should respect stabilization window before scaling", func(ctx context.Context) {
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
		ginkgo.By("Creating an HPA with 5-minute stabilization window")
		stabilizationWindow := int32(300)
		behavior := &v2.HorizontalPodAutoscalerBehavior{
			ScaleUp: &v2.HPAScalingRules{
				StabilizationWindowSeconds: &stabilizationWindow,
			},
			ScaleDown: &v2.HPAScalingRules{
				StabilizationWindowSeconds: &stabilizationWindow,
			},
		}

		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx, rc, metricName, nil, v2.ValueMetricType, 50, int32(initPods), 3, behavior)

		ginkgo.By("Temporarily setting metric to high value")
		err := metricsController.SetMetricValue(ctx, metricName, 200, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting briefly to verify no immediate scale-up due to stabilization")
		time.Sleep(30 * time.Second)

		currentReplicas := rc.ReplicasCount
		gomega.Expect(currentReplicas).To(gomega.Equal(initPods), "HPA should not scale up during stabilization window")

		ginkgo.By("Verifying HPA status indicates stabilization")
		updatedHPA, err := f.ClientSet.AutoscalingV2().HorizontalPodAutoscalers(f.Namespace.Name).Get(ctx, hpa.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(updatedHPA.Status.DesiredReplicas).To(gomega.Equal(int32(initPods)), "DesiredReplicas should remain unchanged during stabilization")

		ginkgo.By("Setting metric to 0 and waiting for eventual scale-down")
		err = metricsController.SetMetricValue(ctx, metricName, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for stabilization window to pass and scale-down to occur")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), 360*time.Second+waitBuffer)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)
	})

	ginkgo.It("should enforce scaling rate limits per minute", func(ctx context.Context) {
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
		ginkgo.By("Creating an HPA with scaling rate limits (200% scale-up, 50% scale-down per minute)")
		scaleUpPercent := int32(200)
		scaleDownPercent := int32(50)
		periodSeconds := int32(60)

		behavior := &v2.HorizontalPodAutoscalerBehavior{
			ScaleUp: &v2.HPAScalingRules{
				SelectPolicy: getPtr(v2.MaxChangePercentScalingPolicy),
				Policies: []v2.HPAScalingPolicy{
					{
						Type:          v2.PercentScalingPolicy,
						Value:         scaleUpPercent,
						PeriodSeconds: periodSeconds,
					},
				},
			},
			ScaleDown: &v2.HPAScalingRules{
				SelectPolicy: getPtr(v2.MinChangePercentScalingPolicy),
				Policies: []v2.HPAScalingPolicy{
					{
						Type:          v2.PercentScalingPolicy,
						Value:         scaleDownPercent,
						PeriodSeconds: periodSeconds,
					},
				},
			},
		}

		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx, rc, metricName, nil, v2.ValueMetricType, 50, int32(initPods), 10, behavior)

		ginkgo.By("Setting high metric value to trigger scale-up")
		err := metricsController.SetMetricValue(ctx, metricName, 500, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for first scale-up cycle (max 200% increase = up to 3 replicas from 1)")
		rc.WaitForReplicas(ctx, 3, maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Verifying replicas respect 200% scale-up rate")
		currentReplicas := rc.ReplicasCount
		gomega.Expect(currentReplicas).To(gomega.BeNumerically("<=", 3), "Scale-up should be limited to 200% per minute")

		ginkgo.By("Setting metric to 0 to trigger scale-down")
		err = metricsController.SetMetricValue(ctx, metricName, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for scale-down with 50% limit per minute")
		rc.WaitForReplicas(ctx, 1, maxResourceConsumerDelay+waitBuffer)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)
	})
})

func getPtr(policy v2.ScalingPolicySelectPolicy) *v2.ScalingPolicySelectPolicy {
	return &policy
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
