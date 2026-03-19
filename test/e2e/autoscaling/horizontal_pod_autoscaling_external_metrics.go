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
		e2eautoscaling.DeleteHorizontalPodAutoscaler(ctx, rc, hpa.Name)
	})

	ginkgo.It("should scale up based on multiple external metrics", func(ctx context.Context) {
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
		metricName2 := "queue_messages_unacknowledged"

		ginkgo.By(fmt.Sprintf("Creating an HPA based on external metrics %s and %s", metricName1, metricName2))
		hpa := e2eautoscaling.CreateCombinedExternalHorizontalPodAutoscaler(ctx, rc, metricName1, metricName2, v2.ValueMetricType, 50, 50, int32(initPods), 3)

		ginkgo.By(fmt.Sprintf("Setting %s metric value to 100 and %s to 0", metricName1, metricName2))
		err := metricsController.SetMetricValue(ctx, metricName1, 100, nil)
		framework.ExpectNoError(err)
		err = metricsController.SetMetricValue(ctx, metricName2, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale up based on the first metric")
		rc.WaitForReplicas(ctx, 2, maxResourceConsumerDelay+waitBuffer)

		ginkgo.By(fmt.Sprintf("Setting %s metric value to 150", metricName2))
		err = metricsController.SetMetricValue(ctx, metricName2, 150, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale up based on the second metric (highest recommendation)")
		rc.WaitForReplicas(ctx, 3, maxResourceConsumerDelay+waitBuffer)

		e2eautoscaling.DeleteHorizontalPodAutoscaler(ctx, rc, hpa.Name)
	})

	ginkgo.It("should respect stabilization window", func(ctx context.Context) {
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
		ginkgo.By(fmt.Sprintf("Creating an HPA based on external metric %s with 30s stabilization window", metricName))
		stabilizationWindow := int32(30)
		behavior := &v2.HorizontalPodAutoscalerBehavior{
			ScaleDown: &v2.HPAScalingRules{
				StabilizationWindowSeconds: &stabilizationWindow,
			},
		}
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx, rc, metricName, nil, v2.ValueMetricType, 50, int32(initPods), 3, behavior)

		ginkgo.By("Waiting for HPA to scale up to max replicas")
		rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), maxResourceConsumerDelay+waitBuffer)

		ginkgo.By(fmt.Sprintf("Setting %s metric value to 0", metricName))
		err := metricsController.SetMetricValue(ctx, metricName, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying that HPA does NOT scale down immediately (respecting stabilization window)")
		// Wait for 15s, which is less than the 30s window
		time.Sleep(15 * time.Second)
		rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), 5*time.Second)

		ginkgo.By("Waiting for stabilization window to pass and HPA to scale down")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), 60*time.Second+waitBuffer)

		e2eautoscaling.DeleteHorizontalPodAutoscaler(ctx, rc, hpa.Name)
	})

	ginkgo.It("should respect scaling behavior limits", func(ctx context.Context) {
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
		ginkgo.By(fmt.Sprintf("Creating an HPA based on external metric %s with 1 pod/minute scale-up limit", metricName))
		period := int32(60)
		value := int32(1)
		behavior := &v2.HorizontalPodAutoscalerBehavior{
			ScaleUp: &v2.HPAScalingRules{
				Policies: []v2.HPAScalingPolicy{
					{
						Type:   v2.PodsScalingPolicy,
						Value:  value,
						PeriodSeconds: period,
					},
				},
			},
		}
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx, rc, metricName, nil, v2.ValueMetricType, 50, int32(initPods), 3, behavior)

		ginkgo.By(fmt.Sprintf("Setting %s metric value to 150 (recommendation would be 3 pods)", metricName))
		err := metricsController.SetMetricValue(ctx, metricName, 150, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying that HPA scales up gradually (limited by policy)")
		rc.WaitForReplicas(ctx, 2, maxResourceConsumerDelay+waitBuffer)

		ginkgo.By("Waiting for next scaling window and verifying scale up to 3 pods")
		rc.WaitForReplicas(ctx, 3, 60*time.Second+waitBuffer)

		e2eautoscaling.DeleteHorizontalPodAutoscaler(ctx, rc, hpa.Name)
	})

	ginkgo.It("should scale to zero based on external metrics", func(ctx context.Context) {
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
		ginkgo.By(fmt.Sprintf("Creating an HPA based on external metric %s with minReplicas: 0", metricName))
		// Disable stabilization window for faster scale-down
		stabilizationWindowZero := int32(0)
		behavior := &v2.HorizontalPodAutoscalerBehavior{
			ScaleDown: &v2.HPAScalingRules{
				StabilizationWindowSeconds: &stabilizationWindowZero,
			},
		}
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscalerWithBehavior(ctx, rc, metricName, nil, v2.ValueMetricType, 50, 0, 3, behavior)

		ginkgo.By(fmt.Sprintf("Setting %s metric value to 0", metricName))
		err := metricsController.SetMetricValue(ctx, metricName, 0, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale down to zero replicas")
		rc.WaitForReplicas(ctx, 0, maxResourceConsumerDelay+waitBuffer)

		ginkgo.By(fmt.Sprintf("Setting %s metric value to 100", metricName))
		err = metricsController.SetMetricValue(ctx, metricName, 100, nil)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale up from zero")
		rc.WaitForReplicas(ctx, 2, maxResourceConsumerDelay+waitBuffer)

		e2eautoscaling.DeleteHorizontalPodAutoscaler(ctx, rc, hpa.Name)
	})
})
