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
	"errors"
	"fmt"
	"sync"
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
	"k8s.io/utils/ptr"
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
		ginkgo.DeferCleanup(e2eautoscaling.CleanupExternalMetricsServer, f.ClientSet, f.Namespace.Name, "external-metrics-server")
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
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)

		ginkgo.By("Waiting for HPA to scale up to max replicas")
		rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), maxResourceConsumerDelay+waitBuffer)

		ginkgo.By(fmt.Sprintf("Setting %s metric value to 0", metricName))
		err := metricsController.SetMetricValue(ctx, metricName, 0, nil)
		framework.ExpectNoError(err, "set external metric %q to %d", metricName, 0)

		ginkgo.By("Waiting for HPA to scale down to min replicas")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), maxResourceConsumerDelay+waitBuffer)
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
			ginkgo.DeferCleanup(e2eautoscaling.CleanupExternalMetricsServer, f.ClientSet, f.Namespace.Name, "external-metrics-server")
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
			framework.ExpectNoError(err, "set external metric %q to %d", metricName, 0)

			ginkgo.By("Waiting for HPA to scale down to zero replicas")
			rc.WaitForReplicas(ctx, 0, maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)

			ginkgo.By("Verifying the ScaledToZero condition is True")
			updatedHPA, err := f.ClientSet.AutoscalingV2().HorizontalPodAutoscalers(f.Namespace.Name).Get(ctx, hpa.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "get HorizontalPodAutoscaler %q in namespace %q", hpa.Name, f.Namespace.Name)
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
			framework.ExpectNoError(err, "set external metric %q to %d", metricName, 200)

			ginkgo.By("Waiting for HPA to scale back up from zero")
			rc.WaitForReplicas(ctx, int(hpa.Spec.MaxReplicas), maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)
		})
	})

// Requires WithSerial as test sets up external metrics server
var _ = SIGDescribe(feature.HPA, framework.WithFeatureGate(features.HPAConfigurableSyncPeriod), framework.WithSerial(), framework.WithSlow(),
	"Horizontal pod autoscaling (configurable sync period)", func() {
		var metricsController *e2eautoscaling.ExternalMetricsController

		f := framework.NewDefaultFramework("horizontal-pod-autoscaling-sync-period")
		f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("Setting up the external metrics server")
			metricsController = e2eautoscaling.RunExternalMetricsServer(ctx, f.ClientSet, f.Namespace.Name, "external-metrics-server", nil)
			ginkgo.DeferCleanup(e2eautoscaling.CleanupExternalMetricsServer, f.ClientSet, f.Namespace.Name, "external-metrics-server")
		})

		ginkgo.It("should reconcile HPAs with short, default, and long sync periods independently", ginkgo.SpecTimeout(15*time.Minute), func(ctx context.Context) {
			const samples = 3
			// Several observations leave room for individual delays while distinguishing
			// overrides from the usual 15-second global period. The unset case has no
			// lower bound because the cluster's global period may be customized.
			tests := []struct {
				name        string
				period      *int32
				minDuration time.Duration
				maxDuration time.Duration
			}{
				{name: "short", period: ptr.To[int32](3), maxDuration: 30 * time.Second},
				{name: "default", maxDuration: 2 * time.Minute},
				{name: "long", period: ptr.To[int32](60), minDuration: 90 * time.Second, maxDuration: 4 * time.Minute},
			}

			for _, tc := range tests {
				name := "consumer-" + tc.name
				metricName := "sync_period_" + tc.name
				framework.ExpectNoError(metricsController.CreateMetric(ctx, metricName, 1, nil, false), "create external metric %q", metricName)

				rc := e2eautoscaling.NewDynamicResourceConsumer(ctx,
					name, f.Namespace.Name, e2eautoscaling.KindDeployment, 1,
					0, 0, 0, int64(podCPURequest), 200,
					f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle, nil)
				ginkgo.DeferCleanup(rc.CleanUp)
				rc.WaitForReplicas(ctx, 1, maxResourceConsumerDelay+time.Minute)

				hpa, err := f.ClientSet.AutoscalingV2().HorizontalPodAutoscalers(f.Namespace.Name).Create(ctx, &v2.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec: v2.HorizontalPodAutoscalerSpec{
						ScaleTargetRef: v2.CrossVersionObjectReference{APIVersion: "apps/v1", Kind: "Deployment", Name: name},
						MinReplicas:    ptr.To[int32](1),
						MaxReplicas:    2,
						Metrics: []v2.MetricSpec{
							e2eautoscaling.CreateExternalMetricSpec(metricName, nil, v2.ValueMetricType, 100),
						},
						SyncPeriodSeconds: tc.period,
					},
				}, metav1.CreateOptions{})
				framework.ExpectNoError(err, "create HPA %q", name)
				ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)
			}

			waitForMetric := func(ctx context.Context, name, metricName string, value int64) error {
				return framework.Gomega().Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (int64, error) {
					hpa, err := f.ClientSet.AutoscalingV2().HorizontalPodAutoscalers(f.Namespace.Name).Get(ctx, name, metav1.GetOptions{})
					if err != nil {
						return 0, err
					}
					for _, metric := range hpa.Status.CurrentMetrics {
						if metric.External != nil && metric.External.Metric.Name == metricName && metric.External.Current.Value != nil {
							return metric.External.Current.Value.Value(), nil
						}
					}
					return 0, nil
				})).WithPolling(time.Second).Should(gomega.Equal(value))
			}

			// Changing only the metrics avoids event-driven reconciliations caused by
			// HPA spec updates. Values below the target keep the workload at one Pod,
			// so Pod startup and stabilization do not enter the cadence measurement.
			ginkgo.By("measuring consecutive metric observations for each HPA")
			errs := make([]error, len(tests))
			var wg sync.WaitGroup
			for i, tc := range tests {
				wg.Go(func() {
					defer ginkgo.GinkgoRecover()
					name := "consumer-" + tc.name
					metricName := "sync_period_" + tc.name
					observe := func(ctx context.Context, value int64) error {
						if err := metricsController.SetMetricValue(ctx, metricName, value, nil); err != nil {
							return fmt.Errorf("set external metric %q to %d: %w", metricName, value, err)
						}
						if err := waitForMetric(ctx, name, metricName, value); err != nil {
							return fmt.Errorf("wait for HPA %q to observe metric %q=%d: %w", name, metricName, value, err)
						}
						return nil
					}

					// Establish a fresh observation after all workloads have been created;
					// their initial reconciliations may have happened during setup.
					bootstrapCtx, cancelBootstrap := context.WithTimeout(ctx, 2*time.Minute)
					err := observe(bootstrapCtx, 2)
					cancelBootstrap()
					if err != nil {
						errs[i] = err
						return
					}
					start := time.Now()
					measurementCtx, cancel := context.WithTimeout(ctx, tc.maxDuration)
					defer cancel()
					for sample := range samples {
						if err := observe(measurementCtx, int64(sample+3)); err != nil {
							errs[i] = err
							return
						}
					}
					elapsed := time.Since(start)
					framework.Logf("HPA %q observed %d metric changes in %s", name, samples, elapsed)
					if elapsed < tc.minDuration {
						errs[i] = fmt.Errorf("HPA %q observed %d metric changes in %s, expected at least %s", name, samples, elapsed, tc.minDuration)
					}
				})
			}
			wg.Wait()
			framework.ExpectNoError(errors.Join(errs...))
		})
	})
