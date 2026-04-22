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
	"k8s.io/apimachinery/pkg/types"
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
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for HPA to scale down to min replicas")
		rc.WaitForReplicas(ctx, int(*hpa.Spec.MinReplicas), maxResourceConsumerDelay+waitBuffer)
	})

	// Regression coverage for the per-pod metric tolerance path. During a rolling
	// update, status.replicas transiently exceeds spec.replicas by maxSurge. If the
	// per-pod metric aggregate scales with pod count (so the usage ratio stays within
	// tolerance at the surged replica count), the HPA must leave spec.replicas alone
	// instead of latching it to status.replicas.
	ginkgo.It("should not drift spec.replicas during a rolling update when a per-pod external metric stays within tolerance", func(ctx context.Context) {
		initPods := 3
		perPodTarget := int64(100)
		steadyMetricValue := int64(initPods) * perPodTarget       // ratio 1.0 at initPods pods
		surgedMetricValue := (int64(initPods) + 1) * perPodTarget // ratio 1.0 while status.replicas = initPods+1

		ginkgo.By("Creating the resource consumer deployment")
		rc = e2eautoscaling.NewDynamicResourceConsumer(ctx,
			hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
			0, 0, 0,
			int64(podCPURequest), 200,
			f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			nil)
		ginkgo.DeferCleanup(rc.CleanUp)
		rc.WaitForReplicas(ctx, initPods, maxResourceConsumerDelay+waitBuffer)

		deployments := f.ClientSet.AppsV1().Deployments(f.Namespace.Name)

		// minReadySeconds holds the surge pod "unavailable" long enough to span
		// several HPA reconciles, so a buggy controller has multiple chances to
		// latch spec.replicas to status.replicas.
		ginkgo.By("Configuring the deployment for a prolonged surge (maxSurge=1, maxUnavailable=0, minReadySeconds=60s)")
		_, err := deployments.Patch(ctx, hpaName, types.StrategicMergePatchType,
			[]byte(`{"spec":{"minReadySeconds":60,"strategy":{"type":"RollingUpdate","rollingUpdate":{"maxSurge":1,"maxUnavailable":0}}}}`),
			metav1.PatchOptions{})
		framework.ExpectNoError(err)

		metricName := "per_pod_tolerance_drift"
		ginkgo.By(fmt.Sprintf("Creating external metric %q at %d (= %d replicas * %d target)", metricName, steadyMetricValue, initPods, perPodTarget))
		framework.ExpectNoError(metricsController.CreateMetric(ctx, metricName, steadyMetricValue, nil, false))

		ginkgo.By(fmt.Sprintf("Creating a per-pod external-metric HPA (AverageValue target=%d, min=%d, max=10)", perPodTarget, initPods))
		hpa := e2eautoscaling.CreateExternalHorizontalPodAutoscaler(ctx, rc, metricName, nil,
			v2.AverageValueMetricType, perPodTarget, int32(initPods), 10)
		ginkgo.DeferCleanup(e2eautoscaling.DeleteHorizontalPodAutoscaler, rc, hpa.Name)

		expectSpecReplicasStable := func(duration time.Duration) {
			stableErr := framework.Gomega().Consistently(ctx, func(ctx context.Context) (int32, error) {
				d, err := deployments.Get(ctx, hpaName, metav1.GetOptions{})
				if err != nil {
					return 0, err
				}
				if d.Spec.Replicas == nil {
					return 0, fmt.Errorf("deployment %s has nil spec.replicas", d.Name)
				}
				return *d.Spec.Replicas, nil
			}).WithTimeout(duration).WithPolling(10 * time.Second).Should(gomega.Equal(int32(initPods)))
			framework.ExpectNoErrorWithOffset(1, stableErr, "deployment spec.replicas drifted from %d within %v", initPods, duration)
		}

		ginkgo.By("Letting the HPA stabilize at the initial replicas with the within-tolerance metric")
		expectSpecReplicasStable(maxHPAReactionTime + maxResourceConsumerDelay)

		ginkgo.By("Triggering a rolling update by annotating the pod template")
		_, err = deployments.Patch(ctx, hpaName, types.StrategicMergePatchType,
			[]byte(`{"spec":{"template":{"metadata":{"annotations":{"e2e.hpa-drift/rollout":"1"}}}}}`),
			metav1.PatchOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for a surge pod to appear (deployment status.replicas > spec.replicas)")
		surgeErr := framework.Gomega().Eventually(ctx, func(ctx context.Context) error {
			d, err := deployments.Get(ctx, hpaName, metav1.GetOptions{})
			if err != nil {
				return err
			}
			if d.Spec.Replicas == nil {
				return fmt.Errorf("deployment %s has nil spec.replicas", d.Name)
			}
			if d.Status.Replicas <= *d.Spec.Replicas {
				return fmt.Errorf("no surge yet: status=%d spec=%d", d.Status.Replicas, *d.Spec.Replicas)
			}
			return nil
		}).WithTimeout(60 * time.Second).WithPolling(2 * time.Second).Should(gomega.Succeed())
		framework.ExpectNoError(surgeErr)

		ginkgo.By(fmt.Sprintf("Raising the external metric to %d so the HPA observes within-tolerance while status.replicas > spec.replicas", surgedMetricValue))
		framework.ExpectNoError(metricsController.SetMetricValue(ctx, metricName, surgedMetricValue, nil))

		// Cover at least four HPA reconciles (~15s each) while status.replicas > spec.replicas.
		// Without the fix, the very first reconcile in this window latches spec.replicas to status.replicas.
		ginkgo.By("Asserting spec.replicas does not drift upward across HPA reconciles during the surge")
		expectSpecReplicasStable(hpaReconciliationInterval*4 + actuationDelay)
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
