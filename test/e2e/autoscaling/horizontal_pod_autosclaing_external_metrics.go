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
})
