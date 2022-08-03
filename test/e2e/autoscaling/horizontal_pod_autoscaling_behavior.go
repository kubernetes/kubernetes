/*
Copyright 2022 The Kubernetes Authors.

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
	"time"

	"k8s.io/kubernetes/test/e2e/framework"
	e2eautoscaling "k8s.io/kubernetes/test/e2e/framework/autoscaling"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("[Feature:HPA] [Serial] [Slow] Horizontal pod autoscaling (non-default behavior)", func() {
	f := framework.NewDefaultFramework("horizontal-pod-autoscaling")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	fullWindowOfNewUsage := 30 * time.Second
	windowWithOldUsagePasses := 30 * time.Second
	newPodMetricsDelay := 15 * time.Second
	metricsAvailableDelay := fullWindowOfNewUsage + windowWithOldUsagePasses + newPodMetricsDelay

	hpaReconciliationInterval := 15 * time.Second
	actuationDelay := 10 * time.Second
	maxHPAReactionTime := metricsAvailableDelay + hpaReconciliationInterval + actuationDelay

	maxConsumeCPUDelay := 30 * time.Second
	waitForReplicasPollInterval := 20 * time.Second
	maxResourceConsumerDelay := maxConsumeCPUDelay + waitForReplicasPollInterval

	waitBuffer := 1 * time.Minute

	ginkgo.Describe("with short downscale stabilization window", func() {
		ginkgo.It("should scale down soon after the stabilization period", func() {
			ginkgo.By("setting up resource consumer and HPA")
			podCPURequest := 500
			targetCPUUtilizationPercent := 25
			usageForSingleReplica := 110
			initPods := 1
			initCPUUsageTotal := initPods * usageForSingleReplica
			upScaleStabilization := 0 * time.Minute
			downScaleStabilization := 1 * time.Minute

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				"consumer", f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
				initCPUUsageTotal, 0, 0, int64(podCPURequest), 200,
				f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			)
			defer rc.CleanUp()

			hpa := e2eautoscaling.CreateCPUHorizontalPodAutoscalerWithBehavior(
				rc, int32(targetCPUUtilizationPercent), 1, 5,
				e2eautoscaling.HPABehaviorWithStabilizationWindows(upScaleStabilization, downScaleStabilization),
			)
			defer e2eautoscaling.DeleteHPAWithBehavior(rc, hpa.Name)

			// making sure HPA is ready, doing its job and already has a recommendation recorded
			// for stabilization logic before lowering the consumption
			ginkgo.By("triggering scale up to record a recommendation")
			rc.ConsumeCPU(3 * usageForSingleReplica)
			rc.WaitForReplicas(3, maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)

			ginkgo.By("triggering scale down by lowering consumption")
			rc.ConsumeCPU(2 * usageForSingleReplica)
			waitStart := time.Now()
			rc.WaitForReplicas(2, downScaleStabilization+maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)
			timeWaited := time.Now().Sub(waitStart)

			ginkgo.By("verifying time waited for a scale down")
			framework.Logf("time waited for scale down: %s", timeWaited)
			framework.ExpectEqual(timeWaited > downScaleStabilization, true, "waited %s, wanted more than %s", timeWaited, downScaleStabilization)
			deadline := downScaleStabilization + maxHPAReactionTime + maxResourceConsumerDelay
			framework.ExpectEqual(timeWaited < deadline, true, "waited %s, wanted less than %s", timeWaited, deadline)
		})
	})

	ginkgo.Describe("with long upscale stabilization window", func() {
		ginkgo.It("should scale up only after the stabilization period", func() {
			ginkgo.By("setting up resource consumer and HPA")
			podCPURequest := 500
			targetCPUUtilizationPercent := 25
			usageForSingleReplica := 110
			initPods := 2
			initCPUUsageTotal := initPods * usageForSingleReplica
			upScaleStabilization := 3 * time.Minute
			downScaleStabilization := 0 * time.Minute

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				"consumer", f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
				initCPUUsageTotal, 0, 0, int64(podCPURequest), 200,
				f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			)
			defer rc.CleanUp()

			hpa := e2eautoscaling.CreateCPUHorizontalPodAutoscalerWithBehavior(
				rc, int32(targetCPUUtilizationPercent), 1, 10,
				e2eautoscaling.HPABehaviorWithStabilizationWindows(upScaleStabilization, downScaleStabilization),
			)
			defer e2eautoscaling.DeleteHPAWithBehavior(rc, hpa.Name)

			// making sure HPA is ready, doing its job and already has a recommendation recorded
			// for stabilization logic before increasing the consumption
			ginkgo.By("triggering scale down to record a recommendation")
			rc.ConsumeCPU(1 * usageForSingleReplica)
			rc.WaitForReplicas(1, maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)

			ginkgo.By("triggering scale up by increasing consumption")
			rc.ConsumeCPU(3 * usageForSingleReplica)
			waitStart := time.Now()
			rc.WaitForReplicas(3, upScaleStabilization+maxHPAReactionTime+maxResourceConsumerDelay+waitBuffer)
			timeWaited := time.Now().Sub(waitStart)

			ginkgo.By("verifying time waited for a scale up")
			framework.Logf("time waited for scale up: %s", timeWaited)
			framework.ExpectEqual(timeWaited > upScaleStabilization, true, "waited %s, wanted more than %s", timeWaited, upScaleStabilization)
			deadline := upScaleStabilization + maxHPAReactionTime + maxResourceConsumerDelay
			framework.ExpectEqual(timeWaited < deadline, true, "waited %s, wanted less than %s", timeWaited, deadline)
		})
	})

	ginkgo.Describe("with upscale autoscaling disabled", func() {
		ginkgo.It("shouldn't scale up", func() {
			ginkgo.By("setting up resource consumer and HPA")
			podCPURequest := 500
			targetCPUUtilizationPercent := 25
			usageForSingleReplica := 110
			initPods := 1
			initCPUUsageTotal := initPods * usageForSingleReplica

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				"consumer", f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
				initCPUUsageTotal, 0, 0, int64(podCPURequest), 200,
				f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			)
			defer rc.CleanUp()

			hpa := e2eautoscaling.CreateCPUHorizontalPodAutoscalerWithBehavior(
				rc, int32(targetCPUUtilizationPercent), 1, 10, e2eautoscaling.HPABehaviorWithScaleUpDisabled(),
			)
			defer e2eautoscaling.DeleteHPAWithBehavior(rc, hpa.Name)

			waitDeadline := maxHPAReactionTime + maxResourceConsumerDelay + waitBuffer

			ginkgo.By("trying to trigger scale up")
			rc.ConsumeCPU(8 * usageForSingleReplica)
			waitStart := time.Now()

			rc.EnsureDesiredReplicasInRange(initPods, initPods, waitDeadline, hpa.Name)
			timeWaited := time.Now().Sub(waitStart)

			ginkgo.By("verifying time waited for a scale up")
			framework.Logf("time waited for scale up: %s", timeWaited)
			framework.ExpectEqual(timeWaited > waitDeadline, true, "waited %s, wanted to wait more than %s", timeWaited, waitDeadline)

			ginkgo.By("verifying number of replicas")
			replicas := rc.GetReplicas()
			framework.ExpectEqual(replicas == initPods, true, "had %s replicas, still have %s replicas after time deadline", initPods, replicas)
		})
	})

	ginkgo.Describe("with downscale autoscaling disabled", func() {
		ginkgo.It("shouldn't scale down", func() {
			ginkgo.By("setting up resource consumer and HPA")
			podCPURequest := 500
			targetCPUUtilizationPercent := 25
			usageForSingleReplica := 110
			initPods := 3
			initCPUUsageTotal := initPods * usageForSingleReplica

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				"consumer", f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
				initCPUUsageTotal, 0, 0, int64(podCPURequest), 200,
				f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			)
			defer rc.CleanUp()

			hpa := e2eautoscaling.CreateCPUHorizontalPodAutoscalerWithBehavior(
				rc, int32(targetCPUUtilizationPercent), 1, 10, e2eautoscaling.HPABehaviorWithScaleDownDisabled(),
			)
			defer e2eautoscaling.DeleteHPAWithBehavior(rc, hpa.Name)

			defaultDownscaleStabilisation := 5 * time.Minute
			waitDeadline := maxHPAReactionTime + maxResourceConsumerDelay + defaultDownscaleStabilisation

			ginkgo.By("trying to trigger scale down")
			rc.ConsumeCPU(1 * usageForSingleReplica)
			waitStart := time.Now()

			rc.EnsureDesiredReplicasInRange(initPods, initPods, waitDeadline, hpa.Name)
			timeWaited := time.Now().Sub(waitStart)

			ginkgo.By("verifying time waited for a scale down")
			framework.Logf("time waited for scale down: %s", timeWaited)
			framework.ExpectEqual(timeWaited > waitDeadline, true, "waited %s, wanted to wait more than %s", timeWaited, waitDeadline)

			ginkgo.By("verifying number of replicas")
			replicas := rc.GetReplicas()
			framework.ExpectEqual(replicas == initPods, true, "had %s replicas, still have %s replicas after time deadline", initPods, replicas)
		})
	})

})
