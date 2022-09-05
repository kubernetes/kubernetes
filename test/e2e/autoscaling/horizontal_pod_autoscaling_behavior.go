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

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eautoscaling "k8s.io/kubernetes/test/e2e/framework/autoscaling"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("[Feature:HPA] [Serial] [Slow] Horizontal pod autoscaling (non-default behavior)", func() {
	f := framework.NewDefaultFramework("horizontal-pod-autoscaling")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	hpaName := "consumer"

	podCPURequest := 500
	targetCPUUtilizationPercent := 25
	usageForSingleReplica := 110

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
			initPods := 1
			initCPUUsageTotal := initPods * usageForSingleReplica
			upScaleStabilization := 0 * time.Minute
			downScaleStabilization := 1 * time.Minute

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
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
			initPods := 2
			initCPUUsageTotal := initPods * usageForSingleReplica
			upScaleStabilization := 3 * time.Minute
			downScaleStabilization := 0 * time.Minute

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
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

	ginkgo.Describe("with autoscaling disabled", func() {
		ginkgo.It("shouldn't scale up", func() {
			ginkgo.By("setting up resource consumer and HPA")
			initPods := 1
			initCPUUsageTotal := initPods * usageForSingleReplica

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
				initCPUUsageTotal, 0, 0, int64(podCPURequest), 200,
				f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			)
			defer rc.CleanUp()

			hpa := e2eautoscaling.CreateCPUHorizontalPodAutoscalerWithBehavior(
				rc, int32(targetCPUUtilizationPercent), 1, 10, e2eautoscaling.HPABehaviorWithScaleDisabled(e2eautoscaling.ScaleUpDirection),
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

		ginkgo.It("shouldn't scale down", func() {
			ginkgo.By("setting up resource consumer and HPA")
			initPods := 3
			initCPUUsageTotal := initPods * usageForSingleReplica

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
				initCPUUsageTotal, 0, 0, int64(podCPURequest), 200,
				f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			)
			defer rc.CleanUp()

			hpa := e2eautoscaling.CreateCPUHorizontalPodAutoscalerWithBehavior(
				rc, int32(targetCPUUtilizationPercent), 1, 10, e2eautoscaling.HPABehaviorWithScaleDisabled(e2eautoscaling.ScaleDownDirection),
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

	ginkgo.Describe("with scale limited by number of Pods rate", func() {
		ginkgo.It("should scale up no more than given number of Pods per minute", func() {
			ginkgo.By("setting up resource consumer and HPA")
			initPods := 1
			initCPUUsageTotal := initPods * usageForSingleReplica
			limitWindowLength := 1 * time.Minute
			podsLimitPerMinute := 2

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
				initCPUUsageTotal, 0, 0, int64(podCPURequest), 200,
				f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			)
			defer rc.CleanUp()

			hpa := e2eautoscaling.CreateCPUHorizontalPodAutoscalerWithBehavior(
				rc, int32(targetCPUUtilizationPercent), 1, 10,
				e2eautoscaling.HPABehaviorWithScaleLimitedByNumberOfPods(e2eautoscaling.ScaleUpDirection, int32(podsLimitPerMinute), int32(limitWindowLength.Seconds())),
			)
			defer e2eautoscaling.DeleteHPAWithBehavior(rc, hpa.Name)

			ginkgo.By("triggering scale up by increasing consumption")
			rc.ConsumeCPU(5 * usageForSingleReplica)

			waitStart := time.Now()
			rc.WaitForReplicas(3, maxHPAReactionTime+maxResourceConsumerDelay+limitWindowLength)
			timeWaitedFor3 := time.Now().Sub(waitStart)

			waitStart = time.Now()
			rc.WaitForReplicas(5, maxHPAReactionTime+maxResourceConsumerDelay+limitWindowLength)
			timeWaitedFor5 := time.Now().Sub(waitStart)

			ginkgo.By("verifying time waited for a scale up to 3 replicas")
			deadline := limitWindowLength + maxHPAReactionTime + maxResourceConsumerDelay
			// First scale event can happen right away, as there were no scale events in the past.
			framework.ExpectEqual(timeWaitedFor3 < deadline, true, "waited %s, wanted less than %s", timeWaitedFor3, deadline)

			ginkgo.By("verifying time waited for a scale up to 5 replicas")
			// Second scale event needs to respect limit window.
			framework.ExpectEqual(timeWaitedFor5 > limitWindowLength, true, "waited %s, wanted to wait more than %s", timeWaitedFor5, limitWindowLength)
			framework.ExpectEqual(timeWaitedFor5 < deadline, true, "waited %s, wanted less than %s", timeWaitedFor5, deadline)
		})

		ginkgo.It("should scale down no more than given number of Pods per minute", func() {
			ginkgo.By("setting up resource consumer and HPA")
			initPods := 6
			initCPUUsageTotal := initPods * usageForSingleReplica
			limitWindowLength := 1 * time.Minute
			podsLimitPerMinute := 2

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
				initCPUUsageTotal, 0, 0, int64(podCPURequest), 200,
				f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			)
			defer rc.CleanUp()

			hpa := e2eautoscaling.CreateCPUHorizontalPodAutoscalerWithBehavior(
				rc, int32(targetCPUUtilizationPercent), 1, 10,
				e2eautoscaling.HPABehaviorWithScaleLimitedByNumberOfPods(e2eautoscaling.ScaleDownDirection, int32(podsLimitPerMinute), int32(limitWindowLength.Seconds())),
			)
			defer e2eautoscaling.DeleteHPAWithBehavior(rc, hpa.Name)

			ginkgo.By("triggering scale down by lowering consumption")
			rc.ConsumeCPU(1 * usageForSingleReplica)

			waitStart := time.Now()
			rc.WaitForReplicas(4, maxHPAReactionTime+maxResourceConsumerDelay+limitWindowLength)
			timeWaitedFor4 := time.Now().Sub(waitStart)

			waitStart = time.Now()
			rc.WaitForReplicas(2, maxHPAReactionTime+maxResourceConsumerDelay+limitWindowLength)
			timeWaitedFor2 := time.Now().Sub(waitStart)

			ginkgo.By("verifying time waited for a scale down to 4 replicas")
			deadline := limitWindowLength + maxHPAReactionTime + maxResourceConsumerDelay
			// First scale event can happen right away, as there were no scale events in the past.
			framework.ExpectEqual(timeWaitedFor4 < deadline, true, "waited %s, wanted less than %s", timeWaitedFor4, deadline)

			ginkgo.By("verifying time waited for a scale down to 2 replicas")
			// Second scale event needs to respect limit window.
			framework.ExpectEqual(timeWaitedFor2 > limitWindowLength, true, "waited %s, wanted more than %s", timeWaitedFor2, limitWindowLength)
			framework.ExpectEqual(timeWaitedFor2 < deadline, true, "waited %s, wanted less than %s", timeWaitedFor2, deadline)
		})
	})

	ginkgo.Describe("with scale limited by percentage", func() {
		ginkgo.It("should scale up no more than given percentage of current Pods per minute", func() {
			ginkgo.By("setting up resource consumer and HPA")
			initPods := 4
			initCPUUsageTotal := initPods * usageForSingleReplica
			limitWindowLength := 1 * time.Minute
			percentageLimitPerMinute := 50

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
				initCPUUsageTotal, 0, 0, int64(podCPURequest), 200,
				f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			)
			defer rc.CleanUp()

			hpa := e2eautoscaling.CreateCPUHorizontalPodAutoscalerWithBehavior(
				rc, int32(targetCPUUtilizationPercent), 1, 10,
				e2eautoscaling.HPABehaviorWithScaleLimitedByPercentage(e2eautoscaling.ScaleUpDirection, int32(percentageLimitPerMinute), int32(limitWindowLength.Seconds())),
			)
			defer e2eautoscaling.DeleteHPAWithBehavior(rc, hpa.Name)

			ginkgo.By("triggering scale up by increasing consumption")
			rc.ConsumeCPU(10 * usageForSingleReplica)

			waitStart := time.Now()
			rc.WaitForReplicas(6, maxHPAReactionTime+maxResourceConsumerDelay+limitWindowLength)
			timeWaitedFor6 := time.Now().Sub(waitStart)

			waitStart = time.Now()
			rc.WaitForReplicas(9, maxHPAReactionTime+maxResourceConsumerDelay+limitWindowLength)
			timeWaitedFor9 := time.Now().Sub(waitStart)

			ginkgo.By("verifying time waited for a scale up to 6 replicas")
			deadline := limitWindowLength + maxHPAReactionTime + maxResourceConsumerDelay
			// First scale event can happen right away, as there were no scale events in the past.
			framework.ExpectEqual(timeWaitedFor6 < deadline, true, "waited %s, wanted less than %s", timeWaitedFor6, deadline)

			ginkgo.By("verifying time waited for a scale up to 9 replicas")
			// Second scale event needs to respect limit window.
			framework.ExpectEqual(timeWaitedFor9 > limitWindowLength, true, "waited %s, wanted to wait more than %s", timeWaitedFor9, limitWindowLength)
			framework.ExpectEqual(timeWaitedFor9 < deadline, true, "waited %s, wanted less than %s", timeWaitedFor9, deadline)
		})

		ginkgo.It("should scale down no more than given percentage of current Pods per minute", func() {
			ginkgo.By("setting up resource consumer and HPA")
			initPods := 8
			initCPUUsageTotal := initPods * usageForSingleReplica
			limitWindowLength := 1 * time.Minute
			percentageLimitPerMinute := 50

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
				initCPUUsageTotal, 0, 0, int64(podCPURequest), 200,
				f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			)
			defer rc.CleanUp()

			hpa := e2eautoscaling.CreateCPUHorizontalPodAutoscalerWithBehavior(
				rc, int32(targetCPUUtilizationPercent), 1, 10,
				e2eautoscaling.HPABehaviorWithScaleLimitedByPercentage(e2eautoscaling.ScaleDownDirection, int32(percentageLimitPerMinute), int32(limitWindowLength.Seconds())),
			)
			defer e2eautoscaling.DeleteHPAWithBehavior(rc, hpa.Name)

			ginkgo.By("triggering scale down by lowering consumption")
			rc.ConsumeCPU(1 * usageForSingleReplica)

			waitStart := time.Now()
			rc.WaitForReplicas(4, maxHPAReactionTime+maxResourceConsumerDelay+limitWindowLength)
			timeWaitedFor4 := time.Now().Sub(waitStart)

			waitStart = time.Now()
			rc.WaitForReplicas(2, maxHPAReactionTime+maxResourceConsumerDelay+limitWindowLength)
			timeWaitedFor2 := time.Now().Sub(waitStart)

			ginkgo.By("verifying time waited for a scale down to 4 replicas")
			deadline := limitWindowLength + maxHPAReactionTime + maxResourceConsumerDelay
			// First scale event can happen right away, as there were no scale events in the past.
			framework.ExpectEqual(timeWaitedFor4 < deadline, true, "waited %s, wanted less than %s", timeWaitedFor4, deadline)

			ginkgo.By("verifying time waited for a scale down to 2 replicas")
			// Second scale event needs to respect limit window.
			framework.ExpectEqual(timeWaitedFor2 > limitWindowLength, true, "waited %s, wanted more than %s", timeWaitedFor2, limitWindowLength)
			framework.ExpectEqual(timeWaitedFor2 < deadline, true, "waited %s, wanted less than %s", timeWaitedFor2, deadline)
		})
	})

	ginkgo.Describe("with both scale up and down controls configured", func() {
		ginkgo.It("should keep recommendation within the range over two stabilization windows", func() {
			ginkgo.By("setting up resource consumer and HPA")
			initPods := 2
			initCPUUsageTotal := initPods * usageForSingleReplica
			upScaleStabilization := 3 * time.Minute
			downScaleStabilization := 3 * time.Minute

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
				initCPUUsageTotal, 0, 0, int64(podCPURequest), 200,
				f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			)
			defer rc.CleanUp()

			hpa := e2eautoscaling.CreateCPUHorizontalPodAutoscalerWithBehavior(
				rc, int32(targetCPUUtilizationPercent), 2, 10,
				e2eautoscaling.HPABehaviorWithStabilizationWindows(upScaleStabilization, downScaleStabilization),
			)
			defer e2eautoscaling.DeleteHPAWithBehavior(rc, hpa.Name)

			ginkgo.By("triggering scale up by increasing consumption")
			rc.ConsumeCPU(5 * usageForSingleReplica)
			waitDeadline := upScaleStabilization

			ginkgo.By("verifying number of replicas stay in desired range within stabilisation window")
			rc.EnsureDesiredReplicasInRange(2, 2, waitDeadline, hpa.Name)

			ginkgo.By("waiting for replicas to scale up after stabilisation window passed")
			waitStart := time.Now()
			waitDeadline = maxHPAReactionTime + maxResourceConsumerDelay + waitBuffer
			rc.WaitForReplicas(4, waitDeadline)
			timeWaited := time.Now().Sub(waitStart)
			framework.Logf("time waited for scale up: %s", timeWaited)
			framework.ExpectEqual(timeWaited < waitDeadline, true, "waited %s, wanted less than %s", timeWaited, waitDeadline)

			ginkgo.By("triggering scale down by lowering consumption")
			rc.ConsumeCPU(2 * usageForSingleReplica)
			waitDeadline = downScaleStabilization

			ginkgo.By("verifying number of replicas stay in desired range within stabilisation window")
			rc.EnsureDesiredReplicasInRange(4, 4, waitDeadline, hpa.Name)

			ginkgo.By("waiting for replicas to scale down after stabilisation window passed")
			waitStart = time.Now()
			waitDeadline = maxHPAReactionTime + maxResourceConsumerDelay + waitBuffer
			rc.WaitForReplicas(2, waitDeadline)
			timeWaited = time.Now().Sub(waitStart)
			framework.Logf("time waited for scale down: %s", timeWaited)
			framework.ExpectEqual(timeWaited < waitDeadline, true, "waited %s, wanted less than %s", timeWaited, waitDeadline)
		})

		ginkgo.It("should keep recommendation within the range with stabilization window and pod limit rate", func() {
			ginkgo.By("setting up resource consumer and HPA")
			initPods := 2
			initCPUUsageTotal := initPods * usageForSingleReplica
			downScaleStabilization := 3 * time.Minute
			limitWindowLength := 2 * time.Minute
			podsLimitPerMinute := 1

			rc := e2eautoscaling.NewDynamicResourceConsumer(
				hpaName, f.Namespace.Name, e2eautoscaling.KindDeployment, initPods,
				initCPUUsageTotal, 0, 0, int64(podCPURequest), 200,
				f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle,
			)
			defer rc.CleanUp()

			scaleUpRule := e2eautoscaling.HPAScalingRuleWithScalingPolicy(autoscalingv2.PodsScalingPolicy, int32(podsLimitPerMinute), int32(limitWindowLength.Seconds()))
			scaleDownRule := e2eautoscaling.HPAScalingRuleWithStabilizationWindow(int32(downScaleStabilization.Seconds()))
			hpa := e2eautoscaling.CreateCPUHorizontalPodAutoscalerWithBehavior(
				rc, int32(targetCPUUtilizationPercent), 2, 10,
				e2eautoscaling.HPABehaviorWithScaleUpAndDownRules(scaleUpRule, scaleDownRule),
			)
			defer e2eautoscaling.DeleteHPAWithBehavior(rc, hpa.Name)

			ginkgo.By("triggering scale up by increasing consumption")
			rc.ConsumeCPU(4 * usageForSingleReplica)
			waitDeadline := limitWindowLength

			ginkgo.By("verifying number of replicas stay in desired range with pod limit rate")
			rc.EnsureDesiredReplicasInRange(2, 3, waitDeadline, hpa.Name)

			ginkgo.By("waiting for replicas to scale up")
			waitStart := time.Now()
			waitDeadline = limitWindowLength + maxHPAReactionTime + maxResourceConsumerDelay + waitBuffer
			rc.WaitForReplicas(4, waitDeadline)
			timeWaited := time.Now().Sub(waitStart)
			framework.Logf("time waited for scale up: %s", timeWaited)
			framework.ExpectEqual(timeWaited < waitDeadline, true, "waited %s, wanted less than %s", timeWaited, waitDeadline)

			ginkgo.By("triggering scale down by lowering consumption")
			rc.ConsumeCPU(2 * usageForSingleReplica)

			ginkgo.By("verifying number of replicas stay in desired range within stabilisation window")
			waitDeadline = downScaleStabilization
			rc.EnsureDesiredReplicasInRange(4, 4, waitDeadline, hpa.Name)

			ginkgo.By("waiting for replicas to scale down after stabilisation window passed")
			waitStart = time.Now()
			waitDeadline = maxHPAReactionTime + maxResourceConsumerDelay + waitBuffer
			rc.WaitForReplicas(2, waitDeadline)
			timeWaited = time.Now().Sub(waitStart)
			framework.Logf("time waited for scale down: %s", timeWaited)
			framework.ExpectEqual(timeWaited < waitDeadline, true, "waited %s, wanted less than %s", timeWaited, waitDeadline)
		})
	})
})
