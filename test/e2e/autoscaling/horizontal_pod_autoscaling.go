/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/pod-security-admission/api"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eautoscaling "k8s.io/kubernetes/test/e2e/framework/autoscaling"

	"github.com/onsi/ginkgo/v2"
)

// These tests don't seem to be running properly in parallel: issue: #20338.
var _ = SIGDescribe("[Feature:HPA] Horizontal pod autoscaling (scale resource: CPU)", func() {
	f := framework.NewDefaultFramework("horizontal-pod-autoscaling")
	f.NamespacePodSecurityEnforceLevel = api.LevelBaseline

	titleUp := "Should scale from 1 pod to 3 pods and from 3 to 5"
	titleDown := "Should scale from 5 pods to 3 pods and from 3 to 1"

	ginkgo.Describe("[Serial] [Slow] Deployment", func() {
		// CPU tests via deployments
		ginkgo.It(titleUp, func() {
			scaleUp("test-deployment", e2eautoscaling.KindDeployment, false, f)
		})
		ginkgo.It(titleDown, func() {
			scaleDown("test-deployment", e2eautoscaling.KindDeployment, false, f)
		})
	})

	ginkgo.Describe("[Serial] [Slow] ReplicaSet", func() {
		// CPU tests via ReplicaSets
		ginkgo.It(titleUp, func() {
			scaleUp("rs", e2eautoscaling.KindReplicaSet, false, f)
		})
		ginkgo.It(titleDown, func() {
			scaleDown("rs", e2eautoscaling.KindReplicaSet, false, f)
		})
	})

	// These tests take ~20 minutes each.
	ginkgo.Describe("[Serial] [Slow] ReplicationController", func() {
		// CPU tests via replication controllers
		ginkgo.It(titleUp+" and verify decision stability", func() {
			scaleUp("rc", e2eautoscaling.KindRC, true, f)
		})
		ginkgo.It(titleDown+" and verify decision stability", func() {
			scaleDown("rc", e2eautoscaling.KindRC, true, f)
		})
	})

	ginkgo.Describe("ReplicationController light", func() {
		ginkgo.It("Should scale from 1 pod to 2 pods", func() {
			scaleTest := &HPAScaleTest{
				initPods:                    1,
				totalInitialCPUUsage:        150,
				perPodCPURequest:            200,
				targetCPUUtilizationPercent: 50,
				minPods:                     1,
				maxPods:                     2,
				firstScale:                  2,
			}
			scaleTest.run("rc-light", e2eautoscaling.KindRC, f)
		})
		ginkgo.It("Should scale from 2 pods to 1 pod [Slow]", func() {
			scaleTest := &HPAScaleTest{
				initPods:                    2,
				totalInitialCPUUsage:        50,
				perPodCPURequest:            200,
				targetCPUUtilizationPercent: 50,
				minPods:                     1,
				maxPods:                     2,
				firstScale:                  1,
			}
			scaleTest.run("rc-light", e2eautoscaling.KindRC, f)
		})
	})

	ginkgo.Describe("[Serial] [Slow] ReplicaSet with idle sidecar (ContainerResource use case)", func() {
		// ContainerResource CPU autoscaling on idle sidecar
		ginkgo.It(titleUp+" on a busy application with an idle sidecar container", func() {
			scaleOnIdleSideCar("rs", e2eautoscaling.KindReplicaSet, false, f)
		})

		// ContainerResource CPU autoscaling on busy sidecar
		ginkgo.It("Should not scale up on a busy sidecar with an idle application", func() {
			doNotScaleOnBusySidecar("rs", e2eautoscaling.KindReplicaSet, true, f)
		})
	})

	ginkgo.Describe("CustomResourceDefinition", func() {
		ginkgo.It("Should scale with a CRD targetRef", func() {
			scaleTest := &HPAScaleTest{
				initPods:                    1,
				totalInitialCPUUsage:        150,
				perPodCPURequest:            200,
				targetCPUUtilizationPercent: 50,
				minPods:                     1,
				maxPods:                     2,
				firstScale:                  2,
				targetRef:                   e2eautoscaling.CustomCRDTargetRef(),
			}
			scaleTest.run("crd-light", e2eautoscaling.KindCRD, f)
		})
	})
})

// HPAScaleTest struct is used by the scale(...) function.
type HPAScaleTest struct {
	initPods                    int
	totalInitialCPUUsage        int
	perPodCPURequest            int64
	targetCPUUtilizationPercent int32
	minPods                     int32
	maxPods                     int32
	firstScale                  int
	firstScaleStasis            time.Duration
	cpuBurst                    int
	secondScale                 int32
	targetRef                   autoscalingv2.CrossVersionObjectReference
}

// run is a method which runs an HPA lifecycle, from a starting state, to an expected
// The initial state is defined by the initPods parameter.
// The first state change is due to the CPU being consumed initially, which HPA responds to by changing pod counts.
// The second state change (optional) is due to the CPU burst parameter, which HPA again responds to.
// TODO The use of 3 states is arbitrary, we could eventually make this test handle "n" states once this test stabilizes.
func (scaleTest *HPAScaleTest) run(name string, kind schema.GroupVersionKind, f *framework.Framework) {
	const timeToWait = 15 * time.Minute
	rc := e2eautoscaling.NewDynamicResourceConsumer(name, f.Namespace.Name, kind, scaleTest.initPods, scaleTest.totalInitialCPUUsage, 0, 0, scaleTest.perPodCPURequest, 200, f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle)
	defer rc.CleanUp()
	var hpa *autoscalingv2.HorizontalPodAutoscaler
	hpa = e2eautoscaling.CreateCPUHorizontalPodAutoscaler(rc, scaleTest.targetCPUUtilizationPercent, scaleTest.minPods, scaleTest.maxPods)
	defer e2eautoscaling.DeleteHorizontalPodAutoscaler(rc, hpa.Name)

	rc.WaitForReplicas(scaleTest.firstScale, timeToWait)
	if scaleTest.firstScaleStasis > 0 {
		rc.EnsureDesiredReplicasInRange(scaleTest.firstScale, scaleTest.firstScale+1, scaleTest.firstScaleStasis, hpa.Name)
	}
	if scaleTest.cpuBurst > 0 && scaleTest.secondScale > 0 {
		rc.ConsumeCPU(scaleTest.cpuBurst)
		rc.WaitForReplicas(int(scaleTest.secondScale), timeToWait)
	}
}

func scaleUp(name string, kind schema.GroupVersionKind, checkStability bool, f *framework.Framework) {
	stasis := 0 * time.Minute
	if checkStability {
		stasis = 10 * time.Minute
	}
	scaleTest := &HPAScaleTest{
		initPods:                    1,
		totalInitialCPUUsage:        250,
		perPodCPURequest:            500,
		targetCPUUtilizationPercent: 20,
		minPods:                     1,
		maxPods:                     5,
		firstScale:                  3,
		firstScaleStasis:            stasis,
		cpuBurst:                    700,
		secondScale:                 5,
	}
	scaleTest.run(name, kind, f)
}

func scaleDown(name string, kind schema.GroupVersionKind, checkStability bool, f *framework.Framework) {
	stasis := 0 * time.Minute
	if checkStability {
		stasis = 10 * time.Minute
	}
	scaleTest := &HPAScaleTest{
		initPods:                    5,
		totalInitialCPUUsage:        325,
		perPodCPURequest:            500,
		targetCPUUtilizationPercent: 30,
		minPods:                     1,
		maxPods:                     5,
		firstScale:                  3,
		firstScaleStasis:            stasis,
		cpuBurst:                    10,
		secondScale:                 1,
	}
	scaleTest.run(name, kind, f)
}

type HPAContainerResourceScaleTest struct {
	initPods                    int
	totalInitialCPUUsage        int
	perContainerCPURequest      int64
	targetCPUUtilizationPercent int32
	minPods                     int32
	maxPods                     int32
	noScale                     bool
	noScaleStasis               time.Duration
	firstScale                  int
	firstScaleStasis            time.Duration
	cpuBurst                    int
	secondScale                 int32
	sidecarStatus               e2eautoscaling.SidecarStatusType
	sidecarType                 e2eautoscaling.SidecarWorkloadType
}

func (scaleTest *HPAContainerResourceScaleTest) run(name string, kind schema.GroupVersionKind, f *framework.Framework) {
	const timeToWait = 15 * time.Minute
	rc := e2eautoscaling.NewDynamicResourceConsumer(name, f.Namespace.Name, kind, scaleTest.initPods, scaleTest.totalInitialCPUUsage, 0, 0, scaleTest.perContainerCPURequest, 200, f.ClientSet, f.ScalesGetter, scaleTest.sidecarStatus, scaleTest.sidecarType)
	defer rc.CleanUp()
	hpa := e2eautoscaling.CreateContainerResourceCPUHorizontalPodAutoscaler(rc, scaleTest.targetCPUUtilizationPercent, scaleTest.minPods, scaleTest.maxPods)
	defer e2eautoscaling.DeleteContainerResourceHPA(rc, hpa.Name)

	if scaleTest.noScale {
		if scaleTest.noScaleStasis > 0 {
			rc.EnsureDesiredReplicasInRange(scaleTest.initPods, scaleTest.initPods, scaleTest.noScaleStasis, hpa.Name)
		}
	} else {
		rc.WaitForReplicas(scaleTest.firstScale, timeToWait)
		if scaleTest.firstScaleStasis > 0 {
			rc.EnsureDesiredReplicasInRange(scaleTest.firstScale, scaleTest.firstScale+1, scaleTest.firstScaleStasis, hpa.Name)
		}
		if scaleTest.cpuBurst > 0 && scaleTest.secondScale > 0 {
			rc.ConsumeCPU(scaleTest.cpuBurst)
			rc.WaitForReplicas(int(scaleTest.secondScale), timeToWait)
		}
	}
}

func scaleOnIdleSideCar(name string, kind schema.GroupVersionKind, checkStability bool, f *framework.Framework) {
	// Scale up on a busy application with an idle sidecar container
	stasis := 0 * time.Minute
	if checkStability {
		stasis = 10 * time.Minute
	}
	scaleTest := &HPAContainerResourceScaleTest{
		initPods:                    1,
		totalInitialCPUUsage:        125,
		perContainerCPURequest:      250,
		targetCPUUtilizationPercent: 20,
		minPods:                     1,
		maxPods:                     5,
		firstScale:                  3,
		firstScaleStasis:            stasis,
		cpuBurst:                    500,
		secondScale:                 5,
		sidecarStatus:               e2eautoscaling.Enable,
		sidecarType:                 e2eautoscaling.Idle,
	}
	scaleTest.run(name, kind, f)
}

func doNotScaleOnBusySidecar(name string, kind schema.GroupVersionKind, checkStability bool, f *framework.Framework) {
	// Do not scale up on a busy sidecar with an idle application
	stasis := 0 * time.Minute
	if checkStability {
		stasis = 1 * time.Minute
	}
	scaleTest := &HPAContainerResourceScaleTest{
		initPods:                    1,
		totalInitialCPUUsage:        250,
		perContainerCPURequest:      500,
		targetCPUUtilizationPercent: 20,
		minPods:                     1,
		maxPods:                     5,
		cpuBurst:                    700,
		sidecarStatus:               e2eautoscaling.Enable,
		sidecarType:                 e2eautoscaling.Busy,
		noScale:                     true,
		noScaleStasis:               stasis,
	}
	scaleTest.run(name, kind, f)
}
