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

package e2e

import (
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

const (
	kindRC         = "replicationController"
	kindDeployment = "deployment"
	kindReplicaSet = "replicaset"
	subresource    = "scale"
)

// These tests don't seem to be running properly in parallel: issue: #20338.
//
var _ = framework.KubeDescribe("[HPA] Horizontal pod autoscaling (scale resource: CPU)", func() {
	var rc *ResourceConsumer
	f := framework.NewDefaultFramework("horizontal-pod-autoscaling")

	titleUp := "Should scale from 1 pod to 3 pods and from 3 to 5"
	titleDown := "Should scale from 5 pods to 3 pods and from 3 to 1"

	framework.KubeDescribe("[Serial] [Slow] Deployment", func() {
		// CPU tests via deployments
		It(titleUp, func() {
			scaleUp("test-deployment", kindDeployment, false, rc, f)
		})
		It(titleDown, func() {
			scaleDown("test-deployment", kindDeployment, false, rc, f)
		})
	})

	framework.KubeDescribe("[Serial] [Slow] ReplicaSet", func() {
		// CPU tests via deployments
		It(titleUp, func() {
			scaleUp("rs", kindReplicaSet, false, rc, f)
		})
		It(titleDown, func() {
			scaleDown("rs", kindReplicaSet, false, rc, f)
		})
	})

	// These tests take ~20 minutes each.
	framework.KubeDescribe("[Serial] [Slow] ReplicationController", func() {
		// CPU tests via replication controllers
		It(titleUp+" and verify decision stability", func() {
			scaleUp("rc", kindRC, true, rc, f)
		})
		It(titleDown+" and verify decision stability", func() {
			scaleDown("rc", kindRC, true, rc, f)
		})
	})

	framework.KubeDescribe("ReplicationController light", func() {
		It("Should scale from 1 pod to 2 pods", func() {
			scaleTest := &HPAScaleTest{
				initPods:                    1,
				totalInitialCPUUsage:        150,
				perPodCPURequest:            200,
				targetCPUUtilizationPercent: 50,
				minPods:                     1,
				maxPods:                     2,
				firstScale:                  2,
			}
			scaleTest.run("rc-light", kindRC, rc, f)
		})
		It("Should scale from 2 pods to 1 pod", func() {
			scaleTest := &HPAScaleTest{
				initPods:                    2,
				totalInitialCPUUsage:        50,
				perPodCPURequest:            200,
				targetCPUUtilizationPercent: 50,
				minPods:                     1,
				maxPods:                     2,
				firstScale:                  1,
			}
			scaleTest.run("rc-light", kindRC, rc, f)
		})
	})
})

// HPAScaleTest struct is used by the scale(...) function.
type HPAScaleTest struct {
	initPods                    int32
	totalInitialCPUUsage        int32
	perPodCPURequest            int64
	targetCPUUtilizationPercent int32
	minPods                     int32
	maxPods                     int32
	firstScale                  int32
	firstScaleStasis            time.Duration
	cpuBurst                    int
	secondScale                 int32
	secondScaleStasis           time.Duration
}

// run is a method which runs an HPA lifecycle, from a starting state, to an expected
// The initial state is defined by the initPods parameter.
// The first state change is due to the CPU being consumed initially, which HPA responds to by changing pod counts.
// The second state change (optional) is due to the CPU burst parameter, which HPA again responds to.
// TODO The use of 3 states is arbitrary, we could eventually make this test handle "n" states once this test stabilizes.
func (scaleTest *HPAScaleTest) run(name, kind string, rc *ResourceConsumer, f *framework.Framework) {
	rc = NewDynamicResourceConsumer(name, kind, int(scaleTest.initPods), int(scaleTest.totalInitialCPUUsage), 0, 0, scaleTest.perPodCPURequest, 200, f)
	defer rc.CleanUp()
	createCPUHorizontalPodAutoscaler(rc, scaleTest.targetCPUUtilizationPercent, scaleTest.minPods, scaleTest.maxPods)
	rc.WaitForReplicas(int(scaleTest.firstScale))
	if scaleTest.firstScaleStasis > 0 {
		rc.EnsureDesiredReplicas(int(scaleTest.firstScale), scaleTest.firstScaleStasis)
	}
	if scaleTest.cpuBurst > 0 && scaleTest.secondScale > 0 {
		rc.ConsumeCPU(scaleTest.cpuBurst)
		rc.WaitForReplicas(int(scaleTest.secondScale))
	}
}

func scaleUp(name, kind string, checkStability bool, rc *ResourceConsumer, f *framework.Framework) {
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
	scaleTest.run(name, kind, rc, f)
}

func scaleDown(name, kind string, checkStability bool, rc *ResourceConsumer, f *framework.Framework) {
	stasis := 0 * time.Minute
	if checkStability {
		stasis = 10 * time.Minute
	}
	scaleTest := &HPAScaleTest{
		initPods:                    5,
		totalInitialCPUUsage:        375,
		perPodCPURequest:            500,
		targetCPUUtilizationPercent: 30,
		minPods:                     1,
		maxPods:                     5,
		firstScale:                  3,
		firstScaleStasis:            stasis,
		cpuBurst:                    10,
		secondScale:                 1,
	}
	scaleTest.run(name, kind, rc, f)
}

func createCPUHorizontalPodAutoscaler(rc *ResourceConsumer, cpu, minReplicas, maxRepl int32) {
	hpa := &autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:      rc.name,
			Namespace: rc.framework.Namespace.Name,
		},
		Spec: autoscaling.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscaling.CrossVersionObjectReference{
				Kind: rc.kind,
				Name: rc.name,
			},
			MinReplicas:                    &minReplicas,
			MaxReplicas:                    maxRepl,
			TargetCPUUtilizationPercentage: &cpu,
		},
	}
	_, errHPA := rc.framework.Client.Autoscaling().HorizontalPodAutoscalers(rc.framework.Namespace.Name).Create(hpa)
	framework.ExpectNoError(errHPA)
}
