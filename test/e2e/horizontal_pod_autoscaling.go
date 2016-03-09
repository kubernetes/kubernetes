/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/apis/extensions"

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

var _ = Describe("Horizontal pod autoscaling (scale resource: CPU)", func() {
	var rc *ResourceConsumer
	f := NewDefaultFramework("horizontal-pod-autoscaling")

	titleUp := "Should scale from 1 pod to 3 pods and from 3 to 5 and verify decision stability"
	titleDown := "Should scale from 5 pods to 3 pods and from 3 to 1 and verify decision stability"

	// These tests take ~20 minutes each.
	Describe("[Serial] [Slow] Deployment", func() {
		// CPU tests via deployments
		It(titleUp, func() {
			scaleUp("test-deployment", kindDeployment, rc, f)
		})
		It(titleDown, func() {
			scaleDown("test-deployment", kindDeployment, rc, f)
		})
	})

	// These tests take ~20 minutes each.
	Describe("[Serial] [Slow] ReplicaSet", func() {
		// CPU tests via deployments
		It(titleUp, func() {
			scaleUp("rs", kindReplicaSet, rc, f)
		})
		It(titleDown, func() {
			scaleDown("rs", kindReplicaSet, rc, f)
		})
	})

	// These tests take ~20 minutes each.
	Describe("[Serial] [Slow] ReplicationController", func() {
		// CPU tests via replication controllers
		It(titleUp, func() {
			scaleUp("rc", kindRC, rc, f)
		})
		It(titleDown, func() {
			scaleDown("rc", kindRC, rc, f)
		})
	})

	Describe("ReplicationController light", func() {
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
		It("Should scale from 2 pods to 1 pod using HPA version v1", func() {
			scaleTest := &HPAScaleTest{
				initPods:                    2,
				totalInitialCPUUsage:        50,
				perPodCPURequest:            200,
				targetCPUUtilizationPercent: 50,
				minPods:                     1,
				maxPods:                     2,
				firstScale:                  1,
				useV1:                       true,
			}
			scaleTest.run("rc-light", kindRC, rc, f)
		})
	})
})

// HPAScaleTest struct is used by the scale(...) function.
type HPAScaleTest struct {
	initPods                    int
	totalInitialCPUUsage        int
	perPodCPURequest            int64
	targetCPUUtilizationPercent int
	minPods                     int
	maxPods                     int
	firstScale                  int
	firstScaleStasis            time.Duration
	cpuBurst                    int
	secondScale                 int
	secondScaleStasis           time.Duration
	useV1                       bool
}

// run is a method which runs an HPA lifecycle, from a starting state, to an expected
// The initial state is defined by the initPods parameter.
// The first state change is due to the CPU being consumed initially, which HPA responds to by changing pod counts.
// The second state change (optional) is due to the CPU burst parameter, which HPA again responds to.
// TODO The use of 3 states is arbitrary, we could eventually make this test handle "n" states once this test stabilizes.
func (scaleTest *HPAScaleTest) run(name, kind string, rc *ResourceConsumer, f *Framework) {
	rc = NewDynamicResourceConsumer(name, kind, scaleTest.initPods, scaleTest.totalInitialCPUUsage, 0, 0, scaleTest.perPodCPURequest, 100, f)
	defer rc.CleanUp()
	createCPUHorizontalPodAutoscaler(rc, scaleTest.targetCPUUtilizationPercent, scaleTest.minPods, scaleTest.maxPods, scaleTest.useV1)
	rc.WaitForReplicas(scaleTest.firstScale)
	if scaleTest.firstScaleStasis > 0 {
		rc.EnsureDesiredReplicas(scaleTest.firstScale, scaleTest.firstScaleStasis)
	}
	if scaleTest.cpuBurst > 0 && scaleTest.secondScale > 0 {
		rc.ConsumeCPU(scaleTest.cpuBurst)
		rc.WaitForReplicas(scaleTest.secondScale)
	}
}

func scaleUp(name, kind string, rc *ResourceConsumer, f *Framework) {
	scaleTest := &HPAScaleTest{
		initPods:                    1,
		totalInitialCPUUsage:        250,
		perPodCPURequest:            500,
		targetCPUUtilizationPercent: 20,
		minPods:                     1,
		maxPods:                     5,
		firstScale:                  3,
		firstScaleStasis:            10 * time.Minute,
		cpuBurst:                    700,
		secondScale:                 5,
	}
	scaleTest.run(name, kind, rc, f)
}

func scaleDown(name, kind string, rc *ResourceConsumer, f *Framework) {
	scaleTest := &HPAScaleTest{
		initPods:                    5,
		totalInitialCPUUsage:        400,
		perPodCPURequest:            500,
		targetCPUUtilizationPercent: 30,
		minPods:                     1,
		maxPods:                     5,
		firstScale:                  3,
		firstScaleStasis:            10 * time.Minute,
		cpuBurst:                    100,
		secondScale:                 1,
	}
	scaleTest.run(name, kind, rc, f)
}

func createCPUHorizontalPodAutoscaler(rc *ResourceConsumer, cpu, minReplicas, maxRepl int, useV1 bool) {
	hpa := &extensions.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:      rc.name,
			Namespace: rc.framework.Namespace.Name,
		},
		Spec: extensions.HorizontalPodAutoscalerSpec{
			ScaleRef: extensions.SubresourceReference{
				Kind:        rc.kind,
				Name:        rc.name,
				Subresource: subresource,
			},
			MinReplicas:    &minReplicas,
			MaxReplicas:    maxRepl,
			CPUUtilization: &extensions.CPUTargetUtilization{TargetPercentage: cpu},
		},
	}
	var errHPA error
	if useV1 {
		_, errHPA = rc.framework.Client.Autoscaling().HorizontalPodAutoscalers(rc.framework.Namespace.Name).Create(hpa)
	} else {
		_, errHPA = rc.framework.Client.Extensions().HorizontalPodAutoscalers(rc.framework.Namespace.Name).Create(hpa)
	}
	expectNoError(errHPA)
}
