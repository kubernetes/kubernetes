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
	subresource    = "scale"
)

// These tests don't seem to be running properly in parallel: issue: #20338.
//
// These tests take ~20 minutes each.
var _ = Describe("Horizontal pod autoscaling (scale resource: CPU) [Serial] [Slow]", func() {
	var rc *ResourceConsumer
	f := NewFramework("horizontal-pod-autoscaling")

	titleUp := "Should scale from 1 pod to 3 pods and from 3 to 5"
	titleDown := "Should scale from 5 pods to 3 pods and from 3 to 1"

	// TODO(madhusudancs): Fix this when Scale group issues are resolved.
	// Describe("Deployment [Feature:Deployment]", func() {
	// 	// CPU tests via deployments
	// 	It(titleUp, func() {
	// 		scaleUp("deployment", kindDeployment, rc, f)
	// 	})
	// 	It(titleDown, func() {
	// 		scaleDown("deployment", kindDeployment, rc, f)
	// 	})
	// })

	Describe("ReplicationController", func() {
		// CPU tests via replication controllers
		It(titleUp, func() {
			scaleUp("rc", kindRC, rc, f)
		})
		It(titleDown, func() {
			scaleDown("rc", kindRC, rc, f)
		})
	})
})

// HPAScaleTest struct is used by the scale(...) function.
type HPAScaleTest struct {
	initPods          int
	cpuStart          int
	maxCPU            int64
	idealCPU          int
	minPods           int
	maxPods           int
	firstScale        int
	firstScaleStasis  time.Duration
	cpuBurst          int
	secondScale       int
	secondScaleStasis time.Duration
}

// run is a method which runs an HPA lifecycle, from a starting state, to an expected
// The initial state is defined by the initPods parameter.
// The first state change is due to the CPU being consumed initially, which HPA responds to by changing pod counts.
// The second state change is due to the CPU burst parameter, which HPA again responds to.
// TODO The use of 3 states is arbitrary, we could eventually make this test handle "n" states once this test stabilizes.
func (scaleTest *HPAScaleTest) run(name, kind string, rc *ResourceConsumer, f *Framework) {
	rc = NewDynamicResourceConsumer(name, kind, scaleTest.initPods, scaleTest.cpuStart, 0, scaleTest.maxCPU, 100, f)
	defer rc.CleanUp()
	createCPUHorizontalPodAutoscaler(rc, scaleTest.idealCPU, scaleTest.minPods, scaleTest.maxPods)
	rc.WaitForReplicas(scaleTest.firstScale)
	rc.EnsureDesiredReplicas(scaleTest.firstScale, scaleTest.firstScaleStasis)
	rc.ConsumeCPU(scaleTest.cpuBurst)
	rc.WaitForReplicas(scaleTest.secondScale)
}

func scaleUp(name, kind string, rc *ResourceConsumer, f *Framework) {
	scaleTest := &HPAScaleTest{
		initPods:         1,
		cpuStart:         250,
		maxCPU:           500,
		idealCPU:         .2 * 100,
		minPods:          1,
		maxPods:          5,
		firstScale:       3,
		firstScaleStasis: 10 * time.Minute,
		cpuBurst:         700,
		secondScale:      5,
	}
	scaleTest.run(name, kind, rc, f)
}

func scaleDown(name, kind string, rc *ResourceConsumer, f *Framework) {
	scaleTest := &HPAScaleTest{
		initPods:         5,
		cpuStart:         400,
		maxCPU:           500,
		idealCPU:         .3 * 100,
		minPods:          1,
		maxPods:          5,
		firstScale:       3,
		firstScaleStasis: 10 * time.Minute,
		cpuBurst:         100,
		secondScale:      1,
	}
	scaleTest.run(name, kind, rc, f)
}

func createCPUHorizontalPodAutoscaler(rc *ResourceConsumer, cpu, minReplicas, maxRepl int) {
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
	_, errHPA := rc.framework.Client.Extensions().HorizontalPodAutoscalers(rc.framework.Namespace.Name).Create(hpa)
	expectNoError(errHPA)
}
