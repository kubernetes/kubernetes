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
	"fmt"
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

var _ = Describe("Horizontal pod autoscaling (scale resource: CPU) [Skipped]", func() {

	titleUp := "Should scale from 1 pod to 3 pods and from 3 to 5"
	titleDown := "Should scale from 5 pods to 3 pods and from 3 to 1"
	titleMult := "Should HPA scale in parallel"

	Describe("Deployment", func() {
		f1 := NewFramework("horizontal-pod-autoscaling-deploy")
		// CPU tests via deployments
		It(titleUp, func() {
			if !scaleUp("deployment", kindDeployment, f1) {
				Failf("Deployment: Scale up failed.")
			}
		})
		It(titleDown, func() {
			if !scaleDown("deployment", kindDeployment, f1) {
				Failf("Deployment: Scale down failed.")
			}
		})
		It(titleMult+"(deployments)", func() {
			scaleMultitest("deployment", kindDeployment, f1)
		})
	})

	Describe("[Autoscaling] ReplicationController", func() {
		f := NewFramework("horizontal-pod-autoscaling-rc")
		// CPU tests via replication controllers
		It(titleUp, func() {
			if !scaleUp("rc", kindRC, f) {
				Failf("RC: Scale up failed.")
			}
		})
		It(titleDown, func() {
			if !scaleDown("rc", kindRC, f) {
				Failf("RC: Scale down failed.")
			}
		})
		It(titleMult+"(replication controllers)", func() {
			scaleMultitest("rc", kindDeployment, f)
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
func (scaleTest *HPAScaleTest) run(name, kind string, f *Framework) bool {
	rc := NewDynamicResourceConsumer(name, kind, scaleTest.initPods, scaleTest.cpuStart, 0, scaleTest.maxCPU, 100, f)
	defer rc.CleanUp()
	createCPUHorizontalPodAutoscaler(rc, scaleTest.idealCPU, scaleTest.minPods, scaleTest.maxPods)
	passed := rc.WaitForReplicas(scaleTest.firstScale)
	if !passed {
		Logf("first scale: timeout waiting for pods size to be %d", scaleTest.secondScale)
		return false
	}
	// Apparently this is failing, because deleted.  logging time to see why.
	rc.EnsureDesiredReplicas(scaleTest.firstScale, scaleTest.firstScaleStasis)
	rc.ConsumeCPU(scaleTest.cpuBurst)
	passed = rc.WaitForReplicas(scaleTest.secondScale)
	if !passed {
		Logf("second scale: timeout waiting for pods size to be %d", scaleTest.secondScale)
		return false
	}
	return true
}

// Tests that multiple HPAs work properly, in parallel.
// Right now, we only test 2 generations.
func scaleMultitest(name, kind string, f *Framework) {
	testResults := [2]bool{}
	testDone := [2]bool{}
	Logf("start %v", testResults)

	// Important: make sure exactly len(testResults) go func's are created,
	// and that each one writes to a unique element of the tests array.
	for i := 0; i < len(testResults); i += 2 {
		go func(ii int) {
			defer GinkgoRecover()
			testResults[ii] = scaleUp(fmt.Sprint(name, "up", ii), kind, f)
			testDone[ii] = true
		}(i)
		go func(ii int) {
			defer GinkgoRecover()
			testResults[ii] = scaleDown(fmt.Sprint(name, "down", ii), kind, f)
			testDone[ii] = true
		}(i + 1)
	}

	// remainingTests is just a helper function for determining if all tests are completed.
	remainingTests := func() int {
		count := 0
		for i := 0; i < len(testDone); i++ {
			if !testDone[i] {
				count++
			}
		}
		return count
	}

	// here we will wait for all goroutines to complete.
	for i := remainingTests(); i > 0; i = remainingTests() {
		Logf("Remaining tests: %v", i)
		time.Sleep(time.Duration(10) * time.Second)
	}

	Logf("All tests done.  computing result...")
	allTestsPassed := true
	for i := 0; i < len(testResults); i++ {
		if testResults[i] == false {
			Logf("Test %v failed !!!", i)
			allTestsPassed = false
		} else {
			Logf("Test %v passed.", i)
		}
	}
	if !allTestsPassed {
		Failf("Some of the HPA parallel scaling tests failed... See logs above.")
	} else {
		Logf("All HPA scaling events for this test ( %v %v ) passed.", name, kind)
	}
}

func scaleUp(name, kind string, f *Framework) bool {
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
	return scaleTest.run(name, kind, f)
}

func scaleDown(name, kind string, f *Framework) bool {
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
	return scaleTest.run(name, kind, f)
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
