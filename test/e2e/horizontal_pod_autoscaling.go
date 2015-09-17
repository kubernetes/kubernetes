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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/apis/experimental"

	. "github.com/onsi/ginkgo"
)

const (
	kind        = "replicationController"
	subresource = "scale"
)

var _ = Describe("Horizontal pod autoscaling", func() {
	var rc *ResourceConsumer
	f := NewFramework("horizontal-pod-autoscaling")

	BeforeEach(func() {
		Skip("Skipped Horizontal pod autoscaling test")
	})

	AfterEach(func() {
	})

	// CPU tests
	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 1 pod to 3 pods (scale resource: CPU)", func() {
		rc = NewResourceConsumer("rc", 1, 700, 0, f)
		createCPUHorizontalPodAutoscaler(rc, "0.3")
		rc.WaitForReplicas(3)
		rc.CleanUp()
	})

	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 3 pods to 1 pod (scale resource: CPU)", func() {
		rc = NewResourceConsumer("rc", 3, 0, 0, f)
		createCPUHorizontalPodAutoscaler(rc, "0.7")
		rc.WaitForReplicas(1)
		rc.CleanUp()
	})

	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 1 pod to maximum 5 pods (scale resource: CPU)", func() {
		rc = NewResourceConsumer("rc", 1, 700, 0, f)
		createCPUHorizontalPodAutoscaler(rc, "0.1")
		rc.WaitForReplicas(5)
		rc.CleanUp()
	})

	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 1 pod to 3 pods and from 3 to 1 (scale resource: CPU)", func() {
		rc = NewResourceConsumer("rc", 1, 700, 0, f)
		createCPUHorizontalPodAutoscaler(rc, "0.3")
		rc.WaitForReplicas(3)
		rc.ConsumeCPU(300)
		rc.WaitForReplicas(1)
		rc.CleanUp()
	})

	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 1 pod to 3 pods and from 3 to 5 (scale resource: CPU)", func() {
		rc = NewResourceConsumer("rc", 1, 300, 0, f)
		createCPUHorizontalPodAutoscaler(rc, "0.1")
		rc.WaitForReplicas(3)
		rc.ConsumeCPU(700)
		rc.WaitForReplicas(5)
		rc.CleanUp()
	})

	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 3 pods to 1 pod and from 1 to 3 (scale resource: CPU)", func() {
		rc = NewResourceConsumer("rc", 3, 0, 0, f)
		createCPUHorizontalPodAutoscaler(rc, "0.3")
		rc.WaitForReplicas(1)
		rc.ConsumeCPU(700)
		rc.WaitForReplicas(3)
		rc.CleanUp()
	})

	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 5 pods to 3 pods and from 3 to 1 (scale resource: CPU)", func() {
		rc = NewResourceConsumer("rc", 5, 700, 0, f)
		createCPUHorizontalPodAutoscaler(rc, "0.3")
		rc.WaitForReplicas(3)
		rc.ConsumeCPU(100)
		rc.WaitForReplicas(1)
		rc.CleanUp()
	})

	// Memory tests
	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 1 pod to 3 pods (scale resource: Memory)", func() {
		rc = NewResourceConsumer("rc", 1, 0, 800, f)
		createMemoryHorizontalPodAutoscaler(rc, "300")
		rc.WaitForReplicas(3)
		rc.CleanUp()
	})

	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 3 pods to 1 pod (scale resource: Memory)", func() {
		rc = NewResourceConsumer("rc", 3, 0, 0, f)
		createMemoryHorizontalPodAutoscaler(rc, "700")
		rc.WaitForReplicas(1)
		rc.CleanUp()
	})

	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 1 pod to maximum 5 pods (scale resource: Memory)", func() {
		rc = NewResourceConsumer("rc", 1, 0, 700, f)
		createMemoryHorizontalPodAutoscaler(rc, "100")
		rc.WaitForReplicas(5)
		rc.CleanUp()
	})

	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 1 pod to 3 pods and from 3 to 1 (scale resource: Memory)", func() {
		rc = NewResourceConsumer("rc", 1, 0, 700, f)
		createMemoryHorizontalPodAutoscaler(rc, "300")
		rc.WaitForReplicas(3)
		rc.ConsumeMem(100)
		rc.WaitForReplicas(1)
		rc.CleanUp()
	})

	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 1 pod to 3 pods and from 3 to 5 (scale resource: Memory)", func() {
		rc = NewResourceConsumer("rc", 1, 0, 500, f)
		createMemoryHorizontalPodAutoscaler(rc, "200")
		rc.WaitForReplicas(3)
		rc.ConsumeMem(1000)
		rc.WaitForReplicas(5)
		rc.CleanUp()
	})

	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 3 pods to 1 pod and from 1 to 3 (scale resource: Memory)", func() {
		rc = NewResourceConsumer("rc", 3, 0, 0, f)
		createMemoryHorizontalPodAutoscaler(rc, "300")
		rc.WaitForReplicas(1)
		rc.ConsumeMem(700)
		rc.WaitForReplicas(3)
		rc.CleanUp()
	})

	It("[Skipped][Horizontal pod autoscaling Suite] should scale from 5 pods to 3 pods and from 3 to 1 (scale resource: Memory)", func() {
		rc = NewResourceConsumer("rc", 5, 0, 700, f)
		createMemoryHorizontalPodAutoscaler(rc, "300")
		rc.WaitForReplicas(3)
		rc.ConsumeMem(100)
		rc.WaitForReplicas(1)
		rc.CleanUp()
	})

})

func createCPUHorizontalPodAutoscaler(rc *ResourceConsumer, cpu string) {
	hpa := &experimental.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:      rc.name,
			Namespace: rc.framework.Namespace.Name,
		},
		Spec: experimental.HorizontalPodAutoscalerSpec{
			ScaleRef: &experimental.SubresourceReference{
				Kind:        kind,
				Name:        rc.name,
				Namespace:   rc.framework.Namespace.Name,
				Subresource: subresource,
			},
			MinCount: 1,
			MaxCount: 5,
			Target:   experimental.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse(cpu)},
		},
	}
	_, errHPA := rc.framework.Client.Experimental().HorizontalPodAutoscalers(rc.framework.Namespace.Name).Create(hpa)
	expectNoError(errHPA)
}

// argument memory is in megabytes
func createMemoryHorizontalPodAutoscaler(rc *ResourceConsumer, memory string) {
	hpa := &experimental.HorizontalPodAutoscaler{
		ObjectMeta: api.ObjectMeta{
			Name:      rc.name,
			Namespace: rc.framework.Namespace.Name,
		},
		Spec: experimental.HorizontalPodAutoscalerSpec{
			ScaleRef: &experimental.SubresourceReference{
				Kind:        kind,
				Name:        rc.name,
				Namespace:   rc.framework.Namespace.Name,
				Subresource: subresource,
			},
			MinCount: 1,
			MaxCount: 5,
			Target:   experimental.ResourceConsumption{Resource: api.ResourceMemory, Quantity: resource.MustParse(memory + "M")},
		},
	}
	_, errHPA := rc.framework.Client.Experimental().HorizontalPodAutoscalers(rc.framework.Namespace.Name).Create(hpa)
	expectNoError(errHPA)
}
