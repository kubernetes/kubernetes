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
	kindRC           = "replicationController"
	kindDeployment   = "deployment"
	subresource      = "scale"
	stabilityTimeout = 10 * time.Minute
)

var _ = Describe("Horizontal pod autoscaling", func() {

	var rc *ResourceConsumer
	f := NewFramework("horizontal-pod-autoscaling")

	titleUp := "%s should scale from 1 pod to 3 pods and from 3 to 5 (via %s, with scale resource: CPU)"
	titleDown := "%s should scale from 5 pods to 3 pods and from 3 to 1 (via %s, with scale resource: CPU)"

	// CPU tests via deployments
	It(fmt.Sprintf(titleUp, "[Skipped]", kindDeployment), func() {
		scaleUp("deployment", kindDeployment, rc, f)
	})
	It(fmt.Sprintf(titleDown, "[Skipped]", kindDeployment), func() {
		scaleDown("deployment", kindDeployment, rc, f)
	})

	// CPU tests via replication controllers
	It(fmt.Sprintf(titleUp, "[Autoscaling Suite]", kindRC), func() {
		scaleUp("rc", kindRC, rc, f)
	})
	It(fmt.Sprintf(titleDown, "[Autoscaling Suite]", kindRC), func() {
		scaleDown("rc", kindRC, rc, f)
	})
})

func scaleUp(name, kind string, rc *ResourceConsumer, f *Framework) {
	rc = NewDynamicResourceConsumer(name, kind, 1, 250, 0, 500, 100, f)
	defer rc.CleanUp()
	createCPUHorizontalPodAutoscaler(rc, 20)
	rc.WaitForReplicas(3)
	rc.EnsureDesiredReplicas(3, stabilityTimeout)
	rc.ConsumeCPU(700)
	rc.WaitForReplicas(5)
}

func scaleDown(name, kind string, rc *ResourceConsumer, f *Framework) {
	rc = NewDynamicResourceConsumer(name, kind, 5, 400, 0, 500, 100, f)
	defer rc.CleanUp()
	createCPUHorizontalPodAutoscaler(rc, 30)
	rc.WaitForReplicas(3)
	rc.EnsureDesiredReplicas(3, stabilityTimeout)
	rc.ConsumeCPU(100)
	rc.WaitForReplicas(1)
}

func createCPUHorizontalPodAutoscaler(rc *ResourceConsumer, cpu int) {
	minReplicas := 1
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
			MaxReplicas:    5,
			CPUUtilization: &extensions.CPUTargetUtilization{TargetPercentage: cpu},
		},
	}
	_, errHPA := rc.framework.Client.Extensions().HorizontalPodAutoscalers(rc.framework.Namespace.Name).Create(hpa)
	expectNoError(errHPA)
}
