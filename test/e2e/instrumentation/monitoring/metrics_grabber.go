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

package monitoring

import (
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/metrics"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"

	gin "github.com/onsi/ginkgo"
	gom "github.com/onsi/gomega"
)

var _ = instrumentation.SIGDescribe("MetricsGrabber", func() {
	f := framework.NewDefaultFramework("metrics-grabber")
	var c, ec clientset.Interface
	var grabber *metrics.Grabber
	gin.BeforeEach(func() {
		var err error
		c = f.ClientSet
		ec = f.KubemarkExternalClusterClientSet
		framework.ExpectNoError(err)
		grabber, err = metrics.NewMetricsGrabber(c, ec, true, true, true, true, true)
		framework.ExpectNoError(err)
	})

	gin.It("should grab all metrics from API server.", func() {
		gin.By("Connecting to /metrics endpoint")
		response, err := grabber.GrabFromAPIServer()
		framework.ExpectNoError(err)
		gom.Expect(response).NotTo(gom.BeEmpty())
	})

	gin.It("should grab all metrics from a Kubelet.", func() {
		gin.By("Proxying to Node through the API server")
		node, err := e2enode.GetRandomReadySchedulableNode(f.ClientSet)
		framework.ExpectNoError(err)
		response, err := grabber.GrabFromKubelet(node.Name)
		framework.ExpectNoError(err)
		gom.Expect(response).NotTo(gom.BeEmpty())
	})

	gin.It("should grab all metrics from a Scheduler.", func() {
		gin.By("Proxying to Pod through the API server")
		// Check if master Node is registered
		nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
		framework.ExpectNoError(err)

		var masterRegistered = false
		for _, node := range nodes.Items {
			if strings.HasSuffix(node.Name, "master") {
				masterRegistered = true
			}
		}
		if !masterRegistered {
			framework.Logf("Master is node api.Registry. Skipping testing Scheduler metrics.")
			return
		}
		response, err := grabber.GrabFromScheduler()
		framework.ExpectNoError(err)
		gom.Expect(response).NotTo(gom.BeEmpty())
	})

	gin.It("should grab all metrics from a ControllerManager.", func() {
		gin.By("Proxying to Pod through the API server")
		// Check if master Node is registered
		nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
		framework.ExpectNoError(err)

		var masterRegistered = false
		for _, node := range nodes.Items {
			if strings.HasSuffix(node.Name, "master") {
				masterRegistered = true
			}
		}
		if !masterRegistered {
			framework.Logf("Master is node api.Registry. Skipping testing ControllerManager metrics.")
			return
		}
		response, err := grabber.GrabFromControllerManager()
		framework.ExpectNoError(err)
		gom.Expect(response).NotTo(gom.BeEmpty())
	})
})
