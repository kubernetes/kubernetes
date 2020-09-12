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
	"context"
	"fmt"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = instrumentation.SIGDescribe("MetricsGrabber", func() {
	f := framework.NewDefaultFramework("metrics-grabber")
	var c, ec clientset.Interface
	var grabber *e2emetrics.Grabber
	ginkgo.BeforeEach(func() {
		var err error
		c = f.ClientSet
		ec = f.KubemarkExternalClusterClientSet
		framework.ExpectNoError(err)
		gomega.Eventually(func() error {
			grabber, err = e2emetrics.NewMetricsGrabber(c, ec, true, true, true, true, true)
			if err != nil {
				return fmt.Errorf("failed to create metrics grabber: %v", err)
			}
			if !grabber.HasControlPlanePods() {
				return fmt.Errorf("unable to get find control plane pods")
			}
			return nil
		}, 5*time.Minute, 10*time.Second).Should(gomega.BeNil())
	})

	ginkgo.It("should grab all metrics from API server.", func() {
		ginkgo.By("Connecting to /metrics endpoint")
		response, err := grabber.GrabFromAPIServer()
		framework.ExpectNoError(err)
		gomega.Expect(response).NotTo(gomega.BeEmpty())
	})

	ginkgo.It("should grab all metrics from a Kubelet.", func() {
		ginkgo.By("Proxying to Node through the API server")
		node, err := e2enode.GetRandomReadySchedulableNode(f.ClientSet)
		framework.ExpectNoError(err)
		response, err := grabber.GrabFromKubelet(node.Name)
		framework.ExpectNoError(err)
		gomega.Expect(response).NotTo(gomega.BeEmpty())
	})

	ginkgo.It("should grab all metrics from a Scheduler.", func() {
		ginkgo.By("Proxying to Pod through the API server")
		// Check if master Node is registered
		nodes, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
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
		gomega.Expect(response).NotTo(gomega.BeEmpty())
	})

	ginkgo.It("should grab all metrics from a ControllerManager.", func() {
		ginkgo.By("Proxying to Pod through the API server")
		// Check if master Node is registered
		nodes, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
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
		gomega.Expect(response).NotTo(gomega.BeEmpty())
	})
})
