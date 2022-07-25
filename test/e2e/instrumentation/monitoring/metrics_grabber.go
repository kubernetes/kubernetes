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
	"errors"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	clientset "k8s.io/client-go/kubernetes"

	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = instrumentation.SIGDescribe("MetricsGrabber", func() {
	f := framework.NewDefaultFramework("metrics-grabber")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	var c, ec clientset.Interface
	var grabber *e2emetrics.Grabber
	ginkgo.BeforeEach(func() {
		var err error
		c = f.ClientSet
		ec = f.KubemarkExternalClusterClientSet
		gomega.Eventually(func() error {
			grabber, err = e2emetrics.NewMetricsGrabber(c, ec, f.ClientConfig(), true, true, true, true, true, true)
			if err != nil {
				return fmt.Errorf("failed to create metrics grabber: %v", err)
			}
			return nil
		}, 5*time.Minute, 10*time.Second).Should(gomega.BeNil())
	})

	ginkgo.It("should grab all metrics from API server.", func() {
		ginkgo.By("Connecting to /metrics endpoint")
		response, err := grabber.GrabFromAPIServer()
		if errors.Is(err, e2emetrics.MetricsGrabbingDisabledError) {
			e2eskipper.Skipf("%v", err)
		}
		framework.ExpectNoError(err)
		gomega.Expect(response).NotTo(gomega.BeEmpty())
	})

	ginkgo.It("should grab all metrics from a Kubelet.", func() {
		ginkgo.By("Proxying to Node through the API server")
		node, err := e2enode.GetRandomReadySchedulableNode(f.ClientSet)
		if errors.Is(err, e2emetrics.MetricsGrabbingDisabledError) {
			e2eskipper.Skipf("%v", err)
		}
		framework.ExpectNoError(err)
		response, err := grabber.GrabFromKubelet(node.Name)
		framework.ExpectNoError(err)
		gomega.Expect(response).NotTo(gomega.BeEmpty())
	})

	ginkgo.It("should grab all metrics from a Scheduler.", func() {
		ginkgo.By("Proxying to Pod through the API server")
		response, err := grabber.GrabFromScheduler()
		if errors.Is(err, e2emetrics.MetricsGrabbingDisabledError) {
			e2eskipper.Skipf("%v", err)
		}
		framework.ExpectNoError(err)
		gomega.Expect(response).NotTo(gomega.BeEmpty())
	})

	ginkgo.It("should grab all metrics from a ControllerManager.", func() {
		ginkgo.By("Proxying to Pod through the API server")
		response, err := grabber.GrabFromControllerManager()
		if errors.Is(err, e2emetrics.MetricsGrabbingDisabledError) {
			e2eskipper.Skipf("%v", err)
		}
		framework.ExpectNoError(err)
		gomega.Expect(response).NotTo(gomega.BeEmpty())
	})
})
