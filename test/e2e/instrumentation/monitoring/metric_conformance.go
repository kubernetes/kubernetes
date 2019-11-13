/*
Copyright 2019 The Kubernetes Authors.

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

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/metrics"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"

	gin "github.com/onsi/ginkgo"
	gom "github.com/onsi/gomega"
	promlint "github.com/prometheus/prometheus/util/promlint"
)

var _ = instrumentation.SIGDescribe("Metric Conformance [Feature:MetricConformance]", func() {
	f := framework.NewDefaultFramework("metrics-conformance")
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

	gin.It("all metrics from API server should conform to promtool.", func() {
		gin.By("Connecting to /metrics endpoint")
		apimetrics, err := grabber.GrabRawFromAPIServer()
		framework.ExpectNoError(err)
		gom.Expect(apimetrics).NotTo(gom.BeEmpty())

		linter := promlint.New(strings.NewReader(apimetrics))
		problems, err := linter.Lint()
		framework.ExpectNoError(err)
		gom.Expect(problems).To(gom.BeEmpty())
	})
})
