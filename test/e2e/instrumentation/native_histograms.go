/*
Copyright The Kubernetes Authors.

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

package instrumentation

import (
	"bytes"
	"context"
	"errors"
	"io"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/expfmt"
	clientset "k8s.io/client-go/kubernetes"
	metricsfeatures "k8s.io/component-base/metrics/features"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/instrumentation/common"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = common.SIGDescribe("NativeHistograms", framework.WithFeatureGate(metricsfeatures.NativeHistograms), func() {
	f := framework.NewDefaultFramework("native-histograms")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var c clientset.Interface

	ginkgo.BeforeEach(func(ctx context.Context) {
		c = f.ClientSet
	})

	/*
	   Release: v1.37
	   Testname: Native Histograms export verification
	   Description: Queries the kube-apiserver /metrics endpoint negotiating Protobuf format
	   and asserts that both native and classic histogram structures are correctly generated for dual-exposition.
	*/
	ginkgo.It("should export both classic and native histograms (in protobuf format) from apiserver /metrics", func(ctx context.Context) {
		ginkgo.By("Grabbing metrics in Protobuf format from APIServer")
		body, err := c.CoreV1().RESTClient().Get().
			RequestURI("/metrics").
			SetHeader("Accept", "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=delimited").
			DoRaw(ctx)
		framework.ExpectNoError(err)
		gomega.Expect(body).NotTo(gomega.BeEmpty())

		ginkgo.By("Parsing Protobuf payload and asserting native histogram schema")
		reader := bytes.NewReader(body)
		dec := expfmt.NewDecoder(reader, expfmt.NewFormat(expfmt.TypeProtoDelim))

		var targetMetricFamily *dto.MetricFamily

		for {
			var mf dto.MetricFamily
			err := dec.Decode(&mf)
			if errors.Is(err, io.EOF) {
				break
			}
			framework.ExpectNoError(err)

			if mf.GetName() == "apiserver_request_duration_seconds" {
				targetMetricFamily = &mf
				break
			}
		}

		if !foundNativeHistogram(targetMetricFamily) {
			ginkgo.Fail("Expected to find a native histogram inside apiserver_request_duration_seconds")
		}
		if !foundClassicHistogram(targetMetricFamily) {
			ginkgo.Fail("Expected to find classic buckets inside apiserver_request_duration_seconds for backward compatibility")
		}
	})
})

func foundNativeHistogram(mf *dto.MetricFamily) bool {
	if mf == nil {
		return false
	}
	if mf.GetType() != dto.MetricType_HISTOGRAM {
		return false
	}

	for _, m := range mf.Metric {
		h := m.GetHistogram()
		if h == nil {
			continue
		}

		// Native Histograms are characterized by having a non-zero/non-nil Schema value.
		if h.Schema != nil && *h.Schema != 0 {
			// Assert that if observations exist, the dynamic positive spans
			// are populated (since API request latencies are positive durations).
			if h.GetSampleCount() > 0 && len(h.GetPositiveSpan()) > 0 {
				return true
			}
		}
	}
	return false
}

func foundClassicHistogram(mf *dto.MetricFamily) bool {
	if mf == nil {
		return false
	}
	if mf.GetType() != dto.MetricType_HISTOGRAM {
		return false
	}

	for _, m := range mf.Metric {
		h := m.GetHistogram()
		if h == nil {
			continue
		}

		// Classic histograms define bucket bounds and cumulative counts.
		if len(h.Bucket) > 0 {
			return true
		}
	}
	return false
}
