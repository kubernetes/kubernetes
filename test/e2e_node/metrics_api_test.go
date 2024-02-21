/*
Copyright 2024 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/prometheus/common/model"
	"k8s.io/component-base/metrics"
	v1 "k8s.io/component-base/metrics/api/v1"
	"k8s.io/kubernetes/pkg/cluster/ports"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
)

var _ = framework.SIGDescribe("metrics-api")("Kubelet Config MetricsAPI", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("kubelet-config-metrics-api-test")

	ginkgo.Context("metrics api should exhibit expected behavior", func() {
		var oldConfig *kubeletconfig.KubeletConfiguration

		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			if oldConfig == nil {
				oldConfig, err = getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
			}
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			updateKubeletConfig(ctx, f, oldConfig, true)
		})

		ginkgo.It("when empty", func(ctx context.Context) {
			updateKubeletConfig(ctx, f, getKubeletConfigurationWith(oldConfig, metrics.Options{}), true)
		})

		// The test case for showHiddenMetricsForVersion is left out from this spec,
		// as testing that requires injecting a test metric that always remains deprecated in the n-1th version
		// (and thus hidden in the registry's default version, i.e., nth version), into kubelet's registry,
		// thus polluting it, and therefore not ideal.

		ginkgo.It("when disabledMetrics is populated", func(ctx context.Context) {
			metricName := "kubelet_http_inflight_requests"

			ginkgo.By(fmt.Sprintf("expecting %s to be present before disabledMetrics is in effect", metricName))
			kubeletMetrics, err := e2emetrics.GrabKubeletMetricsWithoutProxy(ctx, fmt.Sprintf("%s:%d", "localhost", ports.KubeletReadOnlyPort), "/metrics")
			framework.ExpectNoError(err)
			found := false
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					if string(sample.Metric["__name__"]) == metricName {
						found = true
						break
					}
				}
			}
			gomega.Expect(found).To(gomega.BeTrueBecause("expected metric to be present"))

			ginkgo.By(fmt.Sprintf("expecting %s to be dropped after disabledMetrics is in effect", metricName))
			metricsOptions := metrics.Options{
				DisabledMetrics: []string{metricName},
			}
			updateKubeletConfig(ctx, f, getKubeletConfigurationWith(oldConfig, metricsOptions), true)
			kubeletMetrics, err = getKubeletMetricsWithoutProxy(ctx)
			framework.ExpectNoError(err)
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					gomega.Expect(sample.Metric["__name__"]).NotTo(gomega.BeEquivalentTo(metricName))
				}
			}
		})

		ginkgo.It("when allowListMapping is populated", func(ctx context.Context) {
			metricName := "kubelet_http_inflight_requests"
			metricLabel := model.LabelName("path")

			ginkgo.By(fmt.Sprintf("expecting %s's %s label-set to be present before allowListMapping is in effect", metricName, metricLabel))
			kubeletMetrics, err := getKubeletMetrics(ctx)
			framework.ExpectNoError(err)
			found := false
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					if string(sample.Metric["__name__"]) == metricName {
						found = true
						gomega.Expect(sample.Metric[metricLabel]).NotTo(gomega.BeEmpty())
					}
				}
			}
			gomega.Expect(found).To(gomega.BeTrueBecause("expected metric to be present"))

			ginkgo.By(fmt.Sprintf("expecting %s's %s label-set to be restricted after allowListMapping is in effect", metricName, metricLabel))
			metricsOptions := metrics.Options{
				AllowListMapping: map[string]string{
					fmt.Sprintf("%s,%s", metricName, metricLabel): "",
				},
			}
			updateKubeletConfig(ctx, f, getKubeletConfigurationWith(oldConfig, metricsOptions), true)
			kubeletMetrics, err = getKubeletMetricsWithoutProxy(ctx)
			framework.ExpectNoError(err)
			found = false
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					if string(sample.Metric["__name__"]) == metricName {
						found = true
						gomega.Expect(sample.Metric[metricLabel]).To(gomega.BeEquivalentTo("unexpected"))
					}
				}
			}
			gomega.Expect(found).To(gomega.BeTrueBecause("expected metric to be present"))
		})

		ginkgo.It("when allowListMappingManifest is populated", func(ctx context.Context) {
			metricName := "kubelet_http_inflight_requests"
			metricLabel := model.LabelName("path")
			manifest := []byte(fmt.Sprintf("%s,%s:", metricName, metricLabel))
			manifestPath := filepath.Join(os.TempDir(), "allow-list-manifest.yaml")
			framework.ExpectNoError(os.WriteFile(manifestPath, manifest, os.ModePerm))

			ginkgo.By(fmt.Sprintf("expecting %s's %s label-set to be present before allowListMappingManifest is in effect", metricName, metricLabel))
			kubeletMetrics, err := getKubeletMetrics(ctx)
			framework.ExpectNoError(err)
			found := false
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					if string(sample.Metric["__name__"]) == metricName {
						found = true
						gomega.Expect(sample.Metric[metricLabel]).NotTo(gomega.BeEmpty())
					}
				}
			}
			gomega.Expect(found).To(gomega.BeTrueBecause("expected metric to be present"))

			ginkgo.By(fmt.Sprintf("expecting %s's %s label-set to be absent after allowListMappingManifest is in effect", metricName, metricLabel))
			metricsOptions := metrics.Options{
				AllowListMappingManifest: manifestPath,
			}
			updateKubeletConfig(ctx, f, getKubeletConfigurationWith(oldConfig, metricsOptions), true)
			kubeletMetrics, err = getKubeletMetricsWithoutProxy(ctx)
			framework.ExpectNoError(err)
			found = false
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					if string(sample.Metric["__name__"]) == metricName {
						found = true
						gomega.Expect(sample.Metric[metricLabel]).To(gomega.BeEquivalentTo("unexpected"))
					}
				}
			}
			gomega.Expect(found).To(gomega.BeTrueBecause("expected metric to be present"))
			framework.ExpectNoError(os.Remove(manifestPath))
		})
	})
})

func getKubeletConfigurationWith(oldConfig *kubeletconfig.KubeletConfiguration, options metrics.Options) *kubeletconfig.KubeletConfiguration {
	newConfig := oldConfig.DeepCopy()
	newConfig.Metrics = v1.MetricsConfiguration{
		Options: options,
	}
	return newConfig
}

func getKubeletMetricsWithoutProxy(ctx context.Context) (e2emetrics.KubeletMetrics, error) {
	return e2emetrics.GrabKubeletMetricsWithoutProxy(ctx, fmt.Sprintf("%s:%d", "localhost", ports.KubeletReadOnlyPort), "/metrics")
}
