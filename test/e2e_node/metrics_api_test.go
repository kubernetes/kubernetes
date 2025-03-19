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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/component-base/metrics"
	v1 "k8s.io/component-base/metrics/api/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = framework.SIGDescribe("instrumentation")("Kubelet Config MetricsAPI", ginkgo.Label("MetricsAPI"), framework.WithDisruptive(), framework.WithSlow(), framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("kubelet-config-metrics-api-test")

	ginkgo.Context("metrics api should exhibit expected behavior", func() {
		var oldCfg *kubeletconfig.KubeletConfiguration

		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			if oldCfg == nil {
				oldCfg, err = getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
			}
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			updateKubeletConfig(ctx, f, oldCfg, true)
		})

		ginkgo.It("when empty", func(ctx context.Context) {
			newCfg := oldCfg.DeepCopy()
			newCfg.Metrics = v1.MetricsConfiguration{}
			updateKubeletConfig(ctx, f, newCfg, true)
		})

		ginkgo.It("when showHiddenMetricsForVersion is populated", func(ctx context.Context) {
			testMetric := "apiserver_encryption_config_controller_automatic_reload_failures_total"
			newCfg := oldCfg.DeepCopy()
			newCfg.Metrics = v1.MetricsConfiguration{
				Options: metrics.Options{
					ShowHiddenMetricsForVersion: "1.31",
				},
			}

			ginkgo.By("expecting " + testMetric + " absence before kubelet config update")
			kubeletMetrics, err := getKubeletMetrics(ctx)
			framework.ExpectNoError(err)
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					gomega.Expect(sample.Metric["__name__"]).NotTo(gomega.BeEquivalentTo(testMetric))
				}
			}
			updateKubeletConfig(ctx, f, newCfg, true)

			ginkgo.By("expecting " + testMetric + " presence after kubelet config update")
			kubeletMetrics, err = getKubeletMetrics(ctx)
			framework.ExpectNoError(err)
			found := false
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					if string(sample.Metric["__name__"]) == testMetric {
						found = true
						break
					}
				}
			}
			gomega.Expect(found).To(gomega.BeTrueBecause("expected metric to be present"))
		})

		ginkgo.It("when disabledMetrics is populated", func(ctx context.Context) {
			testMetric := "node_cpu_usage_seconds_total"
			newCfg := oldCfg.DeepCopy()
			newCfg.Metrics = v1.MetricsConfiguration{
				Options: metrics.Options{
					DisabledMetrics: []string{testMetric},
				},
			}

			ginkgo.By("expecting " + testMetric + " presence before kubelet config update")
			kubeletMetrics, err := getKubeletMetrics(ctx)
			framework.ExpectNoError(err)
			found := false
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					if string(sample.Metric["__name__"]) == testMetric {
						found = true
						break
					}
				}
			}
			gomega.Expect(found).To(gomega.BeTrueBecause("expected metrics to be present"))
			framework.ExpectNoError(err)
			updateKubeletConfig(ctx, f, newCfg, true)

			ginkgo.By("expecting " + testMetric + " absence after kubelet config update")
			kubeletMetrics, err = getKubeletMetrics(ctx)
			framework.ExpectNoError(err)
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					gomega.Expect(sample.Metric["__name__"]).NotTo(gomega.BeEquivalentTo(testMetric))
				}
			}
		})

		ginkgo.It("when allowListMapping is populated", func(ctx context.Context) {
			testMetric := "kube_apiserver_clusterip_allocator_allocated_ips"
			newCfg := oldCfg.DeepCopy()
			newCfg.Metrics = v1.MetricsConfiguration{
				Options: metrics.Options{
					AllowListMapping: map[string]string{
						testMetric + ",cidr":  "",
						testMetric + ",scope": "static",
					},
				},
			}

			ginkgo.By("expecting all " + testMetric + " label-sets to be present before kubelet config update")
			kubeletMetrics, err := getKubeletMetrics(ctx)
			framework.ExpectNoError(err)
			found := false
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					if string(sample.Metric["__name__"]) == testMetric {
						found = true
						gomega.Expect(sample.Metric["cidr"]).NotTo(gomega.BeNil())
					}
				}
			}
			gomega.Expect(found).To(gomega.BeTrueBecause("expected metric to be present"))
			updateKubeletConfig(ctx, f, newCfg, true)

			ginkgo.By("expecting only a subset of " + testMetric + " label-sets to be present after kubelet config update")
			kubeletMetrics, err = getKubeletMetrics(ctx)
			framework.ExpectNoError(err)
			found = false
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					if string(sample.Metric["__name__"]) == testMetric {
						found = true
						gomega.Expect(sample.Metric["cidr"]).To(gomega.BeNil())
					}
				}
			}
			gomega.Expect(found).To(gomega.BeTrueBecause("expected metric to be present"))
		})

		ginkgo.It("when allowListMappingManifest is populated", func(ctx context.Context) {
			testMetric := "kube_apiserver_clusterip_allocator_allocated_ips"
			manifest := []byte(fmt.Sprintf(`%s,cidr:
%s,scope: static`, testMetric, testMetric))
			manifestPath := "allow-list-mapping.yaml"
			framework.ExpectNoError(os.WriteFile(manifestPath, manifest, 0755))
			newCfg := oldCfg.DeepCopy()
			newCfg.Metrics = v1.MetricsConfiguration{
				Options: metrics.Options{
					AllowListMappingManifest: manifestPath,
				},
			}

			ginkgo.By("expecting all " + testMetric + " label-sets to be present before kubelet config update")
			kubeletMetrics, err := getKubeletMetrics(ctx)
			framework.ExpectNoError(err)
			found := false
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					if string(sample.Metric["__name__"]) == testMetric {
						found = true
						gomega.Expect(sample.Metric["cidr"]).NotTo(gomega.BeNil())
					}
				}
			}
			gomega.Expect(found).To(gomega.BeTrueBecause("expected metric to be present"))
			updateKubeletConfig(ctx, f, newCfg, true)

			ginkgo.By("expecting only a subset of " + testMetric + " label-sets to be present after kubelet config update")
			kubeletMetrics, err = getKubeletMetrics(ctx)
			framework.ExpectNoError(err)
			found = false
			for _, samples := range kubeletMetrics {
				for _, sample := range samples {
					if string(sample.Metric["__name__"]) == testMetric {
						found = true
						gomega.Expect(sample.Metric["cidr"]).To(gomega.BeNil())
					}
				}
			}
			gomega.Expect(found).To(gomega.BeTrueBecause("expected metric to be present"))
			framework.ExpectNoError(os.Remove(manifestPath))
		})
	})
})
