//go:build linux

/*
Copyright 2025 The Kubernetes Authors.

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
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"k8s.io/kubernetes/pkg/features"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	"k8s.io/kubernetes/test/e2e_node/criproxy"
	e2enodekubelet "k8s.io/kubernetes/test/e2e_node/kubeletconfig"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Cgroup Driver From CRI", feature.CriProxy, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("cgroup-driver-from-cri")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func() {
		if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
			ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
		}
	})

	ginkgo.It("should allow temporary recovery when the runtime does not implement RuntimeConfig", func(ctx context.Context) {
		originalConfig, err := getCurrentKubeletConfig(ctx)
		framework.ExpectNoError(err)
		cgroupDriver := originalConfig.CgroupDriver
		// configz reports the driver obtained from CRI, which cannot be supplied as configuration.
		if disabled, set := originalConfig.FeatureGates[string(features.DisableCgroupDriverFallback)]; !set || disabled {
			originalConfig.CgroupDriver = ""
		}
		ginkgo.DeferCleanup(func(ctx context.Context) {
			framework.ExpectNoError(resetCRIProxyInjector(e2eCriProxy))
			framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(originalConfig))
			restartKubelet(ctx, false)
			waitForKubeletToStart(ctx, f)
		})

		config := originalConfig.DeepCopy()
		// Exercise the default even when the suite overrides this gate.
		delete(config.FeatureGates, string(features.DisableCgroupDriverFallback))
		config.CgroupDriver = ""
		updateKubeletConfig(ctx, f, config, false)

		var runtimeConfigCalls atomic.Int64
		framework.ExpectNoError(addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
			if apiName == criproxy.RuntimeConfig {
				runtimeConfigCalls.Add(1)
				return status.Error(codes.Unimplemented, "RuntimeConfig is not implemented")
			}
			return nil
		}))

		ginkgo.By("rejecting the unsupported runtime by default")
		restartKubelet(ctx, true)
		gomega.Eventually(ctx, runtimeConfigCalls.Load, time.Minute, time.Second).Should(gomega.BeNumerically(">", 0))
		gomega.Consistently(ctx, func() bool {
			return e2enode.HealthCheck(kubeletHealthCheckURL)
		}, 15*time.Second, time.Second).Should(gomega.BeFalseBecause("kubelet must not start without RuntimeConfig support"))

		ginkgo.By("recovering with the temporary fallback enabled")
		if config.FeatureGates == nil {
			config.FeatureGates = map[string]bool{}
		}
		config.FeatureGates[string(features.DisableCgroupDriverFallback)] = false
		config.CgroupDriver = cgroupDriver
		// The unhealthy kubelet cannot serve configz or use the usual config update helper.
		framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(config))
		restartKubelet(ctx, false)
		waitForKubeletToStart(ctx, f)

		m, err := e2emetrics.GrabKubeletMetricsWithoutProxy(ctx, nodeNameOrIP()+":10255", "/metrics")
		framework.ExpectNoError(err)
		samples := m[kubeletmetrics.KubeletSubsystem+"_"+kubeletmetrics.CRILosingSupportKey]
		gomega.Expect(samples).NotTo(gomega.BeEmpty())
		gomega.Expect(samples[0].Metric["version"]).To(gomega.BeEquivalentTo("1.38.0"))
	})

	ginkgo.It("should not emit the unsupported runtime metric when RuntimeConfig is implemented", func(ctx context.Context) {
		restartKubelet(ctx, true)
		waitForKubeletToStart(ctx, f)
		m, err := e2emetrics.GrabKubeletMetricsWithoutProxy(ctx, nodeNameOrIP()+":10255", "/metrics")
		framework.ExpectNoError(err)
		gomega.Expect(m[kubeletmetrics.KubeletSubsystem+"_"+kubeletmetrics.CRILosingSupportKey]).To(gomega.BeEmpty())
	})
})
