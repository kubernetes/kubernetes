/*
Copyright 2023 The Kubernetes Authors.

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
	"os"
	"path/filepath"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/nodefeature"
)

var _ = SIGDescribe("Kubelet Config", framework.WithSlow(), framework.WithSerial(), framework.WithDisruptive(), nodefeature.KubeletConfigDropInDir, feature.KubeletConfigDropInDir, func() {
	f := framework.NewDefaultFramework("kubelet-config-drop-in-dir-test")
	ginkgo.Context("when merging drop-in configs", func() {
		var oldcfg *kubeletconfig.KubeletConfiguration
		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			oldcfg, err = getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)
		})
		ginkgo.AfterEach(func(ctx context.Context) {
			files, err := filepath.Glob(filepath.Join(framework.TestContext.KubeletConfigDropinDir, "*"+".conf"))
			framework.ExpectNoError(err)
			for _, file := range files {
				err := os.Remove(file)
				framework.ExpectNoError(err)
			}
			updateKubeletConfig(ctx, f, oldcfg, true)
		})
		ginkgo.It("should merge kubelet configs correctly", func(ctx context.Context) {
			// Get the initial kubelet configuration
			initialConfig, err := getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)

			ginkgo.By("Stopping the kubelet")
			restartKubelet := mustStopKubelet(ctx, f)

			configDir := framework.TestContext.KubeletConfigDropinDir

			contents := []byte(`apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
port: 10255
readOnlyPort: 10257
clusterDNS:
- 192.168.1.10
systemReserved:
  memory: 1Gi
authorization:
  mode: Webhook
  webhook:
    cacheAuthorizedTTL: "5m"
    cacheUnauthorizedTTL: "30s"
staticPodURLHeader:
  kubelet-api-support:
  - "Authorization: 234APSDFA"
  - "X-Custom-Header: 123"
  custom-static-pod:
  - "Authorization: 223EWRWER"
  - "X-Custom-Header: 456"
shutdownGracePeriodByPodPriority:
  - priority: 1
    shutdownGracePeriodSeconds: 60
  - priority: 2
    shutdownGracePeriodSeconds: 45
  - priority: 3
    shutdownGracePeriodSeconds: 30
featureGates:
  DisableKubeletCloudCredentialProviders: true
  PodAndContainerStatsFromCRI: true`)
			framework.ExpectNoError(os.WriteFile(filepath.Join(configDir, "10-kubelet.conf"), contents, 0755))
			contents = []byte(`apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
clusterDNS:
- 192.168.1.1
- 192.168.1.5
- 192.168.1.8
port: 8080
cpuManagerReconcilePeriod: 1s
systemReserved:
  memory: 2Gi
authorization:
  mode: Webhook
  webhook:
    cacheAuthorizedTTL: "6m"
    cacheUnauthorizedTTL: "40s"
staticPodURLHeader:
  kubelet-api-support:
  - "Authorization: 8945AFSG1"
  - "X-Custom-Header: 987"
  custom-static-pod:
  - "Authorization: 223EWRWER"
  - "X-Custom-Header: 345"
shutdownGracePeriodByPodPriority:
  - priority: 1
    shutdownGracePeriodSeconds: 19
  - priority: 2
    shutdownGracePeriodSeconds: 41
  - priority: 6
    shutdownGracePeriodSeconds: 30
featureGates:
  PodAndContainerStatsFromCRI: false
  DynamicResourceAllocation: true`)
			framework.ExpectNoError(os.WriteFile(filepath.Join(configDir, "20-kubelet.conf"), contents, 0755))
			ginkgo.By("Restarting the kubelet")
			restartKubelet(ctx)
			// wait until the kubelet health check will succeed
			gomega.Eventually(ctx, func() bool {
				return kubeletHealthCheck(kubeletHealthCheckURL)
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeTrueBecause("expected kubelet to be in healthy state"))

			mergedConfig, err := getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)

			// Replace specific fields in the initial configuration with expectedConfig values
			initialConfig.Port = int32(8080)                  // not overridden by second file, should be retained.
			initialConfig.ReadOnlyPort = int32(10257)         // overridden by second file.
			initialConfig.SystemReserved = map[string]string{ // overridden by map in second file.
				"memory": "2Gi",
			}
			initialConfig.ClusterDNS = []string{"192.168.1.1", "192.168.1.5", "192.168.1.8"} // overridden by slice in second file.
			// This value was explicitly set in the drop-in, make sure it is retained
			initialConfig.CPUManagerReconcilePeriod = metav1.Duration{Duration: time.Second}
			// Meanwhile, this value was not explicitly set, but could have been overridden by a "default" of 0 for the type.
			// Ensure the true default persists.
			initialConfig.CPUCFSQuotaPeriod = metav1.Duration{Duration: time.Duration(100000000)}
			// This covers the case for a map with the list of values.
			initialConfig.StaticPodURLHeader = map[string][]string{
				"kubelet-api-support": {"Authorization: 8945AFSG1", "X-Custom-Header: 987"},
				"custom-static-pod":   {"Authorization: 223EWRWER", "X-Custom-Header: 345"},
			}
			// This covers the case where the fields within the list of structs are overridden.
			initialConfig.ShutdownGracePeriodByPodPriority = []kubeletconfig.ShutdownGracePeriodByPodPriority{
				{Priority: 1, ShutdownGracePeriodSeconds: 19},
				{Priority: 2, ShutdownGracePeriodSeconds: 41},
				{Priority: 6, ShutdownGracePeriodSeconds: 30},
			}
			// This covers the case where the fields within the struct are overridden.
			initialConfig.Authorization = kubeletconfig.KubeletAuthorization{
				Mode: "Webhook",
				Webhook: kubeletconfig.KubeletWebhookAuthorization{
					CacheAuthorizedTTL:   metav1.Duration{Duration: time.Duration(6 * time.Minute)},
					CacheUnauthorizedTTL: metav1.Duration{Duration: time.Duration(40 * time.Second)},
				},
			}
			// This covers the case where the fields within the map are overridden.
			overrides := map[string]bool{"DisableKubeletCloudCredentialProviders": true, "PodAndContainerStatsFromCRI": false, "DynamicResourceAllocation": true}
			// In some CI jobs, `NodeSwap` is explicitly disabled as the images are cgroupv1 based,
			// so such flags should be picked up directly from the initial configuration
			if _, ok := initialConfig.FeatureGates["NodeSwap"]; ok {
				overrides["NodeSwap"] = initialConfig.FeatureGates["NodeSwap"]
			}
			initialConfig.FeatureGates = overrides
			// Compare the expected config with the merged config
			gomega.Expect(initialConfig).To(gomega.BeComparableTo(mergedConfig), "Merged kubelet config does not match the expected configuration.")
		})
	})

})
