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

package e2e_node

import (
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
)

const (
	defaultObservationTimeout = time.Minute * 4
)

var _ = framework.KubeDescribe("StartupProbe [Serial] [Disruptive] [NodeFeature:StartupProbe]", func() {
	f := framework.NewDefaultFramework("critical-pod-test")

	/*
		Release : v1.15
		Testname: Pod liveness probe, using local file, delayed by startup probe
		Description: A Pod is created with liveness probe that uses ‘exec’ command to cat /temp/health file. Liveness probe MUST not fail until startup probe succeeds.

		This test is located here as it requires tempSetCurrentKubeletConfig
		to enable the feature gate for startupProbe, once removed test should come back to test/e2e/common/container_probe.go.
	*/
	Context("when a container has a startup probe", func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates[string(features.StartupProbeEnabled)] = true
		})

		It("should *not* be restarted with a exec \"cat /tmp/health\" because startup probe delays it", func() {
			common.RunLivenessTest(f, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "liveness-exec",
					Labels: map[string]string{"test": "liveness"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "liveness",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh", "-c", "sleep 600"},
							LivenessProbe: &v1.Probe{
								Handler: v1.Handler{
									Exec: &v1.ExecAction{
										Command: []string{"cat", "/tmp/health"},
									},
								},
								InitialDelaySeconds: 15,
								FailureThreshold:    1,
							},
							StartupProbe: &v1.Probe{
								Handler: v1.Handler{
									Exec: &v1.ExecAction{
										Command: []string{"cat", "/tmp/health"},
									},
								},
								InitialDelaySeconds: 15,
								FailureThreshold:    60,
							},
						},
					},
				},
			}, 0, defaultObservationTimeout)
		})
	})
})
