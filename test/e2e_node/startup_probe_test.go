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
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	defaultObservationTimeout = time.Minute * 4
)

var _ = framework.KubeDescribe("StartupProbe [Serial] [Disruptive] [NodeFeature:StartupProbe]", func() {
	f := framework.NewDefaultFramework("critical-pod-test")
	var podClient *framework.PodClient

	/*
		These tests are located here as they require tempSetCurrentKubeletConfig to enable the feature gate for startupProbe.
		Once the feature gate has been removed, these tests should come back to test/e2e/common/container_probe.go.
	*/
	ginkgo.Context("when a container has a startup probe", func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates[string(features.StartupProbe)] = true
		})

		/*
			Release : v1.16
			Testname: Pod liveness probe, using local file, delayed by startup probe
			Description: A Pod is created with liveness probe that uses ‘exec’ command to cat the non-existent /tmp/health file. Liveness probe MUST NOT fail until startup probe expires.
		*/
		framework.ConformanceIt("should *not* be restarted with a exec \"cat /tmp/health\" because startup probe delays it [NodeConformance]", func() {
			cmd := []string{"/bin/sh", "-c", "sleep 600"}
			livenessProbe := &v1.Probe{
				Handler: v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"cat", "/tmp/health"},
					},
				},
				InitialDelaySeconds: 15,
				FailureThreshold:    1,
			}
			startupProbe := &v1.Probe{
				Handler: v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"cat", "/tmp/health"},
					},
				},
				InitialDelaySeconds: 15,
				FailureThreshold:    60,
			}
			pod := startupPodSpec(startupProbe, nil, livenessProbe, cmd)
			common.RunLivenessTest(f, pod, 0, defaultObservationTimeout)
		})

		/*
			Release : v1.16
			Testname: Pod liveness probe, using local file, delayed by startup probe
			Description: A Pod is created with liveness probe that uses ‘exec’ command to cat the non-existent /tmp/health file. Liveness probe MUST fail after startup probe expires. The Pod MUST now be killed and restarted incrementing restart count to 1.
		*/
		framework.ConformanceIt("should be restarted with a exec \"cat /tmp/health\" because startup probe does not delay it long enough [NodeConformance]", func() {
			cmd := []string{"/bin/sh", "-c", "sleep 600"}
			livenessProbe := &v1.Probe{
				Handler: v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"cat", "/tmp/health"},
					},
				},
				InitialDelaySeconds: 15,
				FailureThreshold:    1,
			}
			startupProbe := &v1.Probe{
				Handler: v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"cat", "/tmp/health"},
					},
				},
				InitialDelaySeconds: 15,
				FailureThreshold:    3,
			}
			pod := startupPodSpec(startupProbe, nil, livenessProbe, cmd)
			common.RunLivenessTest(f, pod, 1, defaultObservationTimeout)
		})

		/*
			Release : v1.16
			Testname: Pod liveness probe, using local file, startup finished restart
			Description: A Pod is created with liveness probe that uses ‘exec’ command to cat /temp/health file. The Container is started by creating /tmp/startup after 10 seconds, triggering liveness probe to fail. The Pod MUST now be killed and restarted incrementing restart count to 1.
		*/
		framework.ConformanceIt("should be restarted with a exec \"cat /tmp/health\" after startup probe succeeds it [NodeConformance]", func() {
			cmd := []string{"/bin/sh", "-c", "sleep 10; echo ok >/tmp/startup; sleep 600"}
			livenessProbe := &v1.Probe{
				Handler: v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"cat", "/tmp/health"},
					},
				},
				InitialDelaySeconds: 15,
				FailureThreshold:    1,
			}
			startupProbe := &v1.Probe{
				Handler: v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"cat", "/tmp/startup"},
					},
				},
				InitialDelaySeconds: 15,
				FailureThreshold:    60,
			}
			pod := startupPodSpec(startupProbe, nil, livenessProbe, cmd)
			common.RunLivenessTest(f, pod, 1, defaultObservationTimeout)
		})

		/*
			Release : v1.16
			Testname: Pod readiness probe, delayed by startup probe
			Description: A Pod is created with startup and readiness probes. The Container is started by creating /tmp/startup after 45 seconds, delaying the ready state by this amount of time. This is similar to the "Pod readiness probe, with initial delay" test.
		*/
		framework.ConformanceIt("should not be ready until startupProbe succeeds [NodeConformance]", func() {
			cmd := []string{"/bin/sh", "-c", "echo ok >/tmp/health; sleep 45; echo ok >/tmp/startup; sleep 600"}
			readinessProbe := &v1.Probe{
				Handler: v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"cat", "/tmp/health"},
					},
				},
				InitialDelaySeconds: 0,
			}
			startupProbe := &v1.Probe{
				Handler: v1.Handler{
					Exec: &v1.ExecAction{
						Command: []string{"cat", "/tmp/startup"},
					},
				},
				InitialDelaySeconds: 0,
				FailureThreshold:    60,
			}
			p := podClient.Create(startupPodSpec(startupProbe, readinessProbe, nil, cmd))

			p, err := podClient.Get(p.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			f.WaitForPodReady(p.Name)
			isReady, err := testutils.PodRunningReady(p)
			framework.ExpectNoError(err)
			gomega.Expect(isReady).To(gomega.BeTrue(), "pod should be ready")

			// We assume the pod became ready when the container became ready. This
			// is true for a single container pod.
			readyTime, err := common.GetTransitionTimeForReadyCondition(p)
			framework.ExpectNoError(err)
			startedTime, err := common.GetContainerStartedTime(p, "busybox")
			framework.ExpectNoError(err)

			e2elog.Logf("Container started at %v, pod became ready at %v", startedTime, readyTime)
			if readyTime.Sub(startedTime) < 40*time.Second {
				e2elog.Failf("Pod became ready before startupProbe succeeded")
			}
		})
	})
})

func startupPodSpec(startupProbe, readinessProbe, livenessProbe *v1.Probe, cmd []string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "startup-" + string(uuid.NewUUID()),
			Labels: map[string]string{"test": "startup"},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:           "busybox",
					Image:          imageutils.GetE2EImage(imageutils.BusyBox),
					Command:        cmd,
					LivenessProbe:  livenessProbe,
					ReadinessProbe: readinessProbe,
					StartupProbe:   startupProbe,
				},
			},
		},
	}
}
