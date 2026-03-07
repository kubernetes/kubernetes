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

package node

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = ginkgo.Describe("[sig-node] ContainerTerminationMetrics Validation", func() {
	f := framework.NewDefaultFramework("metric-validation")

	ginkgo.It("should report terminated_containers_total metric", func(ctx context.Context) {
		f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "sigsegv-test-pod"},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				SecurityContext: &v1.PodSecurityContext{
					RunAsUser:      &[]int64{1000}[0],
					RunAsNonRoot:   &[]bool{true}[0],
					SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeRuntimeDefault},
				},
				Containers: []v1.Container{{
					Name:    "crash-container",
					Image:   "alpine:latest",
					Command: []string{"/bin/sh", "-c", "exit 139"},
					SecurityContext: &v1.SecurityContext{
						AllowPrivilegeEscalation: &[]bool{false}[0],
						Capabilities:             &v1.Capabilities{Drop: []v1.Capability{"ALL"}},
					},
				}},
			},
		}

		createdPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, podSpec, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Checking kubelet metrics via API Server Proxy")
		gomega.Eventually(ctx, func() error {
			// Get the actual node the pod was scheduled on
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, createdPod.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			if p.Spec.NodeName == "" {
				return fmt.Errorf("pod not yet scheduled")
			}

			metrics, err := e2emetrics.GetKubeletMetrics(ctx, f.ClientSet, p.Spec.NodeName)
			if err != nil {
				return err
			}
			if samples, found := metrics["terminated_containers_total"]; found {
				for _, sample := range samples {
					if string(sample.Metric["container_type"]) == "container" &&
						string(sample.Metric["exit_code"]) == "139" {
						framework.Logf("Found metric! Value: %v", sample.Value)
						return nil
					}
				}
			}
			return fmt.Errorf("metric not found")
		}, 4*time.Minute, 5*time.Second).Should(gomega.Succeed())
	})
})
