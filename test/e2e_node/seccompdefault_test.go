//go:build linux
// +build linux

/*
Copyright 2022 The Kubernetes Authors.

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
	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("SeccompDefault [Serial] [Feature:SeccompDefault] [LinuxOnly]", func() {
	f := framework.NewDefaultFramework("seccompdefault-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	ginkgo.Context("with SeccompDefault enabled", func() {
		tempSetCurrentKubeletConfig(f, func(cfg *kubeletconfig.KubeletConfiguration) {
			cfg.SeccompDefault = true
		})

		newPod := func(securityContext *v1.SecurityContext) *v1.Pod {
			name := "seccompdefault-test-" + string(uuid.NewUUID())
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:            name,
							Image:           busyboxImage,
							Command:         []string{"grep", "Seccomp:", "/proc/self/status"},
							SecurityContext: securityContext,
						},
					},
				},
			}
		}

		ginkgo.It("should use the default seccomp profile when unspecified", func() {
			pod := newPod(nil)
			f.TestContainerOutput("SeccompDefault", pod, 0, []string{"2"})
		})

		ginkgo.It("should use unconfined when specified", func() {
			pod := newPod(&v1.SecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeUnconfined}})
			f.TestContainerOutput("SeccompDefault-unconfined", pod, 0, []string{"0"})
		})
	})
})
