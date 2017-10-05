/*
Copyright 2017 The Kubernetes Authors.

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

package auth

import (
	"k8s.io/api/core/v1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/security/apparmor"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = SIGDescribe("PodSecurityPolicy", func() {
	f := framework.NewDefaultFramework("podsecuritypolicy")
	f.SkipDefaultPSPBinding = true

	// Client that will impersonate the default service account, in order to run
	// with reduced privileges.
	var c clientset.Interface
	var ns string // Test namespace, for convenience
	BeforeEach(func() {
		if !framework.IsPodSecurityPolicyEnabled(f) {
			framework.Skipf("PodSecurityPolicy not enabled")
		}
		ns = f.Namespace.Name

		By("Creating a kubernetes client that impersonates the default service account")
		config, err := framework.LoadConfig()
		framework.ExpectNoError(err)
		config.Impersonate = restclient.ImpersonationConfig{
			UserName: serviceaccount.MakeUsername(ns, "default"),
			Groups:   serviceaccount.MakeGroupNames(ns),
		}
		c, err = clientset.NewForConfig(config)
		framework.ExpectNoError(err)

		By("Binding the edit role to the default SA")
		framework.BindClusterRole(f.ClientSet.RbacV1beta1(), "edit", ns,
			rbacv1beta1.Subject{Kind: rbacv1beta1.ServiceAccountKind, Namespace: ns, Name: "default"})
	})

	It("should forbid pod creation when no PSP is available", func() {
		By("Running a restricted pod")
		_, err := c.Core().Pods(ns).Create(restrictedPod(f, "restricted"))
		expectForbidden(err)
	})

	It("should enforce the restricted PodSecurityPolicy", func() {
		By("Binding the restricted policy for the test service account")
		framework.BindClusterRole(f.ClientSet.RbacV1beta1(), "podsecuritypolicy:restricted:v1", ns,
			rbacv1beta1.Subject{Kind: rbacv1beta1.ServiceAccountKind, Namespace: ns, Name: "default"})

		By("Running a restricted pod")
		pod, err := c.Core().Pods(ns).Create(restrictedPod(f, "allowed"))
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodNameRunningInNamespace(c, pod.Name, pod.Namespace))

		testPrivilegedPods(f, func(pod *v1.Pod) {
			_, err := c.Core().Pods(ns).Create(pod)
			expectForbidden(err)
		})
	})

	It("should allow pods under the privileged PodSecurityPolicy", func() {
		By("Binding the privileged policy for the test service account")
		framework.BindClusterRole(f.ClientSet.RbacV1beta1(), "podsecuritypolicy:privileged", ns,
			rbacv1beta1.Subject{Kind: rbacv1beta1.ServiceAccountKind, Namespace: ns, Name: "default"})

		testPrivilegedPods(f, func(pod *v1.Pod) {
			p, err := c.Core().Pods(ns).Create(pod)
			framework.ExpectNoError(err)
			framework.ExpectNoError(framework.WaitForPodNameRunningInNamespace(c, p.Name, p.Namespace))
		})
	})
})

func expectForbidden(err error) {
	Expect(err).To(HaveOccurred(), "should be forbidden")
	Expect(apierrs.IsForbidden(err)).To(BeTrue(), "should be forbidden error")
}

func testPrivilegedPods(f *framework.Framework, tester func(pod *v1.Pod)) {
	By("Running a privileged pod", func() {
		privileged := restrictedPod(f, "privileged")
		privileged.Spec.Containers[0].SecurityContext.Privileged = boolPtr(true)
		tester(privileged)
	})

	By("Running a HostPath pod", func() {
		hostpath := restrictedPod(f, "hostpath")
		hostpath.Spec.Containers[0].VolumeMounts = []v1.VolumeMount{{
			Name:      "hp",
			MountPath: "/hp",
		}}
		hostpath.Spec.Volumes = []v1.Volume{{
			Name: "hp",
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "/tmp"},
			},
		}}
		tester(hostpath)
	})

	By("Running a HostNetwork pod", func() {
		hostnet := restrictedPod(f, "hostnet")
		hostnet.Spec.HostNetwork = true
		tester(hostnet)
	})

	By("Running a HostPID pod", func() {
		hostpid := restrictedPod(f, "hostpid")
		hostpid.Spec.HostPID = true
		tester(hostpid)
	})

	By("Running a HostIPC pod", func() {
		hostipc := restrictedPod(f, "hostipc")
		hostipc.Spec.HostIPC = true
		tester(hostipc)
	})

	if common.IsAppArmorSupported() {
		By("Running a custom AppArmor profile pod", func() {
			aa := restrictedPod(f, "apparmor")
			// Every node is expected to have the docker-default profile.
			aa.Annotations[apparmor.ContainerAnnotationKeyPrefix+"pause"] = "localhost/docker-default"
			tester(aa)
		})
	}

	By("Running an unconfined Seccomp pod", func() {
		unconfined := restrictedPod(f, "seccomp")
		unconfined.Annotations[api.SeccompPodAnnotationKey] = "unconfined"
		tester(unconfined)
	})

	By("Running a CAP_SYS_ADMIN pod", func() {
		sysadmin := restrictedPod(f, "sysadmin")
		sysadmin.Spec.Containers[0].SecurityContext.Capabilities = &v1.Capabilities{
			Add: []v1.Capability{"CAP_SYS_ADMIN"},
		}
		tester(sysadmin)
	})
}

func restrictedPod(f *framework.Framework, name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Annotations: map[string]string{
				api.SeccompPodAnnotationKey:                     "docker/default",
				apparmor.ContainerAnnotationKeyPrefix + "pause": apparmor.ProfileRuntimeDefault,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "pause",
				Image: framework.GetPauseImageName(f.ClientSet),
				SecurityContext: &v1.SecurityContext{
					AllowPrivilegeEscalation: boolPtr(false),
					RunAsUser:                intPtr(65534),
				},
			}},
		},
	}
}

func boolPtr(b bool) *bool {
	return &b
}

func intPtr(i int64) *int64 {
	return &i
}
