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
	"fmt"

	v1 "k8s.io/api/core/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/security/apparmor"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/seccomp"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/auth"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epsp "k8s.io/kubernetes/test/e2e/framework/psp"
	imageutils "k8s.io/kubernetes/test/utils/image"
	utilpointer "k8s.io/utils/pointer"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const nobodyUser = int64(65534)

var _ = SIGDescribe("PodSecurityPolicy", func() {
	f := framework.NewDefaultFramework("podsecuritypolicy")
	f.SkipPrivilegedPSPBinding = true

	// Client that will impersonate the default service account, in order to run
	// with reduced privileges.
	var c clientset.Interface
	var ns string // Test namespace, for convenience
	ginkgo.BeforeEach(func() {
		if !e2epsp.IsPodSecurityPolicyEnabled(f.ClientSet) {
			framework.Skipf("PodSecurityPolicy not enabled")
		}
		if !auth.IsRBACEnabled(f.ClientSet.RbacV1()) {
			framework.Skipf("RBAC not enabled")
		}
		ns = f.Namespace.Name

		ginkgo.By("Creating a kubernetes client that impersonates the default service account")
		config, err := framework.LoadConfig()
		framework.ExpectNoError(err)
		config.Impersonate = restclient.ImpersonationConfig{
			UserName: serviceaccount.MakeUsername(ns, "default"),
			Groups:   serviceaccount.MakeGroupNames(ns),
		}
		c, err = clientset.NewForConfig(config)
		framework.ExpectNoError(err)

		ginkgo.By("Binding the edit role to the default SA")
		err = auth.BindClusterRole(f.ClientSet.RbacV1(), "edit", ns,
			rbacv1.Subject{Kind: rbacv1.ServiceAccountKind, Namespace: ns, Name: "default"})
		framework.ExpectNoError(err)
	})

	ginkgo.It("should forbid pod creation when no PSP is available", func() {
		ginkgo.By("Running a restricted pod")
		_, err := c.CoreV1().Pods(ns).Create(restrictedPod("restricted"))
		expectForbidden(err)
	})

	ginkgo.It("should enforce the restricted policy.PodSecurityPolicy", func() {
		ginkgo.By("Creating & Binding a restricted policy for the test service account")
		_, cleanup := createAndBindPSP(f, restrictedPSP("restrictive"))
		defer cleanup()

		ginkgo.By("Running a restricted pod")
		pod, err := c.CoreV1().Pods(ns).Create(restrictedPod("allowed"))
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(c, pod.Name, pod.Namespace))

		testPrivilegedPods(func(pod *v1.Pod) {
			_, err := c.CoreV1().Pods(ns).Create(pod)
			expectForbidden(err)
		})
	})

	ginkgo.It("should allow pods under the privileged policy.PodSecurityPolicy", func() {
		ginkgo.By("Creating & Binding a privileged policy for the test service account")
		// Ensure that the permissive policy is used even in the presence of the restricted policy.
		_, cleanup := createAndBindPSP(f, restrictedPSP("restrictive"))
		defer cleanup()
		expectedPSP, cleanup := createAndBindPSP(f, privilegedPSP("permissive"))
		defer cleanup()

		testPrivilegedPods(func(pod *v1.Pod) {
			p, err := c.CoreV1().Pods(ns).Create(pod)
			framework.ExpectNoError(err)
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(c, p.Name, p.Namespace))

			// Verify expected PSP was used.
			p, err = c.CoreV1().Pods(ns).Get(p.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			validated, found := p.Annotations[psputil.ValidatedPSPAnnotation]
			gomega.Expect(found).To(gomega.BeTrue(), "PSP annotation not found")
			framework.ExpectEqual(validated, expectedPSP.Name, "Unexpected validated PSP")
		})
	})
})

func expectForbidden(err error) {
	framework.ExpectError(err, "should be forbidden")
	gomega.Expect(apierrs.IsForbidden(err)).To(gomega.BeTrue(), "should be forbidden error")
}

func testPrivilegedPods(tester func(pod *v1.Pod)) {
	ginkgo.By("Running a privileged pod", func() {
		privileged := restrictedPod("privileged")
		privileged.Spec.Containers[0].SecurityContext.Privileged = boolPtr(true)
		privileged.Spec.Containers[0].SecurityContext.AllowPrivilegeEscalation = nil
		tester(privileged)
	})

	ginkgo.By("Running a HostPath pod", func() {
		hostpath := restrictedPod("hostpath")
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

	ginkgo.By("Running a HostNetwork pod", func() {
		hostnet := restrictedPod("hostnet")
		hostnet.Spec.HostNetwork = true
		tester(hostnet)
	})

	ginkgo.By("Running a HostPID pod", func() {
		hostpid := restrictedPod("hostpid")
		hostpid.Spec.HostPID = true
		tester(hostpid)
	})

	ginkgo.By("Running a HostIPC pod", func() {
		hostipc := restrictedPod("hostipc")
		hostipc.Spec.HostIPC = true
		tester(hostipc)
	})

	if common.IsAppArmorSupported() {
		ginkgo.By("Running a custom AppArmor profile pod", func() {
			aa := restrictedPod("apparmor")
			// Every node is expected to have the docker-default profile.
			aa.Annotations[apparmor.ContainerAnnotationKeyPrefix+"pause"] = "localhost/docker-default"
			tester(aa)
		})
	}

	ginkgo.By("Running an unconfined Seccomp pod", func() {
		unconfined := restrictedPod("seccomp")
		unconfined.Annotations[v1.SeccompPodAnnotationKey] = "unconfined"
		tester(unconfined)
	})

	ginkgo.By("Running a SYS_ADMIN pod", func() {
		sysadmin := restrictedPod("sysadmin")
		sysadmin.Spec.Containers[0].SecurityContext.Capabilities = &v1.Capabilities{
			Add: []v1.Capability{"SYS_ADMIN"},
		}
		sysadmin.Spec.Containers[0].SecurityContext.AllowPrivilegeEscalation = nil
		tester(sysadmin)
	})

	ginkgo.By("Running a RunAsGroup pod", func() {
		sysadmin := restrictedPod("runasgroup")
		gid := int64(0)
		sysadmin.Spec.Containers[0].SecurityContext.RunAsGroup = &gid
		tester(sysadmin)
	})

	ginkgo.By("Running a RunAsUser pod", func() {
		sysadmin := restrictedPod("runasuser")
		uid := int64(0)
		sysadmin.Spec.Containers[0].SecurityContext.RunAsUser = &uid
		tester(sysadmin)
	})
}

// createAndBindPSP creates a PSP in the policy API group.
func createAndBindPSP(f *framework.Framework, pspTemplate *policyv1beta1.PodSecurityPolicy) (psp *policyv1beta1.PodSecurityPolicy, cleanup func()) {
	// Create the PodSecurityPolicy object.
	psp = pspTemplate.DeepCopy()
	// Add the namespace to the name to ensure uniqueness and tie it to the namespace.
	ns := f.Namespace.Name
	name := fmt.Sprintf("%s-%s", ns, psp.Name)
	psp.Name = name
	psp, err := f.ClientSet.PolicyV1beta1().PodSecurityPolicies().Create(psp)
	framework.ExpectNoError(err, "Failed to create PSP")

	// Create the Role to bind it to the namespace.
	_, err = f.ClientSet.RbacV1().Roles(ns).Create(&rbacv1.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Rules: []rbacv1.PolicyRule{{
			APIGroups:     []string{"policy"},
			Resources:     []string{"podsecuritypolicies"},
			ResourceNames: []string{name},
			Verbs:         []string{"use"},
		}},
	})
	framework.ExpectNoError(err, "Failed to create PSP role")

	// Bind the role to the namespace.
	err = auth.BindRoleInNamespace(f.ClientSet.RbacV1(), name, ns, rbacv1.Subject{
		Kind:      rbacv1.ServiceAccountKind,
		Namespace: ns,
		Name:      "default",
	})
	framework.ExpectNoError(err)

	framework.ExpectNoError(auth.WaitForNamedAuthorizationUpdate(f.ClientSet.AuthorizationV1(),
		serviceaccount.MakeUsername(ns, "default"), ns, "use", name,
		schema.GroupResource{Group: "policy", Resource: "podsecuritypolicies"}, true))

	return psp, func() {
		// Cleanup non-namespaced PSP object.
		f.ClientSet.PolicyV1beta1().PodSecurityPolicies().Delete(name, &metav1.DeleteOptions{})
	}
}

func restrictedPod(name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Annotations: map[string]string{
				v1.SeccompPodAnnotationKey:                      v1.SeccompProfileRuntimeDefault,
				apparmor.ContainerAnnotationKeyPrefix + "pause": apparmor.ProfileRuntimeDefault,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "pause",
				Image: imageutils.GetPauseImageName(),
				SecurityContext: &v1.SecurityContext{
					AllowPrivilegeEscalation: boolPtr(false),
					RunAsUser:                utilpointer.Int64Ptr(nobodyUser),
					RunAsGroup:               utilpointer.Int64Ptr(nobodyUser),
				},
			}},
		},
	}
}

// privilegedPSPInPolicy creates a PodSecurityPolicy (in the "policy" API Group) that allows everything.
func privilegedPSP(name string) *policyv1beta1.PodSecurityPolicy {
	return &policyv1beta1.PodSecurityPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Annotations: map[string]string{seccomp.AllowedProfilesAnnotationKey: seccomp.AllowAny},
		},
		Spec: policyv1beta1.PodSecurityPolicySpec{
			Privileged:               true,
			AllowPrivilegeEscalation: utilpointer.BoolPtr(true),
			AllowedCapabilities:      []v1.Capability{"*"},
			Volumes:                  []policyv1beta1.FSType{policyv1beta1.All},
			HostNetwork:              true,
			HostPorts:                []policyv1beta1.HostPortRange{{Min: 0, Max: 65535}},
			HostIPC:                  true,
			HostPID:                  true,
			RunAsUser: policyv1beta1.RunAsUserStrategyOptions{
				Rule: policyv1beta1.RunAsUserStrategyRunAsAny,
			},
			RunAsGroup: &policyv1beta1.RunAsGroupStrategyOptions{
				Rule: policyv1beta1.RunAsGroupStrategyRunAsAny,
			},
			SELinux: policyv1beta1.SELinuxStrategyOptions{
				Rule: policyv1beta1.SELinuxStrategyRunAsAny,
			},
			SupplementalGroups: policyv1beta1.SupplementalGroupsStrategyOptions{
				Rule: policyv1beta1.SupplementalGroupsStrategyRunAsAny,
			},
			FSGroup: policyv1beta1.FSGroupStrategyOptions{
				Rule: policyv1beta1.FSGroupStrategyRunAsAny,
			},
			ReadOnlyRootFilesystem: false,
		},
	}
}

// restrictedPSPInPolicy creates a PodSecurityPolicy (in the "policy" API Group) that is most strict.
func restrictedPSP(name string) *policyv1beta1.PodSecurityPolicy {
	return &policyv1beta1.PodSecurityPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Annotations: map[string]string{
				seccomp.AllowedProfilesAnnotationKey:  v1.SeccompProfileRuntimeDefault,
				seccomp.DefaultProfileAnnotationKey:   v1.SeccompProfileRuntimeDefault,
				apparmor.AllowedProfilesAnnotationKey: apparmor.ProfileRuntimeDefault,
				apparmor.DefaultProfileAnnotationKey:  apparmor.ProfileRuntimeDefault,
			},
		},
		Spec: policyv1beta1.PodSecurityPolicySpec{
			Privileged:               false,
			AllowPrivilegeEscalation: utilpointer.BoolPtr(false),
			RequiredDropCapabilities: []v1.Capability{
				"AUDIT_WRITE",
				"CHOWN",
				"DAC_OVERRIDE",
				"FOWNER",
				"FSETID",
				"KILL",
				"MKNOD",
				"NET_RAW",
				"SETGID",
				"SETUID",
				"SYS_CHROOT",
			},
			Volumes: []policyv1beta1.FSType{
				policyv1beta1.ConfigMap,
				policyv1beta1.EmptyDir,
				policyv1beta1.PersistentVolumeClaim,
				"projected",
				policyv1beta1.Secret,
			},
			HostNetwork: false,
			HostIPC:     false,
			HostPID:     false,
			RunAsUser: policyv1beta1.RunAsUserStrategyOptions{
				Rule: policyv1beta1.RunAsUserStrategyMustRunAsNonRoot,
			},
			RunAsGroup: &policyv1beta1.RunAsGroupStrategyOptions{
				Rule: policyv1beta1.RunAsGroupStrategyMustRunAs,
				Ranges: []policyv1beta1.IDRange{
					{Min: nobodyUser, Max: nobodyUser}},
			},
			SELinux: policyv1beta1.SELinuxStrategyOptions{
				Rule: policyv1beta1.SELinuxStrategyRunAsAny,
			},
			SupplementalGroups: policyv1beta1.SupplementalGroupsStrategyOptions{
				Rule: policyv1beta1.SupplementalGroupsStrategyRunAsAny,
			},
			FSGroup: policyv1beta1.FSGroupStrategyOptions{
				Rule: policyv1beta1.FSGroupStrategyRunAsAny,
			},
			ReadOnlyRootFilesystem: false,
		},
	}
}

func boolPtr(b bool) *bool {
	return &b
}
