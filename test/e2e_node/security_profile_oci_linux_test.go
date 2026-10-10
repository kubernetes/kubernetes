/*
Copyright The Kubernetes Authors.

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
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/uuid"
	crierrors "k8s.io/cri-api/pkg/errors"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	"k8s.io/kubernetes/test/e2e_node/criproxy"
	admissionapi "k8s.io/pod-security-admission/api"
)

// Seccomp profile artifacts published by the Security Profiles Operator for
// these tests, pinned by digest.
const (
	securityProfileRepository = "registry.k8s.io/security-profiles-operator/seccomp-test-profiles"
	// Denies chmod, fchmod and fchmodat with EPERM and allows everything else.
	denyChmodProfile = securityProfileRepository + "@sha256:9b9d6cb98e5c58448c341628db6b20e7c4866abc53fbd5e1f5a938739b26fbe6"
	// Allows everything; the runtime baseline still applies.
	permissiveProfile = securityProfileRepository + "@sha256:e7279f26947707f3328552d72167700d80431179f3023e8efaa7a86f08dc5176"
	// A digest that the repository does not serve.
	missingProfile = securityProfileRepository + "@sha256:0000000000000000000000000000000000000000000000000000000000000000"

	securityProfileRejectedReason = "SecurityProfileRejected"
)

var _ = SIGDescribe("SecurityProfileOCI", feature.SecurityProfileOCI, framework.WithFeatureGate(features.SecurityProfileOCI), "[LinuxOnly]", func() {
	f := framework.NewDefaultFramework("security-profile-oci")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should apply a seccomp profile pulled from an OCI registry", func(ctx context.Context) {
		pod := newSecurityProfileOCIPod(denyChmodProfile, "touch /tmp/file && if chmod 600 /tmp/file; then echo allowed; else echo denied; fi")
		e2eoutput.TestContainerOutput(ctx, f, "OCI seccomp profile", pod, 0, []string{"denied"})
	})

	ginkgo.It("should reuse a pulled profile for another pod", func(ctx context.Context) {
		first := e2epod.NewPodClient(f).Create(ctx, newSecurityProfileOCIPod(denyChmodProfile, "true"))
		framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, first.Name, f.Namespace.Name))

		cachedBefore := cachedSecurityProfilePulls(ctx)
		second := e2epod.NewPodClient(f).Create(ctx, newSecurityProfileOCIPod(denyChmodProfile, "true"))
		framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, second.Name, f.Namespace.Name))
		gomega.Expect(cachedSecurityProfilePulls(ctx)).To(gomega.BeNumerically(">", cachedBefore), "the second pod should use the cached profile")
		pulled, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{
			FieldSelector: securityProfileEventSelector(f, second.Name, kubeletevents.PulledSecurityProfile),
		})
		framework.ExpectNoError(err)
		gomega.Expect(pulled.Items).To(gomega.BeEmpty(), "the second pod should not pull the profile again")
	})

	ginkgo.It("should keep the runtime baseline for a permissive profile", func(ctx context.Context) {
		// The permissive profile allows everything, so a syscall that the
		// runtime baseline denies shows that the profiles were merged.
		pod := newSecurityProfileOCIPod(permissiveProfile, "grep "+SeccompProcStatusField+" "+ProcSelfStatusPath+
			"; if unshare -U true 2>/dev/null; then echo unshare allowed; else echo unshare denied; fi")
		e2eoutput.TestContainerOutput(ctx, f, "permissive OCI seccomp profile", pod, 0, []string{"2", "unshare denied"})
	})

	ginkgo.It("should report pull failures and keep the pod pending", func(ctx context.Context) {
		pod := e2epod.NewPodClient(f).Create(ctx, newSecurityProfileOCIPod(missingProfile, "true"))
		framework.ExpectNoError(e2eevents.WaitTimeoutForEvent(ctx, f.ClientSet, f.Namespace.Name,
			securityProfileEventSelector(f, pod.Name, kubeletevents.FailedToPullSecurityProfile), "", 2*time.Minute))
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodPending))
	})
})

var _ = SIGDescribe("SecurityProfileOCI", feature.SecurityProfileOCI, feature.CriProxy, framework.WithFeatureGate(features.SecurityProfileOCI), framework.WithSerial(), "[LinuxOnly]", func() {
	f := framework.NewDefaultFramework("security-profile-oci-cri-proxy")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func() {
		if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
			ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
		}
		ginkgo.DeferCleanup(func() error {
			return resetCRIProxyInjector(e2eCriProxy)
		})
	})

	ginkgo.It("should fail the pod when the runtime rejects the profile", func(ctx context.Context) {
		framework.ExpectNoError(addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
			if apiName == criproxy.PullSecurityProfile {
				return fmt.Errorf("%w: injected rejection", crierrors.ErrSecurityProfileInvalid)
			}
			return nil
		}))
		pod := e2epod.NewPodClient(f).Create(ctx, newSecurityProfileOCIPod(denyChmodProfile, "true"))
		framework.ExpectNoError(e2epod.WaitForPodFailedReason(ctx, f.ClientSet, pod, securityProfileRejectedReason, 2*time.Minute))
	})

	ginkgo.It("should retry transient pull failures", func(ctx context.Context) {
		framework.ExpectNoError(addCRIProxyInjector(e2eCriProxy, func(apiName string) error {
			if apiName == criproxy.PullSecurityProfile {
				return fmt.Errorf("%w: injected outage", crierrors.ErrRegistryUnavailable)
			}
			return nil
		}))
		pod := e2epod.NewPodClient(f).Create(ctx, newSecurityProfileOCIPod(denyChmodProfile, "true"))
		framework.ExpectNoError(e2eevents.WaitTimeoutForEvent(ctx, f.ClientSet, f.Namespace.Name,
			securityProfileEventSelector(f, pod.Name, kubeletevents.FailedToPullSecurityProfile), "", 2*time.Minute))

		framework.ExpectNoError(resetCRIProxyInjector(e2eCriProxy))
		framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))
	})
})

var _ = SIGDescribe("SecurityProfileOCI", feature.SecurityProfileOCI, framework.WithFeatureGate(features.SecurityProfileOCI), framework.WithSerial(), framework.WithSlow(), "[LinuxOnly]", func() {
	f := framework.NewDefaultFramework("security-profile-oci-gc")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// Unused profiles are removed after ImageMaximumGCAge, which also
	// removes unused images, so the images are pulled again afterwards.
	tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
		initialConfig.ImageMaximumGCAge = metav1.Duration{Duration: time.Minute}
		initialConfig.ImageMinimumGCAge = metav1.Duration{Duration: time.Second}
	})
	ginkgo.AfterEach(func(ctx context.Context) {
		framework.ExpectNoError(PrePullAllImages(ctx))
	})

	ginkgo.It("should remove a pulled profile that no pod uses", func(ctx context.Context) {
		_, digest, _ := strings.Cut(permissiveProfile, "@")
		_, imageService, err := getCRIClient(ctx)
		framework.ExpectNoError(err)
		listedDigests := func(ctx context.Context) ([]string, error) {
			profiles, err := imageService.ListSecurityProfiles(ctx)
			if err != nil {
				return nil, err
			}
			var digests []string
			for _, profile := range profiles {
				digests = append(digests, profile.GetDigest())
			}
			return digests, nil
		}

		podClient := e2epod.NewPodClient(f)
		pod := podClient.Create(ctx, newSecurityProfileOCIPod(permissiveProfile, "true"))
		framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))
		gomega.Expect(listedDigests(ctx)).To(gomega.ContainElement(digest))

		// The kubelet collects every 5 minutes.
		podClient.DeleteSync(ctx, pod.Name, metav1.DeleteOptions{}, e2epod.DefaultPodDeletionTimeout)
		gomega.Eventually(ctx, listedDigests).WithTimeout(15 * time.Minute).WithPolling(30 * time.Second).
			ShouldNot(gomega.ContainElement(digest))
	})
})

func newSecurityProfileOCIPod(ref, command string) *v1.Pod {
	name := "security-profile-oci-" + string(uuid.NewUUID())
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			SecurityContext: &v1.PodSecurityContext{
				SeccompProfile: &v1.SeccompProfile{
					Type: v1.SeccompProfileTypeOCI,
					OCI:  &v1.SecurityProfileOCI{Ref: ref},
				},
			},
			Containers: []v1.Container{{
				Name:    name,
				Image:   busyboxImage,
				Command: []string{"sh", "-c", command},
			}},
		},
	}
}

// cachedSecurityProfilePulls returns the number of profile pulls the runtime
// served from its storage, as counted by the kubelet.
func cachedSecurityProfilePulls(ctx context.Context) float64 {
	ms, err := e2emetrics.GrabKubeletMetricsWithoutProxy(ctx, nodeNameOrIP()+":10255", "/metrics")
	framework.ExpectNoError(err)
	var total float64
	for _, sample := range ms["kubelet_security_profile_pull_duration_seconds_count"] {
		if sample.Metric["cached"] == "true" {
			total += float64(sample.Value)
		}
	}
	return total
}

func securityProfileEventSelector(f *framework.Framework, podName, reason string) string {
	return fields.Set{
		"involvedObject.kind":      "Pod",
		"involvedObject.name":      podName,
		"involvedObject.namespace": f.Namespace.Name,
		"reason":                   reason,
	}.AsSelector().String()
}
