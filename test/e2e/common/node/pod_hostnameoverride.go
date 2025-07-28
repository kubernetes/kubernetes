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

package node

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

func newTestPod(namespace string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "hostfqdn-",
			Namespace:    namespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "test-pod-hostname-override",
					Image:   imageutils.GetE2EImage(imageutils.Agnhost),
					Command: []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)';'"},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}

var _ = SIGDescribe("Override hostname of Pod", framework.WithFeatureGate(features.HostnameOverride), func() {
	f := framework.NewDefaultFramework("hostfqdn")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.It("a pod has hostnameOverride field with value that is a valid DNS subdomain.", func(ctx context.Context) {
		pod := newTestPod(f.Namespace.Name)
		hostnameOverride := "override.example.host"
		pod.Spec.HostnameOverride = &hostnameOverride
		output := []string{fmt.Sprintf("%s;%s;", hostnameOverride, hostnameOverride)}
		e2eoutput.TestContainerOutput(ctx, f, "hostnameOverride overrides hostname", pod, 0, output)
	})

	ginkgo.It("a pod with hostname and hostnameOverride fields will have hostnameOverride as hostname", func(ctx context.Context) {
		pod := newTestPod(f.Namespace.Name)
		hostname := "custom-host"
		hostnameOverride := "override-host"
		pod.Spec.Hostname = hostname
		pod.Spec.HostnameOverride = &hostnameOverride
		output := []string{fmt.Sprintf("%s;%s;", hostnameOverride, hostnameOverride)}
		e2eoutput.TestContainerOutput(ctx, f, "hostnameOverride overrides hostname", pod, 0, output)
	})

	ginkgo.It("a pod with only hostnameOverride field will have hostnameOverride as hostname", func(ctx context.Context) {
		pod := newTestPod(f.Namespace.Name)
		hostnameOverride := "override-host"
		pod.Spec.HostnameOverride = &hostnameOverride
		output := []string{fmt.Sprintf("%s;%s;", hostnameOverride, hostnameOverride)}
		e2eoutput.TestContainerOutput(ctx, f, "hostnameOverride only", pod, 0, output)
	})

	ginkgo.It("a pod with subdomain and hostnameOverride fields will have hostnameOverride as hostname", func(ctx context.Context) {
		pod := newTestPod(f.Namespace.Name)
		subdomain := "t"
		hostnameOverride := "override-host"
		pod.Spec.Subdomain = subdomain
		pod.Spec.HostnameOverride = &hostnameOverride
		output := []string{fmt.Sprintf("%s;%s;", hostnameOverride, hostnameOverride)}
		e2eoutput.TestContainerOutput(ctx, f, "subdomain and hostnameOverride", pod, 0, output)
	})

	ginkgo.It("a pod with setHostnameAsFQDN and hostnameOverride fields will fail to be created", func(ctx context.Context) {
		pod := newTestPod(f.Namespace.Name)
		setHostnameAsFQDN := true
		hostnameOverride := "override-host"
		pod.Spec.SetHostnameAsFQDN = &setHostnameAsFQDN
		pod.Spec.HostnameOverride = &hostnameOverride
		_, err := e2epod.NewPodClient(f).TryCreate(ctx, pod)
		gomega.Expect(err).To(gomega.HaveOccurred(), "Pod creation should fail when both setHostnameAsFQDN and hostnameOverride are set")
	})

	ginkgo.It("a pod with hostNetwork and hostnameOverride fields will fail to be created", func(ctx context.Context) {
		pod := newTestPod(f.Namespace.Name)
		hostnameOverride := "override-host"
		pod.Spec.HostNetwork = true
		pod.Spec.HostnameOverride = &hostnameOverride
		_, err := e2epod.NewPodClient(f).TryCreate(ctx, pod)
		gomega.Expect(err).To(gomega.HaveOccurred(), "Pod creation should fail when both hostNetwork and hostnameOverride are set")
	})

	ginkgo.It("a pod with non-RFC1123 subdomain string for hostnameOverride field will fail to be created", func(ctx context.Context) {
		pod := newTestPod(f.Namespace.Name)
		hostnameOverride := "Not-RFC1123"
		pod.Spec.HostNetwork = false
		pod.Spec.HostnameOverride = &hostnameOverride
		_, err := e2epod.NewPodClient(f).TryCreate(ctx, pod)
		gomega.Expect(err).To(gomega.HaveOccurred(), "Pod creation should fail when non-RFC1123 subdomain string for hostnameOverride field are set")
	})
})
