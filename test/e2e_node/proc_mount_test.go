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

package e2enode

import (
	"context"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	gomegatypes "github.com/onsi/gomega/types"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var falseVar = false

var _ = SIGDescribe("DefaultProcMount [LinuxOnly]", framework.WithNodeConformance(), func() {
	f := framework.NewDefaultFramework("proc-mount-default-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.It("will mask proc mounts by default", func(ctx context.Context) {
		testProcMount(ctx, f, v1.DefaultProcMount, gomega.BeNumerically(">=", 10), gomega.BeNumerically(">=", 7))
	})
})

var _ = SIGDescribe("ProcMount [LinuxOnly]", nodefeature.ProcMountType, nodefeature.UserNamespacesSupport, feature.UserNamespacesSupport, func() {
	f := framework.NewDefaultFramework("proc-mount-baseline-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	f.It("will fail to unmask proc mounts if not privileged", func(ctx context.Context) {
		if !supportsUserNS(ctx, f) {
			e2eskipper.Skipf("runtime does not support user namespaces")
		}
		pmt := v1.UnmaskedProcMount
		podClient := e2epod.NewPodClient(f)
		_, err := podClient.PodInterface.Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "proc-mount-pod"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "test-container-1",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sleep"},
						Args:    []string{"10000"},
						SecurityContext: &v1.SecurityContext{
							ProcMount: &pmt,
						},
					},
				},
				HostUsers: &falseVar,
			},
		}, metav1.CreateOptions{})
		gomega.Expect(err).To(gomega.HaveOccurred())
	})
})

var _ = SIGDescribe("ProcMount [LinuxOnly]", nodefeature.ProcMountType, nodefeature.UserNamespacesSupport, feature.UserNamespacesSupport, func() {
	f := framework.NewDefaultFramework("proc-mount-privileged-test")

	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	f.It("will unmask proc mounts if requested", func(ctx context.Context) {
		if !supportsUserNS(ctx, f) {
			e2eskipper.Skipf("runtime does not support user namespaces")
		}
		testProcMount(ctx, f, v1.UnmaskedProcMount, gomega.Equal(1), gomega.BeZero())
	})
})

func testProcMount(ctx context.Context, f *framework.Framework, pmt v1.ProcMountType, expectedLines gomegatypes.GomegaMatcher, expectedReadOnly gomegatypes.GomegaMatcher) {
	ginkgo.By("creating a target pod")
	podClient := e2epod.NewPodClient(f)
	pod := podClient.CreateSync(ctx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "proc-mount-pod"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "test-container-1",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"/bin/sleep"},
					Args:    []string{"10000"},
					SecurityContext: &v1.SecurityContext{
						ProcMount: &pmt,
					},
				},
			},
			HostUsers: &falseVar,
		},
	})

	_, err := testutils.PodRunningReady(pod)
	framework.ExpectNoError(err)

	output := e2epod.ExecCommandInContainer(f, pod.Name, pod.Spec.Containers[0].Name, "/bin/sh", "-ec", "mount | grep /proc")
	ginkgo.By(output)
	lines := strings.Split(output, "\n")
	gomega.Expect(len(lines)).To(expectedLines)
	gomega.Expect(strings.Count(output, "(ro")).To(expectedReadOnly)
}

func supportsUserNS(ctx context.Context, f *framework.Framework) bool {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)
	// Assuming that there is only one node, because this is a node e2e test.
	gomega.Expect(nodeList.Items).To(gomega.HaveLen(1))
	node := nodeList.Items[0]
	for _, rc := range node.Status.RuntimeHandlers {
		if rc.Name == "" && rc.Features != nil && *rc.Features.UserNamespaces {
			return true
		}
	}
	return false
}
