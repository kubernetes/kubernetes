//go:build linux

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
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// cgroupOptionsFeatureName is the name the kubelet declares in
// node.status.declaredFeatures once cgroup mount modes are supported on the node.
const cgroupOptionsFeatureName = "CgroupOptions"

// mkdirCmd creates and removes a descendant cgroup, which only succeeds when
// /sys/fs/cgroup is mounted read-write for the container.
var mkdirCmd = []string{"sh", "-c", "mkdir /sys/fs/cgroup/e2e-test && rmdir /sys/fs/cgroup/e2e-test"}

func expectCgroupReadOnly(f *framework.Framework, podName, containerName string) {
	ginkgo.GinkgoHelper()
	stdout, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, podName, containerName, mkdirCmd...)
	gomega.Expect(err).To(gomega.HaveOccurred(), "expected mkdir in read-only /sys/fs/cgroup to fail; stdout=%q stderr=%q", stdout, stderr)
	gomega.Expect(stderr).To(gomega.ContainSubstring("Read-only file system"), "expected mkdir to fail because /sys/fs/cgroup is read-only")
}

func nodeDeclaresCgroupOptions(ctx context.Context, f *framework.Framework) bool {
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)
	// Node e2e tests run against a single node.
	gomega.Expect(nodeList.Items).To(gomega.HaveLen(1))
	return slices.Contains(nodeList.Items[0].Status.DeclaredFeatures, cgroupOptionsFeatureName)
}

func cgroupOptionsContainer(name string, mountMode *v1.CgroupMountMode) v1.Container {
	sc := &v1.SecurityContext{}
	if mountMode != nil {
		sc.CgroupOptions = &v1.CgroupOptions{MountMode: mountMode}
	}
	return v1.Container{
		Name:            name,
		Image:           imageutils.GetE2EImage(imageutils.BusyBox),
		Command:         []string{"/bin/sleep", "10000"},
		SecurityContext: sc,
	}
}

func cgroupOptionsPod(name string, containers ...v1.Container) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers:    containers,
		},
	}
}

var _ = SIGDescribe("CgroupOptions", framework.WithFeatureGate(features.CgroupOptions), func() {
	f := framework.NewDefaultFramework("cgroup-options-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var podClient *e2epod.PodClient

	ginkgo.BeforeEach(func(ctx context.Context) {
		podClient = e2epod.NewPodClient(f)
		waitForNodeReady(ctx)
	})

	ginkgo.It("should start a container with the runtime's default cgroup mount mode", func(ctx context.Context) {
		pod := cgroupOptionsPod("cgroup-default", cgroupOptionsContainer("test", nil))
		podClient.CreateSync(ctx, pod)
	})

	ginkgo.Context("when the node declares CgroupOptions", func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			if !IsCgroup2UnifiedMode() {
				ginkgo.Skip("This test requires cgroups v2")
			}
			if !nodeDeclaresCgroupOptions(ctx, f) {
				e2eskipper.Skipf("node does not declare the %s feature", cgroupOptionsFeatureName)
			}
		})

		ginkgo.It("should mount /sys/fs/cgroup read-only when mountMode is ReadOnly", func(ctx context.Context) {
			pod := cgroupOptionsPod("cgroup-explicit-readonly",
				cgroupOptionsContainer("test", ptr.To(v1.CgroupMountModeReadOnly)))
			podClient.CreateSync(ctx, pod)

			ginkgo.By("verifying the container cannot create a descendant cgroup")
			expectCgroupReadOnly(f, pod.Name, "test")
		})

		ginkgo.It("should mount /sys/fs/cgroup writable when mountMode is Writable", func(ctx context.Context) {
			pod := cgroupOptionsPod("cgroup-writable",
				cgroupOptionsContainer("test", ptr.To(v1.CgroupMountModeWritable)))
			podClient.CreateSync(ctx, pod)

			ginkgo.By("verifying the container can create a descendant cgroup")
			stdout, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, "test", mkdirCmd...)
			framework.ExpectNoError(err, "expected mkdir in /sys/fs/cgroup to succeed; stdout=%q stderr=%q", stdout, stderr)
		})

		ginkgo.It("should apply the mount mode per container", func(ctx context.Context) {
			pod := cgroupOptionsPod("cgroup-mixed",
				cgroupOptionsContainer("writable", ptr.To(v1.CgroupMountModeWritable)),
				cgroupOptionsContainer("readonly", ptr.To(v1.CgroupMountModeReadOnly)))
			podClient.CreateSync(ctx, pod)

			ginkgo.By("verifying only the opted-in container can create a descendant cgroup")
			stdout, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, "writable", mkdirCmd...)
			framework.ExpectNoError(err, "expected mkdir to succeed in the writable container; stdout=%q stderr=%q", stdout, stderr)

			expectCgroupReadOnly(f, pod.Name, "readonly")
		})

		ginkgo.DescribeTable("should enforce the pod cgroup hierarchy limit",
			func(ctx context.Context, limitFile string, expectedLimit int, createCgroup string) {
				pod := cgroupOptionsPod("cgroup-limit",
					cgroupOptionsContainer("test", ptr.To(v1.CgroupMountModeWritable)))
				pod = podClient.CreateSync(ctx, pod)

				ginkgo.By("verifying " + limitFile + " on the pod cgroup")
				cgroupPath := makeCgroupPathForPod(pod, kubeletCfg.CgroupRoot, kubeletCfg.CgroupDriver, true)
				limit, err := os.ReadFile(filepath.Join(cgroupPath, limitFile))
				framework.ExpectNoError(err)
				gomega.Expect(strings.TrimSpace(string(limit))).To(gomega.Equal(strconv.Itoa(expectedLimit)))

				ginkgo.By("verifying cgroup creation fails at the limit")
				// Attempt one more than the Pod limit so this test finishes even if the
				// limit is not enforced.
				command := fmt.Sprintf(`export LC_ALL=C
dir=/sys/fs/cgroup
i=0
while [ "$i" -le %d ]; do
	%s || { echo "$i"; exit 1; }
	i=$((i+1))
done
echo "$i"`, expectedLimit, createCgroup)
				created, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, "test", "sh", "-c", command)
				gomega.Expect(err).To(gomega.HaveOccurred(), "expected cgroup creation to fail; stdout=%q stderr=%q", created, stderr)
				gomega.Expect(stderr).To(gomega.ContainSubstring("Resource temporarily unavailable"), "expected cgroup creation to fail with EAGAIN at the limit")
				count, err := strconv.Atoi(strings.TrimSpace(created))
				framework.ExpectNoError(err, "expected a count of created cgroups, got %q", created)
				gomega.Expect(count).To(gomega.BeNumerically(">", 0), "the container should be able to create at least one cgroup")
				// Runtime-created cgroups also count toward the Pod's limit.
				gomega.Expect(count).To(gomega.BeNumerically("<", expectedLimit))
			},
			ginkgo.Entry("descendants", "cgroup.max.descendants", 250, `mkdir "$dir/d$i"`),
			ginkgo.Entry("depth", "cgroup.max.depth", 50, `dir="$dir/d"; mkdir "$dir"`),
		)
	})

	ginkgo.Context("when the node does not declare CgroupOptions", func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			if nodeDeclaresCgroupOptions(ctx, f) {
				e2eskipper.Skipf("node declares the %s feature", cgroupOptionsFeatureName)
			}
		})

		ginkgo.DescribeTable("should reject a pod that requests a cgroup mount mode", func(ctx context.Context, mountMode v1.CgroupMountMode) {
			pod := cgroupOptionsPod("cgroup-unsupported",
				cgroupOptionsContainer("test", new(mountMode)))
			pod = podClient.Create(ctx, pod)

			framework.ExpectNoError(e2epod.WaitForPodFailedReason(ctx, f.ClientSet, pod, lifecycle.PodFeatureUnsupported, framework.PodStartShortTimeout))
		},
			ginkgo.Entry("ReadOnly", v1.CgroupMountModeReadOnly),
			ginkgo.Entry("Writable", v1.CgroupMountModeWritable),
		)
	})
})
