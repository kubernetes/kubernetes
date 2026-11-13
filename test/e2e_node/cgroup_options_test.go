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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/component-helpers/nodedeclaredfeatures/features/cgroupoptions"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/mount-utils"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

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
	gomega.Expect(nodeList.Items).To(gomega.HaveLen(1))
	return slices.Contains(nodeList.Items[0].Status.DeclaredFeatures, cgroupoptions.CgroupOptionsFeatureName)
}

func hostSupportsWritableCgroups() bool {
	if !kubeletCfg.CgroupsPerQOS {
		return false
	}
	mounts, err := mount.ParseMountInfo("/proc/self/mountinfo")
	framework.ExpectNoError(err)
	// Walk backwards since a later mount takes precedence at the same path.
	for _, m := range slices.Backward(mounts) {
		if m.MountPoint == "/sys/fs/cgroup" {
			return m.FsType == "cgroup2" && slices.Contains(m.SuperOptions, "nsdelegate")
		}
	}
	return false
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
				e2eskipper.Skipf("node does not declare the %s feature", cgroupoptions.CgroupOptionsFeatureName)
			}
		})

		ginkgo.It("should mount /sys/fs/cgroup read-only when mountMode is ReadOnly", func(ctx context.Context) {
			pod := cgroupOptionsPod("cgroup-explicit-readonly",
				cgroupOptionsContainer("test", ptr.To(v1.CgroupMountModeReadOnly)))
			podClient.CreateSync(ctx, pod)

			ginkgo.By("verifying the container cannot create a descendant cgroup")
			expectCgroupReadOnly(f, pod.Name, "test")
		})

		ginkgo.Context("and the host supports writable cgroups", func() {
			ginkgo.BeforeEach(func() {
				if !hostSupportsWritableCgroups() {
					e2eskipper.Skipf("host does not support writable cgroups: it needs cgroup v2 mounted with nsdelegate and a cgroup per pod")
				}
			})

			ginkgo.It("should mount /sys/fs/cgroup read-write when mountMode is ReadWrite", func(ctx context.Context) {
				pod := cgroupOptionsPod("cgroup-readwrite",
					cgroupOptionsContainer("test", ptr.To(v1.CgroupMountModeReadWrite)))
				podClient.CreateSync(ctx, pod)

				ginkgo.By("verifying the container can create a descendant cgroup")
				stdout, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, "test", mkdirCmd...)
				framework.ExpectNoError(err, "expected mkdir in /sys/fs/cgroup to succeed; stdout=%q stderr=%q", stdout, stderr)
			})

			ginkgo.It("should apply the mount mode per container", func(ctx context.Context) {
				pod := cgroupOptionsPod("cgroup-mixed",
					cgroupOptionsContainer("readwrite", ptr.To(v1.CgroupMountModeReadWrite)),
					cgroupOptionsContainer("readonly", ptr.To(v1.CgroupMountModeReadOnly)))
				podClient.CreateSync(ctx, pod)

				ginkgo.By("verifying only the opted-in container can create a descendant cgroup")
				stdout, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, "readwrite", mkdirCmd...)
				framework.ExpectNoError(err, "expected mkdir to succeed in the read-write container; stdout=%q stderr=%q", stdout, stderr)

				expectCgroupReadOnly(f, pod.Name, "readonly")
			})

			ginkgo.DescribeTable("should enforce the pod cgroup hierarchy limit",
				func(ctx context.Context, limitFile string, expectedLimit int, createCgroup string) {
					pod := cgroupOptionsPod("cgroup-limit",
						cgroupOptionsContainer("test", ptr.To(v1.CgroupMountModeReadWrite)))
					pod = podClient.CreateSync(ctx, pod)

					ginkgo.By("verifying " + limitFile + " on the pod cgroup")
					cgroupPath := makeCgroupPathForPod(pod, kubeletCfg.CgroupRoot, kubeletCfg.CgroupDriver, true)
					limit, err := os.ReadFile(filepath.Join(cgroupPath, limitFile))
					framework.ExpectNoError(err)
					gomega.Expect(strings.TrimSpace(string(limit))).To(gomega.Equal(strconv.Itoa(expectedLimit)))

					ginkgo.By("verifying cgroup creation fails at the limit")
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
					gomega.Expect(count).To(gomega.BeNumerically("<", expectedLimit), "runtime-created cgroups count toward the pod's limit, so the container should hit it before creating %d of its own", expectedLimit)
				},
				ginkgo.Entry("descendants", "cgroup.max.descendants", 250, `mkdir "$dir/d$i"`),
				ginkgo.Entry("depth", "cgroup.max.depth", 50, `dir="$dir/d"; mkdir "$dir"`),
			)

			ginkgo.It("should keep the pod cgroup hierarchy limits when the pod is resized in place", func(ctx context.Context) {
				container := cgroupOptionsContainer("test", ptr.To(v1.CgroupMountModeReadWrite))
				container.Resources = v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("100Mi"),
					},
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("200m"),
						v1.ResourceMemory: resource.MustParse("200Mi"),
					},
				}
				pod := podClient.CreateSync(ctx, cgroupOptionsPod("cgroup-resize", container))

				cgroupPath := makeCgroupPathForPod(pod, kubeletCfg.CgroupRoot, kubeletCfg.CgroupDriver, true)
				readPodCgroupFile := func(name string) (string, error) {
					value, err := os.ReadFile(filepath.Join(cgroupPath, name))
					return strings.TrimSpace(string(value)), err
				}
				expectHierarchyLimits := func() {
					ginkgo.GinkgoHelper()
					gomega.Expect(readPodCgroupFile("cgroup.max.descendants")).To(gomega.Equal("250"))
					gomega.Expect(readPodCgroupFile("cgroup.max.depth")).To(gomega.Equal("50"))
				}

				ginkgo.By("verifying the hierarchy limits before the resize")
				expectHierarchyLimits()

				ginkgo.By("resizing the pod's CPU and memory limits")
				patch := `{"spec":{"containers":[{"name":"test","resources":{"limits":{"cpu":"400m","memory":"400Mi"}}}]}}`
				_, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Patch(ctx, pod.Name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{}, "resize")
				framework.ExpectNoError(err)

				ginkgo.By("waiting for the kubelet to write the new limits to the pod cgroup")
				gomega.Eventually(ctx, readPodCgroupFile).WithArguments("memory.max").
					WithTimeout(framework.PodStartShortTimeout).Should(gomega.Equal(strconv.Itoa(400 << 20)))
				gomega.Eventually(ctx, readPodCgroupFile).WithArguments("cpu.max").
					WithTimeout(framework.PodStartShortTimeout).Should(gomega.Equal("40000 100000"))

				ginkgo.By("verifying the hierarchy limits after the resize")
				expectHierarchyLimits()
			})
		})
	})

	ginkgo.Context("when the node does not declare CgroupOptions", func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			if nodeDeclaresCgroupOptions(ctx, f) {
				e2eskipper.Skipf("node declares the %s feature", cgroupoptions.CgroupOptionsFeatureName)
			}
		})

		ginkgo.DescribeTable("should reject a pod that requests a cgroup mount mode", func(ctx context.Context, mountMode v1.CgroupMountMode) {
			pod := cgroupOptionsPod("cgroup-unsupported",
				cgroupOptionsContainer("test", new(mountMode)))
			pod = podClient.Create(ctx, pod)

			framework.ExpectNoError(e2epod.WaitForPodFailedReason(ctx, f.ClientSet, pod, lifecycle.PodFeatureUnsupported, framework.PodStartShortTimeout))
		},
			ginkgo.Entry("ReadOnly", v1.CgroupMountModeReadOnly),
			ginkgo.Entry("ReadWrite", v1.CgroupMountModeReadWrite),
		)
	})
})
