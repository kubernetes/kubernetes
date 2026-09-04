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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("CgroupOptions", framework.WithFeatureGate(features.CgroupOptions), func() {
	f := framework.NewDefaultFramework("cgroup-options-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var podClient *e2epod.PodClient

	ginkgo.BeforeEach(func() {
		if !IsCgroup2UnifiedMode() {
			ginkgo.Skip("This test requires cgroups v2")
		}
		podClient = e2epod.NewPodClient(f)
	})

	makePod := func(name string, mountMode *v1.CgroupMountMode) *v1.Pod {
		sc := &v1.SecurityContext{}
		if mountMode != nil {
			sc.CgroupOptions = &v1.CgroupOptions{MountMode: mountMode}
		}
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				Containers: []v1.Container{
					{
						Name:            "test",
						Image:           imageutils.GetE2EImage(imageutils.BusyBox),
						Command:         []string{"/bin/sleep", "10000"},
						SecurityContext: sc,
					},
				},
			},
		}
	}

	// mkdirCmd attempts to create and remove a descendant cgroup, which only
	// succeeds when /sys/fs/cgroup is mounted read-write for the container.
	mkdirCmd := []string{"sh", "-c", "mkdir /sys/fs/cgroup/e2e-test && rmdir /sys/fs/cgroup/e2e-test"}

	ginkgo.It("should mount /sys/fs/cgroup writable when mountMode is Writable", func(ctx context.Context) {
		pod := makePod("cgroup-writable", ptr.To(v1.CgroupMountModeWritable))
		podClient.CreateSync(ctx, pod)

		ginkgo.By("verifying the container can create a descendant cgroup")
		stdout, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, "test", mkdirCmd...)
		framework.ExpectNoError(err, "expected mkdir in /sys/fs/cgroup to succeed; stdout=%q stderr=%q", stdout, stderr)
	})

	ginkgo.It("should mount /sys/fs/cgroup read-only by default", func(ctx context.Context) {
		pod := makePod("cgroup-readonly", nil)
		podClient.CreateSync(ctx, pod)

		ginkgo.By("verifying the container cannot create a descendant cgroup")
		_, _, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, "test", mkdirCmd...)
		gomega.Expect(err).To(gomega.HaveOccurred(), "expected mkdir in read-only /sys/fs/cgroup to fail")
	})

	ginkgo.It("should mount /sys/fs/cgroup read-only when mountMode is ReadOnly", func(ctx context.Context) {
		pod := makePod("cgroup-explicit-readonly", ptr.To(v1.CgroupMountModeReadOnly))
		podClient.CreateSync(ctx, pod)

		ginkgo.By("verifying the container cannot create a descendant cgroup")
		_, _, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, "test", mkdirCmd...)
		gomega.Expect(err).To(gomega.HaveOccurred(), "expected mkdir in read-only /sys/fs/cgroup to fail")
	})
})
