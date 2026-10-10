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
	"math/bits"
	"strconv"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/common/node/framework/cgroups"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("OOMScoreAdjTieBreak [LinuxOnly]", framework.WithSerial(), framework.WithDisruptive(), framework.WithFeatureGate(features.KubeletOOMScoreAdjTieBreak), func() {
	f := framework.NewDefaultFramework("oom-score-adj-tie-break")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
		if initialConfig.FeatureGates == nil {
			initialConfig.FeatureGates = make(map[string]bool)
		}
		initialConfig.FeatureGates[string(features.KubeletOOMScoreAdjTieBreak)] = true
	})

	ginkgo.It("should rank containers that cannot exceed their memory request below others", func(ctx context.Context) {
		const request = 128 * 1024 * 1024
		mem := func(q int64) v1.ResourceList {
			return v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(q, resource.BinarySI)}
		}
		container := func(name string, res v1.ResourceRequirements) v1.Container {
			return v1.Container{Name: name, Image: busyboxImage, Command: []string{"sleep", "3600"}, Resources: res}
		}
		sidecar := container("sidecar", v1.ResourceRequirements{Requests: mem(request / 2)})
		sidecar.RestartPolicy = new(v1.ContainerRestartPolicyAlways)
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "oom-score-adj-tie-break"},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{sidecar},
				Containers: []v1.Container{
					container("bounded", v1.ResourceRequirements{Requests: mem(request), Limits: mem(request)}),
					container("unbounded", v1.ResourceRequirements{Requests: mem(request)}),
				},
			},
		}
		pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
		ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, pod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

		node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, pod.Spec.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		capacity := node.Status.Capacity.Memory().Value()

		// Independent of pkg/kubelet/qos: the linear score, minus one point per
		// doubling of request above 64Mi capped at log2(capacity/1000/64Mi),
		// minus that cap + 1 for containers that cannot exceed their request.
		const floor = 64 * 1024 * 1024
		log2Floor := func(x int64) int64 {
			if x < 1 {
				return 0
			}
			return int64(bits.Len64(uint64(x))) - 1
		}
		maxNudge := log2Floor(capacity / 1000 / floor)
		unbounded := 1000 - (1000*request)/capacity - min(log2Floor(request/floor), maxNudge)
		bounded := unbounded - (maxNudge + 1)

		gomega.Expect(bounded).To(gomega.BeNumerically("<", unbounded))
		// The sidecar's smaller request scores higher on its own, so it is
		// capped at the highest regular container score.
		for name, want := range map[string]int64{"bounded": bounded, "unbounded": unbounded, "sidecar": unbounded} {
			ginkgo.By("verifying oom_score_adj of container " + name)
			framework.ExpectNoError(cgroups.VerifyOomScoreAdjValue(f, pod, name, strconv.FormatInt(want, 10)))
		}
	})
})
