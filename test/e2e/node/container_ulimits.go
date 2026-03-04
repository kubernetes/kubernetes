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

package node

import (
	"context"
	"fmt"
	"slices"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures/features/containerulimits"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

var _ = SIGDescribe("ContainerUlimits", framework.WithFeatureGate(features.ContainerUlimits), func() {
	f := framework.NewDefaultFramework("container-ulimits")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should apply independent limits to init and regular containers [LinuxOnly]", func(ctx context.Context) {
		nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		supported := false
		for _, node := range nodes.Items {
			if node.Labels[v1.LabelOSStable] == "linux" && slices.Contains(node.Status.DeclaredFeatures, containerulimits.ContainerUlimits) {
				supported = true
				break
			}
		}
		if !supported {
			e2eskipper.Skipf("no ready schedulable Linux node supports ContainerUlimits")
		}

		container := func(name string, soft, hard int64) v1.Container {
			return v1.Container{
				Name:  name,
				Image: imageutils.GetE2EImage(imageutils.BusyBox),
				Command: []string{"sh", "-c", fmt.Sprintf(`set -eu
cat /proc/self/limits
test "$(ulimit -Sn)" = "%d"
test "$(ulimit -Hn)" = "%d"
test "$(ulimit -Sc)" = "0"
test "$(ulimit -Hc)" = "0"`, soft, hard)},
				SecurityContext: &v1.SecurityContext{Ulimits: &v1.Ulimits{
					Nofile: &v1.Ulimit{Soft: ptr.To(soft), Hard: ptr.To(hard)},
					Core:   &v1.Ulimit{Soft: ptr.To[int64](0), Hard: ptr.To[int64](0)},
				}},
			}
		}
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "container-ulimits"},
			Spec: v1.PodSpec{
				OS:             &v1.PodOS{Name: v1.Linux},
				NodeSelector:   map[string]string{v1.LabelOSStable: "linux"},
				RestartPolicy:  v1.RestartPolicyNever,
				InitContainers: []v1.Container{container("init", 1024, 2048)},
				Containers:     []v1.Container{container("first", 2048, 4096), container("second", 4096, 8192)},
			},
		}
		e2epod.NewPodClient(f).Create(ctx, pod)
		framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))
	})
})
