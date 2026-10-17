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
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// Node e2e runs without a scheduler or a kube-controller-manager: the pod
// client binds pods to the single node directly, and nothing can evict a
// running pod for a NoExecute taint. These tests therefore exercise only the
// kubelet's own admission check for NoExecute taints.
var _ = SIGDescribe("NoExecute taint admission", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("taint-admission-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	taint := v1.Taint{
		Key:    "e2e-node.k8s.io/taint-admission",
		Value:  "true",
		Effect: v1.TaintEffectNoExecute,
	}

	ginkgo.It("should reject a pod that does not tolerate a NoExecute taint on the node", func(ctx context.Context) {
		node := getNodeName(ctx, f)

		ginkgo.By("adding a NoExecute taint to the node")
		// Register the cleanup first: if the add fails after the API applied it,
		// the only node must not stay tainted for the rest of the serial suite.
		// RemoveTaintOffNode is a no-op when the taint is absent.
		ginkgo.DeferCleanup(e2enode.RemoveTaintOffNode, f.ClientSet, node, taint)
		e2enode.AddOrUpdateTaintOnNode(ctx, f.ClientSet, node, taint)
		gomega.Eventually(ctx, func(ctx context.Context) (bool, error) {
			return e2enode.NodeHasTaint(ctx, f.ClientSet, node, &taint)
		}).WithTimeout(30 * time.Second).WithPolling(time.Second).Should(gomega.BeTrueBecause("node %q should carry taint %s", node, taint.ToString()))

		// The node and pod watches are independent, so a pod can reach the
		// kubelet before its informer has the taint. Admission then passes on
		// the clean cached node without a refetch, because the refetch only
		// runs on failure. Retry with fresh pods until one is admitted against
		// the tainted view.
		ginkgo.By("creating pods with no tolerations until the kubelet rejects one")
		podClient := e2epod.NewPodClient(f)
		attempt := 0
		gomega.Eventually(ctx, func(ctx context.Context) error {
			attempt++
			pod := podClient.Create(ctx, taintAdmissionPausePod(fmt.Sprintf("taint-admission-rejected-%d", attempt)))
			var observed *v1.Pod
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "running or failed", time.Minute, func(p *v1.Pod) (bool, error) {
				observed = p
				return p.Status.Phase == v1.PodRunning || p.Status.Phase == v1.PodFailed, nil
			})
			if err != nil {
				return gomega.StopTrying(fmt.Sprintf("pod %q reached neither Running nor Failed", pod.Name)).Wrap(err)
			}
			if observed.Status.Phase == v1.PodFailed {
				if observed.Status.Reason != tainttoleration.Name {
					return gomega.StopTrying(fmt.Sprintf("pod %q failed with reason %q, want %q", pod.Name, observed.Status.Reason, tainttoleration.Name))
				}
				return nil
			}
			podClient.DeleteSync(ctx, pod.Name, metav1.DeleteOptions{}, e2epod.DefaultPodDeletionTimeout)
			return fmt.Errorf("pod %q was admitted before the kubelet observed the taint", pod.Name)
		}).WithTimeout(3 * time.Minute).WithPolling(time.Second).Should(gomega.Succeed())
	})

	ginkgo.It("should admit a pod after a NoExecute taint is removed from the node", func(ctx context.Context) {
		node := getNodeName(ctx, f)

		ginkgo.By("adding a NoExecute taint to the node")
		// Register the cleanup first: if the add fails after the API applied it,
		// the only node must not stay tainted for the rest of the serial suite.
		// RemoveTaintOffNode is a no-op when the taint is absent.
		ginkgo.DeferCleanup(e2enode.RemoveTaintOffNode, f.ClientSet, node, taint)
		e2enode.AddOrUpdateTaintOnNode(ctx, f.ClientSet, node, taint)
		gomega.Eventually(ctx, func(ctx context.Context) (bool, error) {
			return e2enode.NodeHasTaint(ctx, f.ClientSet, node, &taint)
		}).WithTimeout(30 * time.Second).WithPolling(time.Second).Should(gomega.BeTrueBecause("node %q should carry taint %s", node, taint.ToString()))

		ginkgo.By("removing the NoExecute taint from the node")
		// RemoveTaintOffNode verifies the taint is gone from the API before returning.
		e2enode.RemoveTaintOffNode(ctx, f.ClientSet, node, taint)

		// Non-regression only: whether the kubelet's informer still holds the
		// tainted node when this pod arrives cannot be controlled here, so this
		// does not reliably reproduce the stale-cache race. It asserts the pod is
		// admitted either way.
		ginkgo.By("creating a pod with no tolerations after the taint removal")
		pod := e2epod.NewPodClient(f).Create(ctx, taintAdmissionPausePod("taint-admission-admitted"))
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))
	})
})

// taintAdmissionPausePod returns a pause pod with no tolerations. The pod
// client sets NodeName for node e2e, so it is left empty here.
func taintAdmissionPausePod(name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "pause",
					Image: imageutils.GetPauseImageName(),
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}
