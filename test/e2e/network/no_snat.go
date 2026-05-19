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

package network

import (
	"context"
	"fmt"
	"net"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	testPodPort    = "8080"
	noSNATTestName = "no-snat-test"
)

var (
	testPod = v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: noSNATTestName,
			Labels: map[string]string{
				noSNATTestName: "",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  noSNATTestName,
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"netexec", "--http-port", testPodPort},
				},
			},
		},
	}
)

// This test verifies that a Pod on each node in a cluster can talk to Pods on every other node without SNAT.
// We use the [Feature:NoSNAT] tag so that most jobs will skip this test by default.
var _ = common.SIGDescribe("NoSNAT", func() {
	f := framework.NewDefaultFramework("no-snat-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.It("Should be able to send traffic between Pods without SNAT", func(ctx context.Context) {
		cs := f.ClientSet
		pc := cs.CoreV1().Pods(f.Namespace.Name)

		ginkgo.By("creating a test pod on each Node")
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 3)
		framework.ExpectNoError(err)
		gomega.Expect(nodes.Items).ToNot(gomega.BeEmpty(), "no Nodes in the cluster")

		for _, node := range nodes.Items {
			// target Pod at Node
			nodeSelection := e2epod.NodeSelection{Name: node.Name}
			e2epod.SetNodeSelection(&testPod.Spec, nodeSelection)
			_, err = pc.Create(ctx, &testPod, metav1.CreateOptions{})
			framework.ExpectNoError(err)
		}

		ginkgo.By("waiting for all of the no-snat-test pods to be scheduled and running")
		err = wait.PollImmediate(10*time.Second, 1*time.Minute, func() (bool, error) {
			pods, err := pc.List(ctx, metav1.ListOptions{LabelSelector: noSNATTestName})
			if err != nil {
				return false, err
			}

			// check all pods are running
			for _, pod := range pods.Items {
				if pod.Status.Phase != v1.PodRunning {
					if pod.Status.Phase != v1.PodPending {
						return false, fmt.Errorf("expected pod to be in phase \"Pending\" or \"Running\"")
					}
					return false, nil // pod is still pending
				}
			}
			return true, nil // all pods are running
		})
		framework.ExpectNoError(err)

		ginkgo.By("sending traffic from each pod to the others and checking that SNAT does not occur")
		pods, err := pc.List(ctx, metav1.ListOptions{LabelSelector: noSNATTestName})
		framework.ExpectNoError(err)

		// hit the /clientip endpoint on every other Pods to check if source ip is preserved
		// this test is O(n^2) but it doesn't matter because we only run this test on small clusters (~3 nodes)
		for _, sourcePod := range pods.Items {
			for _, targetPod := range pods.Items {
				if targetPod.Name == sourcePod.Name {
					continue
				}
				targetAddr := net.JoinHostPort(targetPod.Status.PodIP, testPodPort)
				sourceIP, execPodIP := execSourceIPTest(sourcePod, targetAddr)
				ginkgo.By("Verifying the preserved source ip")
				gomega.Expect(sourceIP).To(gomega.Equal(execPodIP))
			}
		}
	})
})
