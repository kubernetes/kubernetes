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
	"net"
	"sync"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	// try to use a no common port so it doesn't conflict using hostNetwork
	testPodPort = "8085"
	// we have 2 tests that run in paralle with pods on hostNetwork
	// each test needs a different port to avoid scheduling port conflicts
	testPodPort2 = "8086"
	// maximum number of nodes uses for this test
	// limit the number of nodes to avoid duration issues on large clusters
	maxNodes       = 4
	noSNATTestName = "no-snat-test"
)

func createTestPod(port string, hostNetwork bool) *v1.Pod {
	return &v1.Pod{
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
					Args:  []string{"netexec", "--http-port", port},
				},
			},
			HostNetwork: hostNetwork,
		},
	}
}

// This test verifies that a Pod on each node in a cluster can talk to Pods on every other node without SNAT.
// Kubernetes imposes the following fundamental requirements on any networking implementation
// (barring any intentional network segmentation policies):
//
// pods on a node can communicate with all pods on all nodes without NAT
// agents on a node (e.g. system daemons, kubelet) can communicate with all pods on that node without NAT
// Note: For those platforms that support Pods running in the host network (e.g. Linux):
// pods in the host network of a node can communicate with all pods on all nodes without NAT
// xref: https://kubernetes.io/docs/concepts/cluster-administration/networking/
var _ = common.SIGDescribe("NoSNAT", func() {
	var (
		cs    clientset.Interface
		pc    v1core.PodInterface
		nodes *v1.NodeList
	)

	f := framework.NewDefaultFramework("no-snat-test")

	ginkgo.BeforeEach(func() {
		var err error
		cs = f.ClientSet
		pc = cs.CoreV1().Pods(f.Namespace.Name)
		nodes = &v1.NodeList{}
		nodes, err = e2enode.GetBoundedReadySchedulableNodes(cs, maxNodes)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			ginkgo.Skip("At least 2 nodes are required to run the test")
		}

	})

	ginkgo.It("Should be able to send traffic between Pods without SNAT", func() {
		framework.Logf("nodes DEBUG %v", nodes)
		ginkgo.By("creating a test pod on each Node")
		testPod := createTestPod(testPodPort, false)
		var wg sync.WaitGroup
		for _, node := range nodes.Items {
			// target Pod at Node
			ginkgo.By("creating pod on node " + node.Name)
			nodeSelection := e2epod.NodeSelection{Name: node.Name}
			e2epod.SetNodeSelection(&testPod.Spec, nodeSelection)
			wg.Add(1)
			go func() {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()
				f.PodClient().CreateSync(testPod)
			}()
		}
		wg.Wait()

		ginkgo.By("sending traffic from each pod to the others and checking that SNAT does not occur")
		pods, err := pc.List(context.TODO(), metav1.ListOptions{LabelSelector: noSNATTestName})
		framework.ExpectNoError(err)

		// hit the /clientip endpoint on every other Pods to check if source ip is preserved
		for _, sourcePod := range pods.Items {
			sourcePod := sourcePod
			for _, targetPod := range pods.Items {
				if targetPod.Name == sourcePod.Name {
					continue
				}
				targetAddr := net.JoinHostPort(targetPod.Status.PodIP, testPodPort)
				ginkgo.By("testing from pod " + sourcePod.Name + " to pod " + targetPod.Name)
				wg.Add(1)
				go func() {
					defer ginkgo.GinkgoRecover()
					defer wg.Done()
					sourceIP, execPodIP := execSourceIPTest(sourcePod, targetAddr)
					ginkgo.By("Verifying the preserved source ip")
					framework.ExpectEqual(sourceIP, execPodIP)
				}()
			}
		}
		wg.Wait()
	})

	ginkgo.It("Should be able to send traffic between Pods and an agent on that Node without SNAT", func() {
		ginkgo.By("creating a test pod on one Node")
		// use one node
		node := nodes.Items[0]
		// target Pod at Node

		testPod := createTestPod(testPodPort2, false)
		nodeSelection := e2epod.NodeSelection{Name: node.Name}
		e2epod.SetNodeSelection(&testPod.Spec, nodeSelection)
		f.PodClient().CreateSync(testPod)
		ginkgo.By("creating a hostnetwork test pod on the same Node")
		testPodHost := createTestPod(testPodPort2, true)
		e2epod.SetNodeSelection(&testPodHost.Spec, nodeSelection)
		f.PodClient().CreateSync(testPodHost)

		ginkgo.By("sending traffic from each pod to the others and checking that SNAT does not occur")
		pods, err := pc.List(context.TODO(), metav1.ListOptions{LabelSelector: noSNATTestName})
		framework.ExpectNoError(err)

		// hit the /clientip endpoint on every other Pods to check if source ip is preserved
		for _, sourcePod := range pods.Items {
			sourcePod := sourcePod
			for _, targetPod := range pods.Items {
				if targetPod.Name == sourcePod.Name {
					continue
				}
				targetAddr := net.JoinHostPort(targetPod.Status.PodIP, testPodPort2)
				ginkgo.By("testing from pod " + sourcePod.Name + " to pod " + targetPod.Name)

				sourceIP, execPodIP := execSourceIPTest(sourcePod, targetAddr)
				ginkgo.By("Verifying the preserved source ip")
				framework.ExpectEqual(sourceIP, execPodIP)
			}
		}

	})

	ginkgo.It("Should be able to send traffic between Pods and HostNetwork pods without SNAT [LinuxOnly]", func() {
		ginkgo.By("creating a test pod on each Node")
		// create a hostNetwork pod in one of the nodes
		testPodHost := createTestPod(testPodPort, true)
		nodeSelection := e2epod.NodeSelection{Name: nodes.Items[0].Name}
		e2epod.SetNodeSelection(&testPodHost.Spec, nodeSelection)
		f.PodClient().CreateSync(testPodHost)

		// create pods without hostNetwork in all nodes
		testPod := createTestPod(testPodPort, false)
		var wg sync.WaitGroup
		for _, node := range nodes.Items {
			// target Pod at Node
			ginkgo.By("creating pod on node " + node.Name)
			nodeSelection := e2epod.NodeSelection{Name: node.Name}
			e2epod.SetNodeSelection(&testPod.Spec, nodeSelection)
			wg.Add(1)
			go func() {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()
				f.PodClient().CreateSync(testPod)
			}()
		}
		wg.Wait()

		ginkgo.By("sending traffic from each pod to the others and checking that SNAT does not occur")
		pods, err := pc.List(context.TODO(), metav1.ListOptions{LabelSelector: noSNATTestName})
		framework.ExpectNoError(err)

		// hit the /clientip endpoint on every other Pods to check if source ip is preserved
		for _, sourcePod := range pods.Items {
			sourcePod := sourcePod
			for _, targetPod := range pods.Items {
				if targetPod.Name == sourcePod.Name {
					continue
				}
				targetAddr := net.JoinHostPort(targetPod.Status.PodIP, testPodPort)
				ginkgo.By("testing from pod " + sourcePod.Name + " to pod " + targetPod.Name)
				wg.Add(1)
				go func() {
					defer ginkgo.GinkgoRecover()
					defer wg.Done()
					sourceIP, execPodIP := execSourceIPTest(sourcePod, targetAddr)
					ginkgo.By("Verifying the preserved source ip")
					framework.ExpectEqual(sourceIP, execPodIP)
				}()
			}
		}
		wg.Wait()
	})

})
