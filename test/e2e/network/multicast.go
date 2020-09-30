/*
Copyright 2020 The Kubernetes Authors.

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
	"strings"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
)

const (
	mcastSource  = "pod-client"
	mcastServer1 = "pod-server1"
	mcastServer2 = "pod-server2"
)

var _ = SIGDescribe("Multicast", func() {

	fr := framework.NewDefaultFramework("multicast")

	type nodeInfo struct {
		name   string
		nodeIP string
	}

	var (
		cs                             clientset.Interface
		ns                             string
		clientNodeInfo, serverNodeInfo nodeInfo
	)

	ginkgo.BeforeEach(func() {
		cs = fr.ClientSet
		ns = fr.Namespace.Name

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf(
				"Test requires >= 2 Ready nodes, but there are only %v nodes",
				len(nodes.Items))
		}

		ips := e2enode.CollectAddresses(nodes, v1.NodeInternalIP)

		clientNodeInfo = nodeInfo{
			name:   nodes.Items[0].Name,
			nodeIP: ips[0],
		}

		serverNodeInfo = nodeInfo{
			name:   nodes.Items[1].Name,
			nodeIP: ips[1],
		}
	})

	ginkgo.It("should be able to send multicast UDP traffic between nodes [Experimental]", func() {
		mcastGroup := "224.3.3.3"
		mcastGroupBad := "224.5.5.5"
		mcastPrefix := "224"
		if framework.TestContext.ClusterIsIPv6() {
			mcastGroup = "ff00:0:3:3::3"
			mcastGroupBad = "ff00:0:5:5::5"
			mcastPrefix = "ff00"
		}

		// Start the multicast source (iperf client is the sender in multicast)
		ginkgo.By("creating a pod as a multicast source in node " + clientNodeInfo.name)
		// multicast group (-c 224.3.3.3), UDP (-u), TTL (-T 2), during (-t 3000) seconds, report every (-i 5) seconds
		iperf := []string{"iperf", "-c", mcastGroup, "-u", "-T", "2", "-t", "3000", "-i", "5"}
		if framework.TestContext.ClusterIsIPv6() {
			iperf = append(iperf, "-V")
		}
		clientPod := newAgnhostPod(mcastSource, iperf...)
		clientPod.Spec.NodeName = clientNodeInfo.name
		fr.PodClient().CreateSync(clientPod)

		// Start a multicast listener on the same groups and verify it received the traffic (iperf server is the multicast listener)
		// join multicast group (-B 224.3.3.3), UDP (-u), during (-t 10) seconds, report every (-i 1) seconds
		ginkgo.By("creating first multicast listener pod in node " + serverNodeInfo.name)
		iperf = []string{"iperf", "-s", "-B", mcastGroup, "-u", "-t", "10", "-i", "5"}
		if framework.TestContext.ClusterIsIPv6() {
			iperf = append(iperf, "-V")
		}
		mcastServerPod1 := newAgnhostPod(mcastServer1, iperf...)
		mcastServerPod1.Spec.NodeName = serverNodeInfo.name
		fr.PodClient().CreateSync(mcastServerPod1)

		// Start a multicast listener on on other group and verify it does not receive the traffic (iperf server is the multicast listener)
		// join multicast group (-B 224.4.4.4), UDP (-u), during (-t 10) seconds, report every (-i 1) seconds
		ginkgo.By("creating second multicast listener pod in node " + serverNodeInfo.name)
		iperf = []string{"iperf", "-s", "-B", mcastGroupBad, "-u", "-t", "10", "-i", "5"}
		if framework.TestContext.ClusterIsIPv6() {
			iperf = append(iperf, "-V")
		}
		mcastServerPod2 := newAgnhostPod(mcastServer2, iperf...)
		mcastServerPod2.Spec.NodeName = serverNodeInfo.name
		fr.PodClient().CreateSync(mcastServerPod2)

		// wait for pods to finish
		ginkgo.By("waiting for pods to finish")
		framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(cs, mcastServer1, ns))
		framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(cs, mcastServer2, ns))

		ginkgo.By("checking if pod server1 received multicast traffic")
		logs, err := e2epod.GetPodLogs(cs, ns, mcastServer1, mcastServer1)
		framework.ExpectNoError(err)
		if !strings.Contains(string(logs), mcastPrefix) {
			framework.Logf("Pod %s logs: %s", mcastServer1, logs)
			framework.Failf("Failed to receive multicast on server 1")
		}

		ginkgo.By("checking if pod server2 does not received multicast traffic")
		logs, err = e2epod.GetPodLogs(cs, ns, mcastServer2, mcastServer2)
		framework.ExpectNoError(err)
		if strings.Contains(string(logs), mcastPrefix) {
			framework.Logf("Pod %s logs: %s", mcastServer2, logs)
			framework.Failf("Error, server 2 should not receive any multicast traffic")
		}
	})

})
