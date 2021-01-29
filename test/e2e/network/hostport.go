/*
Copyright 2021 The Kubernetes Authors.

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
	"strconv"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var _ = SIGDescribe("HostPort", func() {

	f := framework.NewDefaultFramework("hostport")

	var (
		cs clientset.Interface
		ns string
	)

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name

	})

	/*
		Release: v1.16, v1.20
		Testname: Scheduling, HostPort matching and HostIP and Protocol not-matching
		Description: Pods with the same HostPort value MUST be able to be scheduled to the same node
		if the HostIP or Protocol is different. This test is marked LinuxOnly since hostNetwork is not supported on
		Windows.
	*/

	framework.ConformanceIt("validates that there is no conflict between pods with same hostPort but different hostIP and protocol [LinuxOnly]", func() {

		localhost := "127.0.0.1"
		family := v1.IPv4Protocol
		if framework.TestContext.ClusterIsIPv6() {
			localhost = "::1"
			family = v1.IPv6Protocol
		}
		// Get a node where to schedule the pods
		nodes, err := e2enode.GetBoundedReadySchedulableNodes(cs, 1)
		framework.ExpectNoError(err)
		if len(nodes.Items) == 0 {
			framework.Failf("No nodes available")

		}
		nodeName := nodes.Items[0].Name

		ips := e2enode.GetAddressesByTypeAndFamily(&nodes.Items[0], v1.NodeInternalIP, family)
		if len(ips) == 0 {
			framework.Failf("Failed to get NodeIP")
		}
		hostIP := ips[0]
		port := int32(54321)

		// Create pods with the same HostPort
		ginkgo.By(fmt.Sprintf("Trying to create a pod(pod1) with hostport %v and hostIP %s and expect scheduled", port, localhost))
		createHostPortPodOnNode(f, "pod1", ns, localhost, port, v1.ProtocolTCP, nodeName)

		ginkgo.By(fmt.Sprintf("Trying to create another pod(pod2) with hostport %v but hostIP %s on the node which pod1 resides and expect scheduled", port, hostIP))
		createHostPortPodOnNode(f, "pod2", ns, hostIP, port, v1.ProtocolTCP, nodeName)

		ginkgo.By(fmt.Sprintf("Trying to create a third pod(pod3) with hostport %v, hostIP %s but use UDP protocol on the node which pod2 resides", port, hostIP))
		createHostPortPodOnNode(f, "pod3", ns, hostIP, port, v1.ProtocolUDP, nodeName)

		// check that the port is being actually exposed to each container
		// create a pod on the host network in the same node
		hostExecPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "e2e-host-exec",
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				HostNetwork: true,
				NodeName:    nodeName,
				Containers: []v1.Container{
					{
						Name:  "e2e-host-exec",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
					},
				},
			},
		}
		f.PodClient().CreateSync(hostExecPod)

		// use a 5 seconds timeout per connection
		timeout := 5
		// IPv6 doesn't NAT from localhost -> localhost, it doesn't have the route_localnet kernel hack, so we need to specify the source IP
		cmdPod1 := []string{"/bin/sh", "-c", fmt.Sprintf("curl -g --connect-timeout %v --interface %s http://%s/hostname", timeout, hostIP, net.JoinHostPort(localhost, strconv.Itoa(int(port))))}
		cmdPod2 := []string{"/bin/sh", "-c", fmt.Sprintf("curl -g --connect-timeout %v http://%s/hostname", timeout, net.JoinHostPort(hostIP, strconv.Itoa(int(port))))}
		cmdPod3 := []string{"/bin/sh", "-c", fmt.Sprintf("nc -vuz -w %v %s %d", timeout, hostIP, port)}
		// try 5 times to connect to the exposed ports
		success := false
		for i := 0; i < 5; i++ {
			// check pod1
			ginkgo.By(fmt.Sprintf("checking connectivity from pod %s to serverIP: %s, port: %d", hostExecPod.Name, localhost, port))
			hostname1, _, err := f.ExecCommandInContainerWithFullOutput(hostExecPod.Name, "e2e-host-exec", cmdPod1...)
			if err != nil {
				framework.Logf("Can not connect from %s to pod(pod1) to serverIP: %s, port: %d", hostExecPod.Name, localhost, port)
				continue
			}
			// check pod2
			ginkgo.By(fmt.Sprintf("checking connectivity from pod %s to serverIP: %s, port: %d", hostExecPod.Name, hostIP, port))
			hostname2, _, err := f.ExecCommandInContainerWithFullOutput(hostExecPod.Name, "e2e-host-exec", cmdPod2...)
			if err != nil {
				framework.Logf("Can not connect from %s to pod(pod2) to serverIP: %s, port: %d", hostExecPod.Name, hostIP, port)
				continue
			}
			// the hostname returned has to be different because we are exposing the same port to two different pods
			if hostname1 == hostname2 {
				framework.Logf("pods must have different hostname: pod1 has hostname %s, pod2 has hostname %s", hostname1, hostname2)
				continue
			}
			// check pod3
			ginkgo.By(fmt.Sprintf("checking connectivity from pod %s to serverIP: %s, port: %d UDP", hostExecPod.Name, hostIP, port))
			_, _, err = f.ExecCommandInContainerWithFullOutput(hostExecPod.Name, "e2e-host-exec", cmdPod3...)
			if err != nil {
				framework.Logf("Can not connect from %s to pod(pod2) to serverIP: %s, port: %d", hostExecPod.Name, hostIP, port)
				continue
			}
			success = true
		}
		if !success {
			framework.Failf("Failed to connect to exposed host ports")
		}
	})
})

// create pod which using hostport on the specified node according to the nodeSelector
// it starts an http server on the exposed port
func createHostPortPodOnNode(f *framework.Framework, podName, ns, hostIP string, port int32, protocol v1.Protocol, nodeName string) {
	hostPortPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "agnhost",
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"netexec", "--http-port=8080", "--udp-port=8080"},
					Ports: []v1.ContainerPort{
						{
							HostPort:      port,
							ContainerPort: 8080,
							Protocol:      protocol,
							HostIP:        hostIP,
						},
					},
					ReadinessProbe: &v1.Probe{
						Handler: v1.Handler{
							HTTPGet: &v1.HTTPGetAction{
								Path: "/hostname",
								Port: intstr.IntOrString{
									IntVal: int32(8080),
								},
								Scheme: v1.URISchemeHTTP,
							},
						},
					},
				},
			},
			NodeName: nodeName,
		},
	}
	_, err := f.ClientSet.CoreV1().Pods(ns).Create(context.TODO(), hostPortPod, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	err = e2epod.WaitForPodNotPending(f.ClientSet, ns, podName)
	framework.ExpectNoError(err)
}
