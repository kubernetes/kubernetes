/*
Copyright 2015 The Kubernetes Authors.

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

// Tests network performance using iperf or other containers.
import (
	"fmt"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	// empirically derived as a baseline for expectations from running this test using kube-up.sh.
	gceBandwidthBitsEstimate = int64(30000000000)
	// on 4 node clusters, we found this test passes very quickly, generally in less then 100 seconds.
	smallClusterTimeout = 200 * time.Second
)

// Declared as Flakey since it has not been proven to run in parallel on small nodes or slow networks in CI
var _ = SIGDescribe("Networking IPerf [Experimental] [Slow] [Feature:Networking-Performance]", func() {

	f := framework.NewDefaultFramework("network-perf")

	// A few simple bandwidth tests which are capped by nodes.
	// TODO replace the 1 with the scale option implementation
	// TODO: Make this a function parameter, once we distribute iperf endpoints, possibly via session affinity.
	numClient := 1
	numServer := 1
	maxBandwidthBits := gceBandwidthBitsEstimate

	familyStr := ""
	if framework.TestContext.ClusterIsIPv6() {
		familyStr = "-V "
	}

	ginkgo.It(fmt.Sprintf("should transfer ~ 1GB onto the service endpoint %v servers (maximum of %v clients)", numServer, numClient), func() {
		nodes, err := e2enode.GetReadySchedulableNodes(f.ClientSet)
		framework.ExpectNoError(err)
		totalPods := len(nodes.Items)
		// for a single service, we expect to divide bandwidth between the network.  Very crude estimate.
		expectedBandwidth := int(float64(maxBandwidthBits) / float64(totalPods))
		appName := "iperf-e2e"
		_, err = e2eservice.CreateServiceForSimpleAppWithPods(
			f.ClientSet,
			8001,
			8002,
			f.Namespace.Name,
			appName,
			func(n v1.Node) v1.PodSpec {
				return v1.PodSpec{
					Containers: []v1.Container{{
						Name:    "iperf-server",
						Image:   imageutils.GetE2EImage(imageutils.Agnhost),
						Command: []string{"/bin/sh"},
						Args: []string{
							"-c",
							"/usr/local/bin/iperf " + familyStr + "-s -p 8001 ",
						},
						Ports: []v1.ContainerPort{{ContainerPort: 8001}},
					}},
					NodeName:      n.Name,
					RestartPolicy: v1.RestartPolicyOnFailure,
				}
			},
			// this will be used to generate the -service name which all iperf clients point at.
			numServer, // Generally should be 1 server unless we do affinity or use a version of iperf that supports LB
			true,      // Make sure we wait, otherwise all the clients will die and need to restart.
		)

		if err != nil {
			framework.Failf("Fatal error waiting for iperf server endpoint : %v", err)
		}

		iperfClientPodLabels := e2enode.CreatePodsPerNodeForSimpleApp(
			f.ClientSet,
			f.Namespace.Name,
			"iperf-e2e-cli",
			func(n v1.Node) v1.PodSpec {
				return v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "iperf-client",
							Image:   imageutils.GetE2EImage(imageutils.Agnhost),
							Command: []string{"/bin/sh"},
							Args: []string{
								"-c",
								"/usr/local/bin/iperf " + familyStr + "-c service-for-" + appName + " -p 8002 --reportstyle C && sleep 5",
							},
						},
					},
					RestartPolicy: v1.RestartPolicyOnFailure, // let them successfully die.
				}
			},
			numClient,
		)
		expectedCli := numClient
		if len(nodes.Items) < expectedCli {
			expectedCli = len(nodes.Items)
		}

		framework.Logf("Reading all perf results to stdout.")
		framework.Logf("date,cli,cliPort,server,serverPort,id,interval,transferBits,bandwidthBits")

		// Extra 1/10 second per client.
		iperfTimeout := smallClusterTimeout + (time.Duration(expectedCli/10) * time.Second)
		iperfResults := &IPerfResults{}

		iperfClusterVerification := f.NewClusterVerification(
			f.Namespace,
			framework.PodStateVerification{
				Selectors:   iperfClientPodLabels,
				ValidPhases: []v1.PodPhase{v1.PodSucceeded},
			},
		)

		pods, err2 := iperfClusterVerification.WaitFor(expectedCli, iperfTimeout)
		if err2 != nil {
			framework.Failf("Error in wait...")
		} else if len(pods) < expectedCli {
			framework.Failf("IPerf restuls : Only got %v out of %v, after waiting %v", len(pods), expectedCli, iperfTimeout)
		} else {
			// For each builds up a collection of IPerfRecords
			iperfClusterVerification.ForEach(
				func(p v1.Pod) {
					resultS, err := framework.LookForStringInLog(f.Namespace.Name, p.Name, "iperf-client", "0-", 1*time.Second)
					if err == nil {
						framework.Logf(resultS)
						iperfResults.Add(NewIPerf(resultS))
					} else {
						framework.Failf("Unexpected error, %v when running forEach on the pods.", err)
					}
				})
		}
		fmt.Println("[begin] Node,Bandwidth CSV")
		fmt.Println(iperfResults.ToTSV())
		fmt.Println("[end] Node,Bandwidth CSV")

		for ipClient, bandwidth := range iperfResults.BandwidthMap {
			framework.Logf("%v had bandwidth %v.  Ratio to expected (%v) was %f", ipClient, bandwidth, expectedBandwidth, float64(bandwidth)/float64(expectedBandwidth))
		}
	})
})
