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

package e2e

// Tests network performance using iperf or other containers.
import (
	"fmt"
	"math"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// empirically derived as a baseline for expectations from running this test using kube-up.sh.
	gceBandwidthBitsEstimate = int64(30000000000)
	// on 4 node clusters, we found this test passes very quickly, generally in less then 100 seconds.
	smallClusterTimeout = 200 * time.Second
)

// Declared as Flakey since it has not been proven to run in parallel on small nodes or slow networks in CI
// TODO jayunit100 : Retag this test according to semantics from #22401
var _ = framework.KubeDescribe("Networking IPerf [Experimental] [Slow] [Feature:Networking-Performance]", func() {

	f := framework.NewDefaultFramework("network-perf")

	// A few simple bandwidth tests which are capped by nodes.
	// TODO replace the 1 with the scale option implementation
	// TODO: Make this a function parameter, once we distribute iperf endpoints, possibly via session affinity.
	numClient := 1
	numServer := 1
	maxBandwidthBits := gceBandwidthBitsEstimate

	It(fmt.Sprintf("should transfer ~ 1GB onto the service endpoint %v servers (maximum of %v clients)", numServer, numClient), func() {
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		totalPods := len(nodes.Items)
		// for a single service, we expect to divide bandwidth between the network.  Very crude estimate.
		expectedBandwidth := int(float64(maxBandwidthBits) / float64(totalPods))
		Expect(totalPods).NotTo(Equal(0))
		appName := "iperf-e2e"
		err, _ := f.CreateServiceForSimpleAppWithPods(
			8001,
			8002,
			appName,
			func(n api.Node) api.PodSpec {
				return api.PodSpec{
					Containers: []api.Container{{
						Name:  "iperf-server",
						Image: "gcr.io/google_containers/iperf:e2e",
						Args: []string{
							"/bin/sh",
							"-c",
							"/usr/local/bin/iperf -s -p 8001 ",
						},
						Ports: []api.ContainerPort{{ContainerPort: 8001}},
					}},
					NodeName:      n.Name,
					RestartPolicy: api.RestartPolicyOnFailure,
				}
			},
			// this will be used to generate the -service name which all iperf clients point at.
			numServer, // Generally should be 1 server unless we do affinity or use a version of iperf that supports LB
			true,      // Make sure we wait, otherwise all the clients will die and need to restart.
		)

		if err != nil {
			framework.Failf("Fatal error waiting for iperf server endpoint : %v", err)
		}

		iperfClientPodLabels := f.CreatePodsPerNodeForSimpleApp(
			"iperf-e2e-cli",
			func(n api.Node) api.PodSpec {
				return api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "iperf-client",
							Image: "gcr.io/google_containers/iperf:e2e",
							Args: []string{
								"/bin/sh",
								"-c",
								"/usr/local/bin/iperf -c service-for-" + appName + " -p 8002 --reportstyle C && sleep 5",
							},
						},
					},
					RestartPolicy: api.RestartPolicyOnFailure, // let them successfully die.
				}
			},
			numClient,
		)

		framework.Logf("Reading all perf results to stdout.")
		framework.Logf("date,cli,cliPort,server,serverPort,id,interval,transferBits,bandwidthBits")

		// Calculate expected number of clients based on total nodes.
		expectedCli := func() int {
			nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
			return int(math.Min(float64(len(nodes.Items)), float64(numClient)))
		}()

		// Extra 1/10 second per client.
		iperfTimeout := smallClusterTimeout + (time.Duration(expectedCli/10) * time.Second)
		iperfResults := &IPerfResults{}

		iperfClusterVerification := f.NewClusterVerification(
			framework.PodStateVerification{
				Selectors:   iperfClientPodLabels,
				ValidPhases: []api.PodPhase{api.PodSucceeded},
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
				func(p api.Pod) {
					resultS, err := framework.LookForStringInLog(f.Namespace.Name, p.Name, "iperf-client", "0-", 1*time.Second)
					if err == nil {
						framework.Logf(resultS)
						iperfResults.Add(NewIPerf(resultS))
					} else {
						framework.Failf("Unexpected error, %v when running forEach on the pods.", err)
					}
				})
		}
		fmt.Println("[begin] Node,Bandwith CSV")
		fmt.Println(iperfResults.ToTSV())
		fmt.Println("[end] Node,Bandwith CSV")

		for ipClient, bandwidth := range iperfResults.BandwidthMap {
			framework.Logf("%v had bandwidth %v.  Ratio to expected (%v) was %f", ipClient, bandwidth, expectedBandwidth, float64(bandwidth)/float64(expectedBandwidth))
		}
	})
})
