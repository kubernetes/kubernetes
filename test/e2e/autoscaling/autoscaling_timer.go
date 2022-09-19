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

package autoscaling

import (
	"context"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eautoscaling "k8s.io/kubernetes/test/e2e/framework/autoscaling"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega/gmeasure"
)

var _ = SIGDescribe("[Feature:ClusterSizeAutoscalingScaleUp] [Slow] Autoscaling", func() {
	f := framework.NewDefaultFramework("autoscaling")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	var experiment *gmeasure.Experiment

	ginkgo.Describe("Autoscaling a service", func() {
		ginkgo.BeforeEach(func() {
			// Check if Cloud Autoscaler is enabled by trying to get its ConfigMap.
			_, err := f.ClientSet.CoreV1().ConfigMaps("kube-system").Get(context.TODO(), "cluster-autoscaler-status", metav1.GetOptions{})
			if err != nil {
				e2eskipper.Skipf("test expects Cluster Autoscaler to be enabled")
			}
			experiment = gmeasure.NewExperiment("Autoscaling a service")
			ginkgo.AddReportEntry(experiment.Name, experiment)
		})

		ginkgo.Context("from 1 pod and 3 nodes to 8 pods and >=4 nodes", func() {
			const nodesNum = 3       // Expect there to be 3 nodes before and after the test.
			var nodeGroupName string // Set by BeforeEach, used by AfterEach to scale this node group down after the test.
			var nodes *v1.NodeList   // Set by BeforeEach, used by Measure to calculate CPU request based on node's sizes.

			ginkgo.BeforeEach(func() {
				// Make sure there is only 1 node group, otherwise this test becomes useless.
				nodeGroups := strings.Split(framework.TestContext.CloudConfig.NodeInstanceGroup, ",")
				if len(nodeGroups) != 1 {
					e2eskipper.Skipf("test expects 1 node group, found %d", len(nodeGroups))
				}
				nodeGroupName = nodeGroups[0]

				// Make sure the node group has exactly 'nodesNum' nodes, otherwise this test becomes useless.
				nodeGroupSize, err := framework.GroupSize(nodeGroupName)
				framework.ExpectNoError(err)
				if nodeGroupSize != nodesNum {
					e2eskipper.Skipf("test expects %d nodes, found %d", nodesNum, nodeGroupSize)
				}

				// Make sure all nodes are schedulable, otherwise we are in some kind of a problem state.
				nodes, err = e2enode.GetReadySchedulableNodes(f.ClientSet)
				framework.ExpectNoError(err)
				schedulableCount := len(nodes.Items)
				framework.ExpectEqual(schedulableCount, nodeGroupSize, "not all nodes are schedulable")
			})

			ginkgo.AfterEach(func() {
				// Attempt cleanup only if a node group was targeted for scale up.
				// Otherwise the test was probably skipped and we'll get a gcloud error due to invalid parameters.
				if len(nodeGroupName) > 0 {
					// Scale down back to only 'nodesNum' nodes, as expected at the start of the test.
					framework.ExpectNoError(framework.ResizeGroup(nodeGroupName, nodesNum))
					framework.ExpectNoError(e2enode.WaitForReadyNodes(f.ClientSet, nodesNum, 15*time.Minute))
				}
			})

			ginkgo.It("takes less than 15 minutes", func() {
				// Measured over multiple samples, scaling takes 10 +/- 2 minutes, so 15 minutes should be fully sufficient.
				const timeToWait = 15 * time.Minute

				// Calculate the CPU request of the service.
				// This test expects that 8 pods will not fit in 'nodesNum' nodes, but will fit in >='nodesNum'+1 nodes.
				// Make it so that 'nodesNum' pods fit perfectly per node.
				nodeCpus := nodes.Items[0].Status.Allocatable[v1.ResourceCPU]
				nodeCPUMillis := (&nodeCpus).MilliValue()
				cpuRequestMillis := int64(nodeCPUMillis / nodesNum)

				// Start the service we want to scale and wait for it to be up and running.
				nodeMemoryBytes := nodes.Items[0].Status.Allocatable[v1.ResourceMemory]
				nodeMemoryMB := (&nodeMemoryBytes).Value() / 1024 / 1024
				memRequestMB := nodeMemoryMB / 10 // Ensure each pod takes not more than 10% of node's allocatable memory.
				replicas := 1
				resourceConsumer := e2eautoscaling.NewDynamicResourceConsumer("resource-consumer", f.Namespace.Name, e2eautoscaling.KindDeployment, replicas, 0, 0, 0, cpuRequestMillis, memRequestMB, f.ClientSet, f.ScalesGetter, e2eautoscaling.Disable, e2eautoscaling.Idle)
				defer resourceConsumer.CleanUp()
				resourceConsumer.WaitForReplicas(replicas, 1*time.Minute) // Should finish ~immediately, so 1 minute is more than enough.

				// Enable Horizontal Pod Autoscaler with 50% target utilization and
				// scale up the CPU usage to trigger autoscaling to 8 pods for target to be satisfied.
				targetCPUUtilizationPercent := int32(50)
				hpa := e2eautoscaling.CreateCPUResourceHorizontalPodAutoscaler(resourceConsumer, targetCPUUtilizationPercent, 1, 10)
				defer e2eautoscaling.DeleteHorizontalPodAutoscaler(resourceConsumer, hpa.Name)
				cpuLoad := 8 * cpuRequestMillis * int64(targetCPUUtilizationPercent) / 100 // 8 pods utilized to the target level
				resourceConsumer.ConsumeCPU(int(cpuLoad))

				// Measure the time it takes for the service to scale to 8 pods with 50% CPU utilization each.
				experiment.SampleDuration("total scale-up time", func(idx int) {
					resourceConsumer.WaitForReplicas(8, timeToWait)
				}, gmeasure.SamplingConfig{N: 1})
			}) // Increase to run the test more than once.
		})
	})
})
