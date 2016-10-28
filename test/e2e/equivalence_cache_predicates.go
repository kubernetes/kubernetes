/*
Copyright 2016 The Kubernetes Authors.

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

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	_ "github.com/stretchr/testify/assert"
)

var _ = framework.KubeDescribe("EquivalenceCache [Serial]", func() {
	var cs clientset.Interface
	var nodeList *v1.NodeList
	var systemPodsNo int
	var ns string
	f := framework.NewDefaultFramework("equivalence-cache")
	ignoreLabels := framework.ImagePullerLabels

	BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
		nodeList = &v1.NodeList{}

		framework.WaitForAllNodesHealthy(cs, time.Minute)
		masterNodes, nodeList = framework.GetMasterAndWorkerNodesOrDie(cs)

		err := framework.CheckTestingNSDeletedExcept(cs, ns)
		framework.ExpectNoError(err)

		// Every test case in this suite assumes that cluster add-on pods stay stable and
		// cannot be run in parallel with any other test that touches Nodes or Pods.
		// It is so because we need to have precise control on what's running in the cluster.
		systemPods, err := framework.GetPodsInNamespace(cs, ns, ignoreLabels)
		Expect(err).NotTo(HaveOccurred())
		systemPodsNo = 0
		for _, pod := range systemPods {
			if !masterNodes.Has(pod.Spec.NodeName) && pod.DeletionTimestamp == nil {
				systemPodsNo++
			}
		}

		err = framework.WaitForPodsRunningReady(cs, api.NamespaceSystem, int32(systemPodsNo), framework.PodReadyBeforeTimeout, ignoreLabels, true)
		Expect(err).NotTo(HaveOccurred())

		for _, node := range nodeList.Items {
			framework.Logf("\nLogging pods the kubelet thinks is on node %v before test", node.Name)
			framework.PrintAllKubeletPods(cs, node.Name)
		}

	})

	// This test verifies that GenericPredicates works as expected:
	// When a replica pod (with HostPorts) is scheduled to a node, it will invalidate GenericPredicates cache on this node,
	// so that subsequent replica pods with same host port claim will be rejected.
	// We enforce all replica pods bind to the same node so there will always be conflicts.
	It("validates GenericPredicates is properly invalidated when a pod is scheduled [Slow]", func() {
		By("Launching a RC with two replica pods with HostPorts")
		nodeName := getNodeThatCanRunPod(f)

		// bind all replicas to same node
		nodeSelector := map[string]string{"kubernetes.io/hostname": nodeName}

		// CreateNodeSelectorPods creates RC with host port 4312
		CreateNodeSelectorPods(f, "host-port", 2, nodeSelector, false)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, ns, "host-port")

		By("One pod should be scheduled, the other should be rejected")
		waitForScheduler()
		// the first replica pod is scheduled, and the second pod will be rejected.
		verifyResult(cs, 1, 1, ns)
	})

	// This test verifies that MatchInterPodAffinity works as expected.
	// In equivalence cache, it dose not handle inter pod affinity (anti-affinity) specially (unless node label changed),
	// because current predicates algorithm will ensure newly scheduled pod does not break existing affinity in cluster.
	It("validates pod affinity works properly when new replica pod is scheduled", func() {
		// create a pod running with label {security: S1}, and choose this node
		nodeName, _ := runAndKeepPodWithLabelAndGetNodeName(f)

		By("Trying to apply a random label on the found node.")
		// we need to use real failure domains, since scheduler only know them
		k := "failure-domain.beta.kubernetes.io/zone"
		v := "equivalence-e2e-test"
		oldValue := framework.AddOrUpdateLabelOnNodeAndReturnOldValue(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		// restore the node label
		defer framework.AddOrUpdateLabelOnNode(cs, nodeName, k, oldValue)

		By("Trying to schedule RC with Pod Affinity should success.")
		framework.WaitForStableCluster(cs, masterNodes)
		affinityRCName := "with-pod-affinity-" + string(uuid.NewUUID())
		replica := 2
		labelsMap := map[string]string{
			"name": affinityRCName,
		}
		affinityString := `{
				"podAffinity": {
					"requiredDuringSchedulingIgnoredDuringExecution": [{
						"labelSelector": {
							"matchExpressions": [{
								"key": "security",
								"operator": "In",
								"values": ["S1"]
							}]
						},
						"topologyKey": "` + k + `",
						"namespaces":["` + ns + `"]
					}]
				}
			}`
		rc := getRCWithInterPodAffinity(affinityRCName, labelsMap, replica, affinityString, framework.GetPauseImageName(f.ClientSet))
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, ns, affinityRCName)

		_, err := cs.Core().ReplicationControllers(ns).Create(rc)
		framework.ExpectNoError(err)

		waitForScheduler()
		verifyReplicasResult(cs, replica, 0, ns, affinityRCName)

		By("Remove node failure domain label")
		framework.RemoveLabelOffNode(cs, nodeName, k)

		By("Trying to schedule another equivalent Pod should fail due to node label has been removed.")
		// use scale to create another equivalent pod
		framework.ScaleRC(f.ClientSet, f.InternalClientset, ns, affinityRCName, uint(replica+1), false)

		waitForScheduler()
		verifyReplicasResult(cs, replica, 1, ns, affinityRCName)
	})

	// This test verifies that MatchInterPodAffinity (anti-affinity) is respected as expected.
	It("validates pod anti-affinity works properly when new replica pod is scheduled", func() {
		By("Launching two pods on two distinct nodes to get two node names")
		CreateHostPortPods(f, "host-port", 2, true)
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, ns, "host-port")
		podList, err := cs.Core().Pods(ns).List(v1.ListOptions{})
		framework.ExpectNoError(err)
		Expect(len(podList.Items)).To(Equal(2))
		nodeNames := []string{podList.Items[0].Spec.NodeName, podList.Items[1].Spec.NodeName}
		Expect(nodeNames[0]).ToNot(Equal(nodeNames[1]))

		By("Applying a random label to both nodes.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "equivalence-e2etest"
		for _, nodeName := range nodeNames {
			framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
			framework.ExpectNodeHasLabel(cs, nodeName, k, v)
			defer framework.RemoveLabelOffNode(cs, nodeName, k)
		}

		By("Trying to launch a pod with the service label on the selected nodes.")
		// run a pod with label {"service": "S1"} and expect it to be running
		runPausePod(f, pausePodConfig{
			Name:         "with-label-" + string(uuid.NewUUID()),
			Labels:       map[string]string{"service": "S1"},
			NodeSelector: map[string]string{k: v}, // only launch on our two nodes
		})

		By("Trying to launch RC with podAntiAffinity on these two nodes should be rejected.")
		labelRCName := "with-podantiaffinity-" + string(uuid.NewUUID())
		replica := 2
		labelsMap := map[string]string{
			"name": labelRCName,
		}
		affinityString := `{
				"podAntiAffinity": {
					"requiredDuringSchedulingIgnoredDuringExecution": [{
						"labelSelector":{
							"matchExpressions": [{
								"key": "service",
								"operator": "In",
								"values": ["S1"]
							}]
						},
						"topologyKey": "` + k + `",
						"namespaces": ["` + ns + `"]
					}]
				}
			}`
		rc := getRCWithInterPodAffinityNodeSelector(labelRCName, labelsMap, replica, affinityString,
			framework.GetPauseImageName(f.ClientSet), map[string]string{k: v})
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, ns, labelRCName)

		_, err = cs.Core().ReplicationControllers(ns).Create(rc)
		framework.ExpectNoError(err)

		waitForScheduler()
		// these two replicas should all be rejected since podAntiAffinity says it they anit-affinity with pod {"service": "S1"}
		verifyReplicasResult(cs, 0, replica, ns, labelRCName)
	})
})

// getRCWithInterPodAffinity returns RC with given affinity rules.
func getRCWithInterPodAffinity(name string, labelsMap map[string]string, replica int, affinityString string, image string) *v1.ReplicationController {
	return getRCWithInterPodAffinityNodeSelector(name, labelsMap, replica, affinityString, image, map[string]string{})
}

// getRCWithInterPodAffinity returns RC with given affinity rules and node selector.
func getRCWithInterPodAffinityNodeSelector(name string, labelsMap map[string]string, replica int, affinityString string, image string, nodeSelector map[string]string) *v1.ReplicationController {
	replicaInt32 := int32(replica)
	annotations := map[string]string{
		v1.AffinityAnnotationKey: affinityString,
	}
	return &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &replicaInt32,
			Selector: labelsMap,
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      labelsMap,
					Annotations: annotations,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  name,
							Image: image,
						},
					},
					DNSPolicy:    v1.DNSDefault,
					NodeSelector: nodeSelector,
				},
			},
		},
	}
}
