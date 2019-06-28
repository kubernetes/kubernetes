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

package scheduling

import (
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	// ensure libs have a chance to initialize
	_ "github.com/stretchr/testify/assert"
)

const (
	defaultTimeout = 3 * time.Minute
)

var _ = framework.KubeDescribe("EquivalenceCache [Serial]", func() {
	var cs clientset.Interface
	var nodeList *v1.NodeList
	var masterNodes sets.String
	var systemPodsNo int
	var ns string
	f := framework.NewDefaultFramework("equivalence-cache")

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name

		e2enode.WaitForTotalHealthy(cs, time.Minute)
		masterNodes, nodeList = framework.GetMasterAndWorkerNodesOrDie(cs)

		framework.ExpectNoError(framework.CheckTestingNSDeletedExcept(cs, ns))

		// Every test case in this suite assumes that cluster add-on pods stay stable and
		// cannot be run in parallel with any other test that touches Nodes or Pods.
		// It is so because we need to have precise control on what's running in the cluster.
		systemPods, err := e2epod.GetPodsInNamespace(cs, ns, map[string]string{})
		framework.ExpectNoError(err)
		systemPodsNo = 0
		for _, pod := range systemPods {
			if !masterNodes.Has(pod.Spec.NodeName) && pod.DeletionTimestamp == nil {
				systemPodsNo++
			}
		}

		err = e2epod.WaitForPodsRunningReady(cs, api.NamespaceSystem, int32(systemPodsNo), int32(systemPodsNo), framework.PodReadyBeforeTimeout, map[string]string{})
		framework.ExpectNoError(err)

		for _, node := range nodeList.Items {
			e2elog.Logf("\nLogging pods the kubelet thinks is on node %v before test", node.Name)
			framework.PrintAllKubeletPods(cs, node.Name)
		}

	})

	/*
		Release : v1.16
		Testname: GeneralPredicates, invalidate
		Description: Create a pod which MUST be sheculed. Create Replica Controller with 2 replicas using the same node selector
		and the same host-port for both replicas. First replica MUST succeed and second one MUST be rejected.
	*/
	framework.ConformanceIt("validates GeneralPredicates is properly invalidated when a pod is scheduled [Slow]", func() {
		ginkgo.By("Launching a RC with two replica pods with HostPorts")
		nodeName := getNodeThatCanRunPodWithoutToleration(f)
		rcName := "host-port"

		// bind all replicas to same node
		nodeSelector := map[string]string{"kubernetes.io/hostname": nodeName}

		ginkgo.By("One pod should be scheduled, the other should be rejected")
		// CreateNodeSelectorPods creates RC with host port 4321
		WaitForSchedulerAfterAction(f, func() error {
			err := CreateNodeSelectorPods(f, rcName, 2, nodeSelector, false)
			return err
		}, ns, rcName, false)
		defer framework.DeleteRCAndWaitForGC(f.ClientSet, ns, rcName)
		// the first replica pod is scheduled, and the second pod will be rejected.
		verifyResult(cs, 1, 1, ns)
	})

	/*
		Release : v1.16
		Testname: Pod affinity, new replica pod
		Description: Launch pod with a label "security: S1". Update existing node label with a new value - "equivalence-e2e-test".
		Create Replication Controller with 2 replicas which MUST start successfully.
		Remove a label from the node. Create another equivalent pod and it MUST be rejected since the node label has been updated.
	*/
	framework.ConformanceIt("validates pod affinity works properly when new replica pod is scheduled", func() {
		// create a pod running with label {security: S1}, and choose this node
		nodeName, _ := runAndKeepPodWithLabelAndGetNodeName(f)

		ginkgo.By("Trying to apply a random label on the found node.")
		// we need to use real failure domains, since scheduler only know them
		k := "failure-domain.beta.kubernetes.io/zone"
		v := "equivalence-e2e-test"
		oldValue := framework.AddOrUpdateLabelOnNodeAndReturnOldValue(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		// restore the node label
		defer framework.AddOrUpdateLabelOnNode(cs, nodeName, k, oldValue)

		ginkgo.By("Trying to schedule RC with Pod Affinity should success.")
		framework.WaitForStableCluster(cs, masterNodes)
		affinityRCName := "with-pod-affinity-" + string(uuid.NewUUID())
		replica := 2
		labelsMap := map[string]string{
			"name": affinityRCName,
		}
		affinity := &v1.Affinity{
			PodAffinity: &v1.PodAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
					{
						LabelSelector: &metav1.LabelSelector{
							MatchExpressions: []metav1.LabelSelectorRequirement{
								{
									Key:      "security",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"S1"},
								},
							},
						},
						TopologyKey: k,
						Namespaces:  []string{ns},
					},
				},
			},
		}
		rc := getRCWithInterPodAffinity(affinityRCName, labelsMap, replica, affinity, imageutils.GetPauseImageName())
		defer framework.DeleteRCAndWaitForGC(f.ClientSet, ns, affinityRCName)

		// RC should be running successfully
		// TODO: WaitForSchedulerAfterAction() can on be used to wait for failure event,
		// not for successful RC, since no specific pod name can be provided.
		_, err := cs.CoreV1().ReplicationControllers(ns).Create(rc)
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitForControlledPodsRunning(cs, ns, affinityRCName, api.Kind("ReplicationController")))

		ginkgo.By("Remove node failure domain label")
		framework.RemoveLabelOffNode(cs, nodeName, k)

		ginkgo.By("Trying to schedule another equivalent Pod should fail due to node label has been removed.")
		// use scale to create another equivalent pod and wait for failure event
		WaitForSchedulerAfterAction(f, func() error {
			err := framework.ScaleRC(f.ClientSet, f.ScalesGetter, ns, affinityRCName, uint(replica+1), false)
			return err
		}, ns, affinityRCName, false)
		// and this new pod should be rejected since node label has been updated
		verifyReplicasResult(cs, replica, 1, ns, affinityRCName)
	})

	/*
		Release : v1.16
		Testname: Pod anti-affinity, two replicas rejected
		Description: Create 2 pods with a replication controller and they MUST be shecduled. Two different nodes MUST becreated.
		Apply the same label "equivalence-e2etest" for both nodes. Launch a pod with a label {"service": "S1"} on both nodes.
		Create two replica pods with the same label '"service": "S1"'. Both replicas MUST be rejected.
	*/
	framework.ConformanceIt("validates pod anti-affinity works properly when new replica pod is scheduled", func() {
		// check if there are at least 2 worker nodes available, else skip this test.
		if len(nodeList.Items) < 2 {
			framework.Skipf("Skipping as the test requires at least two worker nodes, current number of nodes: %d", len(nodeList.Items))
		}
		ginkgo.By("Launching two pods on two distinct nodes to get two node names")
		CreateHostPortPods(f, "host-port", 2, true)
		defer framework.DeleteRCAndWaitForGC(f.ClientSet, ns, "host-port")
		podList, err := cs.CoreV1().Pods(ns).List(metav1.ListOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(len(podList.Items)).To(gomega.Equal(2))
		nodeNames := []string{podList.Items[0].Spec.NodeName, podList.Items[1].Spec.NodeName}
		gomega.Expect(nodeNames[0]).ToNot(gomega.Equal(nodeNames[1]))

		ginkgo.By("Applying a random label to both nodes.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "equivalence-e2etest"
		for _, nodeName := range nodeNames {
			framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
			framework.ExpectNodeHasLabel(cs, nodeName, k, v)
			defer framework.RemoveLabelOffNode(cs, nodeName, k)
		}

		ginkgo.By("Trying to launch a pod with the service label on the selected nodes.")
		// run a pod with label {"service": "S1"} and expect it to be running
		runPausePod(f, pausePodConfig{
			Name:         "with-label-" + string(uuid.NewUUID()),
			Labels:       map[string]string{"service": "S1"},
			NodeSelector: map[string]string{k: v}, // only launch on our two nodes
		})

		ginkgo.By("Trying to launch RC with podAntiAffinity on these two nodes should be rejected.")
		labelRCName := "with-podantiaffinity-" + string(uuid.NewUUID())
		replica := 2
		labelsMap := map[string]string{
			"name": labelRCName,
		}
		affinity := &v1.Affinity{
			PodAntiAffinity: &v1.PodAntiAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
					{
						LabelSelector: &metav1.LabelSelector{
							MatchExpressions: []metav1.LabelSelectorRequirement{
								{
									Key:      "service",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"S1"},
								},
							},
						},
						TopologyKey: k,
						Namespaces:  []string{ns},
					},
				},
			},
		}
		rc := getRCWithInterPodAffinityNodeSelector(labelRCName, labelsMap, replica, affinity,
			imageutils.GetPauseImageName(), map[string]string{k: v})
		defer framework.DeleteRCAndWaitForGC(f.ClientSet, ns, labelRCName)

		WaitForSchedulerAfterAction(f, func() error {
			_, err := cs.CoreV1().ReplicationControllers(ns).Create(rc)
			return err
		}, ns, labelRCName, false)

		// these two replicas should all be rejected since podAntiAffinity says it they anti-affinity with pod {"service": "S1"}
		verifyReplicasResult(cs, 0, replica, ns, labelRCName)
	})
})

// getRCWithInterPodAffinity returns RC with given affinity rules.
func getRCWithInterPodAffinity(name string, labelsMap map[string]string, replica int, affinity *v1.Affinity, image string) *v1.ReplicationController {
	return getRCWithInterPodAffinityNodeSelector(name, labelsMap, replica, affinity, image, map[string]string{})
}

// getRCWithInterPodAffinity returns RC with given affinity rules and node selector.
func getRCWithInterPodAffinityNodeSelector(name string, labelsMap map[string]string, replica int, affinity *v1.Affinity, image string, nodeSelector map[string]string) *v1.ReplicationController {
	replicaInt32 := int32(replica)
	return &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &replicaInt32,
			Selector: labelsMap,
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labelsMap,
				},
				Spec: v1.PodSpec{
					Affinity: affinity,
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

// CreateNodeSelectorPods creates RC with host port 4321 and defines node selector
func CreateNodeSelectorPods(f *framework.Framework, id string, replicas int, nodeSelector map[string]string, expectRunning bool) error {
	ginkgo.By(fmt.Sprintf("Running RC which reserves host port and defines node selector"))

	config := &testutils.RCConfig{
		Client:       f.ClientSet,
		Name:         id,
		Namespace:    f.Namespace.Name,
		Timeout:      defaultTimeout,
		Image:        imageutils.GetPauseImageName(),
		Replicas:     replicas,
		HostPorts:    map[string]int{"port1": 4321},
		NodeSelector: nodeSelector,
	}
	err := framework.RunRC(*config)
	if expectRunning {
		return err
	}
	return nil
}
