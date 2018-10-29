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
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
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

	BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name

		framework.WaitForAllNodesHealthy(cs, time.Minute)
		masterNodes, nodeList = framework.GetMasterAndWorkerNodesOrDie(cs)

		framework.ExpectNoError(framework.CheckTestingNSDeletedExcept(cs, ns))

		// Every test case in this suite assumes that cluster add-on pods stay stable and
		// cannot be run in parallel with any other test that touches Nodes or Pods.
		// It is so because we need to have precise control on what's running in the cluster.
		systemPods, err := framework.GetPodsInNamespace(cs, ns, map[string]string{})
		Expect(err).NotTo(HaveOccurred())
		systemPodsNo = 0
		for _, pod := range systemPods {
			if !masterNodes.Has(pod.Spec.NodeName) && pod.DeletionTimestamp == nil {
				systemPodsNo++
			}
		}

		err = framework.WaitForPodsRunningReady(cs, api.NamespaceSystem, int32(systemPodsNo), int32(systemPodsNo), framework.PodReadyBeforeTimeout, map[string]string{})
		Expect(err).NotTo(HaveOccurred())

		for _, node := range nodeList.Items {
			framework.Logf("\nLogging pods the kubelet thinks is on node %v before test", node.Name)
			framework.PrintAllKubeletPods(cs, node.Name)
		}

	})

	// This test verifies that GeneralPredicates works as expected:
	// When a replica pod (with HostPorts) is scheduled to a node, it will invalidate GeneralPredicates cache on this node,
	// so that subsequent replica pods with same host port claim will be rejected.
	// We enforce all replica pods bind to the same node so there will always be conflicts.
	It("validates GeneralPredicates is properly invalidated when a pod is scheduled [Slow]", func() {
		By("Launching a RC with two replica pods with HostPorts")
		nodeName := getNodeThatCanRunPodWithoutToleration(f)
		rcName := "host-port"

		// bind all replicas to same node
		nodeSelector := map[string]string{"kubernetes.io/hostname": nodeName}

		By("One pod should be scheduled, the other should be rejected")
		// CreateNodeSelectorPods creates RC with host port 4312
		WaitForSchedulerAfterAction(f, func() error {
			err := CreateNodeSelectorPods(f, rcName, 2, nodeSelector, false)
			return err
		}, ns, rcName, false)
		defer framework.DeleteRCAndWaitForGC(f.ClientSet, ns, rcName)
		// the first replica pod is scheduled, and the second pod will be rejected.
		verifyResult(cs, 1, 1, ns)
	})

	// This test verifies that MatchInterPodAffinity works as expected.
	// In equivalence cache, it does not handle inter pod affinity (anti-affinity) specially (unless node label changed),
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
		framework.ExpectNoError(framework.WaitForControlledPodsRunning(cs, ns, affinityRCName, api.Kind("ReplicationController")))

		By("Remove node failure domain label")
		framework.RemoveLabelOffNode(cs, nodeName, k)

		By("Trying to schedule another equivalent Pod should fail due to node label has been removed.")
		// use scale to create another equivalent pod and wait for failure event
		WaitForSchedulerAfterAction(f, func() error {
			err := framework.ScaleRC(f.ClientSet, f.ScalesGetter, ns, affinityRCName, uint(replica+1), false)
			return err
		}, ns, affinityRCName, false)
		// and this new pod should be rejected since node label has been updated
		verifyReplicasResult(cs, replica, 1, ns, affinityRCName)
	})

	// This test verifies that MatchInterPodAffinity (anti-affinity) is respected as expected.
	It("validates pod anti-affinity works properly when new replica pod is scheduled", func() {
		By("Launching two pods on two distinct nodes to get two node names")
		CreateHostPortPods(f, "host-port", 2, true)
		defer framework.DeleteRCAndWaitForGC(f.ClientSet, ns, "host-port")
		podList, err := cs.CoreV1().Pods(ns).List(metav1.ListOptions{})
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

		// these two replicas should all be rejected since podAntiAffinity says it they anit-affinity with pod {"service": "S1"}
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

func CreateNodeSelectorPods(f *framework.Framework, id string, replicas int, nodeSelector map[string]string, expectRunning bool) error {
	By(fmt.Sprintf("Running RC which reserves host port and defines node selector"))

	config := &testutils.RCConfig{
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      f.Namespace.Name,
		Timeout:        defaultTimeout,
		Image:          imageutils.GetPauseImageName(),
		Replicas:       replicas,
		HostPorts:      map[string]int{"port1": 4321},
		NodeSelector:   nodeSelector,
	}
	err := framework.RunRC(*config)
	if expectRunning {
		return err
	}
	return nil
}
