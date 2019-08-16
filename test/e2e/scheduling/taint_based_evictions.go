/*
Copyright 2018 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	clientset "k8s.io/client-go/kubernetes"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"

	"github.com/onsi/ginkgo"
)

func newUnreachableNoExecuteTaint() *v1.Taint {
	return &v1.Taint{
		Key:    schedulerapi.TaintNodeUnreachable,
		Effect: v1.TaintEffectNoExecute,
	}
}

func getTolerationSeconds(tolerations []v1.Toleration) (int64, error) {
	for _, t := range tolerations {
		if t.Key == schedulerapi.TaintNodeUnreachable && t.Effect == v1.TaintEffectNoExecute && t.Operator == v1.TolerationOpExists {
			return *t.TolerationSeconds, nil
		}
	}
	return 0, errors.New("cannot find toleration")
}

var _ = SIGDescribe("TaintBasedEvictions [Serial]", func() {
	f := framework.NewDefaultFramework("sched-taint-based-evictions")
	var cs clientset.Interface
	var ns string

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
		// skip if TaintBasedEvictions is not enabled
		// TODO(Huang-Wei): remove this when TaintBasedEvictions is GAed
		framework.SkipUnlessTaintBasedEvictionsEnabled()
		// it's required to run on a cluster that has more than 1 node
		// otherwise node lifecycle manager enters a fully disruption mode
		framework.SkipUnlessNodeCountIsAtLeast(2)
	})

	// This test verifies that when a node becomes unreachable
	// 1. node lifecycle manager generate a status change: [NodeReady=true, status=ConditionUnknown]
	// 1. it's applied with node.kubernetes.io/unreachable=:NoExecute taint
	// 2. pods without toleration are applied with toleration with tolerationSeconds=300
	// 3. pods with toleration and without tolerationSeconds won't be modified, and won't be evicted
	// 4. pods with toleration and with tolerationSeconds won't be modified, and will be evicted after tolerationSeconds
	// When network issue recovers, it's expected to see:
	// 5. node lifecycle manager generate a status change: [NodeReady=true, status=ConditionTrue]
	// 6. node.kubernetes.io/unreachable=:NoExecute taint is taken off the node
	ginkgo.It("Checks that the node becomes unreachable", func() {
		framework.SkipUnlessSSHKeyPresent()

		// find an available node
		nodeName := GetNodeThatCanRunPod(f)
		ginkgo.By("Finding an available node " + nodeName)

		// pod0 is a pod with unschedulable=:NoExecute toleration, and tolerationSeconds=0s
		// pod1 is a pod with unschedulable=:NoExecute toleration, and tolerationSeconds=200s
		// pod2 is a pod without any toleration
		base := "taint-based-eviction"
		tolerationSeconds := []int64{0, 200}
		numPods := len(tolerationSeconds) + 1
		ginkgo.By(fmt.Sprintf("Preparing %v pods", numPods))
		pods := make([]*v1.Pod, numPods)
		zero := int64(0)
		// build pod0, pod1
		for i := 0; i < numPods-1; i++ {
			pods[i] = createPausePod(f, pausePodConfig{
				Name:     fmt.Sprintf("%v-%v", base, i),
				NodeName: nodeName,
				Tolerations: []v1.Toleration{
					{
						Key:               schedulerapi.TaintNodeUnreachable,
						Operator:          v1.TolerationOpExists,
						Effect:            v1.TaintEffectNoExecute,
						TolerationSeconds: &tolerationSeconds[i],
					},
				},
				DeletionGracePeriodSeconds: &zero,
			})
		}
		// build pod2
		pods[numPods-1] = createPausePod(f, pausePodConfig{
			Name:     fmt.Sprintf("%v-%v", base, numPods-1),
			NodeName: nodeName,
		})

		ginkgo.By("Verifying all pods are running properly")
		for _, pod := range pods {
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(cs, pod))
		}

		// get the node API object
		nodeSelector := fields.OneTermEqualSelector("metadata.name", nodeName)
		nodeList, err := cs.CoreV1().Nodes().List(metav1.ListOptions{FieldSelector: nodeSelector.String()})
		if err != nil || len(nodeList.Items) != 1 {
			e2elog.Failf("expected no err, got %v; expected len(nodes) = 1, got %v", err, len(nodeList.Items))
		}
		node := nodeList.Items[0]

		ginkgo.By(fmt.Sprintf("Blocking traffic from node %s to the master", nodeName))
		host, err := e2enode.GetExternalIP(&node)
		if err != nil {
			host, err = e2enode.GetInternalIP(&node)
		}
		framework.ExpectNoError(err)
		masterAddresses := framework.GetAllMasterAddresses(cs)
		taint := newUnreachableNoExecuteTaint()

		defer func() {
			ginkgo.By(fmt.Sprintf("Unblocking traffic from node %s to the master", node.Name))
			for _, masterAddress := range masterAddresses {
				framework.UnblockNetwork(host, masterAddress)
			}

			if ginkgo.CurrentGinkgoTestDescription().Failed {
				e2elog.Failf("Current e2e test has failed, so return from here.")
				return
			}

			ginkgo.By(fmt.Sprintf("Expecting to see node %q becomes Ready", nodeName))
			e2enode.WaitForNodeToBeReady(cs, nodeName, time.Minute*1)
			ginkgo.By("Expecting to see unreachable=:NoExecute taint is taken off")
			err := framework.WaitForNodeHasTaintOrNot(cs, nodeName, taint, false, time.Second*30)
			framework.ExpectNoError(err)
		}()

		for _, masterAddress := range masterAddresses {
			framework.BlockNetwork(host, masterAddress)
		}

		ginkgo.By(fmt.Sprintf("Expecting to see node %q becomes NotReady", nodeName))
		if !e2enode.WaitForNodeToBeNotReady(cs, nodeName, time.Minute*3) {
			e2elog.Failf("node %q doesn't turn to NotReady after 3 minutes", nodeName)
		}
		ginkgo.By("Expecting to see unreachable=:NoExecute taint is applied")
		err = framework.WaitForNodeHasTaintOrNot(cs, nodeName, taint, true, time.Second*30)
		framework.ExpectNoError(err)

		ginkgo.By("Expecting pod0 to be evicted immediately")
		err = e2epod.WaitForPodCondition(cs, ns, pods[0].Name, "pod0 terminating", time.Second*15, func(pod *v1.Pod) (bool, error) {
			// as node is unreachable, pod0 is expected to be in Terminating status
			// rather than getting deleted
			if pod.DeletionTimestamp != nil {
				return true, nil
			}
			return false, nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("Expecting pod2 to be updated with a toleration with tolerationSeconds=300")
		err = e2epod.WaitForPodCondition(cs, ns, pods[2].Name, "pod2 updated with tolerationSeconds=300", time.Second*15, func(pod *v1.Pod) (bool, error) {
			if seconds, err := getTolerationSeconds(pod.Spec.Tolerations); err == nil {
				return seconds == 300, nil
			}
			return false, nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("Expecting pod1 to be unchanged")
		livePod1, err := cs.CoreV1().Pods(pods[1].Namespace).Get(pods[1].Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		seconds, err := getTolerationSeconds(livePod1.Spec.Tolerations)
		framework.ExpectNoError(err)
		if seconds != 200 {
			e2elog.Failf("expect tolerationSeconds of pod1 is 200, but got %v", seconds)
		}
	})
})
