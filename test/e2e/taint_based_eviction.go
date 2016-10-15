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
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Eviction based on taints [Serial] [Slow] [Destructive]", func() {
	const pollTimeout = 1 * time.Minute

	var cs clientset.Interface
	f := framework.NewDefaultFramework("pod-eviction")

	It("validates that unreachable taint can be auto added/removed [Feature:forgiveness]", func() {
		cs = f.ClientSet
		nodeName := getNodeThatCanRunPod(f)

		node, err := cs.Core().Nodes().Get(nodeName)
		framework.ExpectNoError(err)

		// switch the network interface off for a while to simulate a network outage
		// We sleep 10 seconds to give some time for ssh command to cleanly finish before network is down.
		executeCmd := "nohup sh -c 'sleep 10 && (sudo ifdown eth0 || sudo ip link set eth0 down) &" +
			"& sleep 120 && (sudo ifup eth0 || sudo ip link set eth0 up)' >/dev/null 2>&1 &"
		err = framework.IssueSSHCommand(executeCmd, framework.TestContext.Provider, node)
		framework.ExpectNoError(err, "Error while issuing ssh command")

		unreachableTaint := api.Taint{
			Key:    unversioned.TaintNodeUnreachable,
			Effect: api.TaintEffectNoExecute,
		}
		framework.ExpectNodeHasTaint(cs, nodeName, unreachableTaint)

		isNodeReady := !framework.WaitForNodeToBeReady(cs, nodeName, framework.NodeReadyInitialTimeout)
		Expect(isNodeReady).To(BeTrue())

		framework.ExpectNodeDoesNotHaveTaint(cs, nodeName, unreachableTaint)
	})

	It("validates that pod evicted by nodecontroller when NoExectue taint added to node", func() {
		cs = f.ClientSet
		nodeName, podName := runAndKeepPodWithLabelAndGetNodeName(f)

		By("Trying to apply a NoExectue taint on the found node.")
		testTaint := api.Taint{
			Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-%s", string(uuid.NewUUID())),
			Value:  "testing-taint-value",
			Effect: api.TaintEffectNoExecute,
		}
		framework.AddOrUpdateTaintOnNode(cs, nodeName, testTaint)
		framework.ExpectNodeHasTaint(cs, nodeName, testTaint)
		defer framework.RemoveTaintOffNode(cs, nodeName, testTaint)

		Expect(framework.WaitForPodToDisappear(cs, f.Namespace.Name, podName, labels.Everything(), pollInterval, pollTimeout)).To(Succeed())
	})
})
