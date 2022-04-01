//go:build cgo && linux
// +build cgo,linux

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

package e2enode

import (
	"context"
	"fmt"
	"runtime"
	"time"

	e2eutils "k8s.io/kubernetes/test/e2e/framework/utils"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = SIGDescribe("OSArchLabelReconciliation [Serial] [Slow] [Disruptive]", func() {
	f := framework.NewDefaultFramework("node-label-reconciliation")
	ginkgo.Context("Kubelet", func() {
		ginkgo.It("should reconcile the OS and Arch labels when restarted", func() {
			node := getLocalNode(f)
			e2eutils.ExpectNodeHasLabel(f.ClientSet, node.Name, v1.LabelOSStable, runtime.GOOS)
			e2eutils.ExpectNodeHasLabel(f.ClientSet, node.Name, v1.LabelArchStable, runtime.GOARCH)

			ginkgo.By("killing and restarting kubelet")
			// Let's kill the kubelet
			startKubelet := stopKubelet()
			// Update labels
			newNode := node.DeepCopy()
			newNode.Labels[v1.LabelOSStable] = "dummyOS"
			newNode.Labels[v1.LabelArchStable] = "dummyArch"
			_, _, err := nodeutil.PatchNodeStatus(f.ClientSet.CoreV1(), types.NodeName(node.Name), node, newNode)
			e2eutils.ExpectNoError(err)
			// Restart kubelet
			startKubelet()
			e2eutils.ExpectNoError(e2eutils.WaitForAllNodesSchedulable(f.ClientSet, e2eutils.RestartNodeReadyAgainTimeout))
			// If this happens right, node should have all the labels reset properly
			err = waitForNodeLabels(f.ClientSet.CoreV1(), node.Name, 5*time.Minute)
			e2eutils.ExpectNoError(err)
		})
		ginkgo.It("should reconcile the OS and Arch labels when running", func() {

			node := getLocalNode(f)
			e2eutils.ExpectNodeHasLabel(f.ClientSet, node.Name, v1.LabelOSStable, runtime.GOOS)
			e2eutils.ExpectNodeHasLabel(f.ClientSet, node.Name, v1.LabelArchStable, runtime.GOARCH)

			// Update labels
			newNode := node.DeepCopy()
			newNode.Labels[v1.LabelOSStable] = "dummyOS"
			newNode.Labels[v1.LabelArchStable] = "dummyArch"
			_, _, err := nodeutil.PatchNodeStatus(f.ClientSet.CoreV1(), types.NodeName(node.Name), node, newNode)
			e2eutils.ExpectNoError(err)
			err = waitForNodeLabels(f.ClientSet.CoreV1(), node.Name, 5*time.Minute)
			e2eutils.ExpectNoError(err)
		})
	})
})

// waitForNodeLabels waits for the nodes to be have appropriate labels.
func waitForNodeLabels(c v1core.CoreV1Interface, nodeName string, timeout time.Duration) error {
	ginkgo.By(fmt.Sprintf("Waiting for node %v to have appropriate labels", nodeName))
	// Poll until the node has desired labels
	return wait.Poll(e2eutils.Poll, timeout,
		func() (bool, error) {
			node, err := c.Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			osLabel, ok := node.Labels[v1.LabelOSStable]
			if !ok || osLabel != runtime.GOOS {
				return false, nil
			}
			archLabel, ok := node.Labels[v1.LabelArchStable]
			if !ok || archLabel != runtime.GOARCH {
				return false, nil
			}
			return true, nil
		})
}
