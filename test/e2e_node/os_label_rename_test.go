//go:build linux

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

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("OSArchLabelReconciliation", framework.WithSerial(), framework.WithSlow(), framework.WithDisruptive(), func() {
	f := framework.NewDefaultFramework("node-label-reconciliation")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Context("Kubelet", func() {
		ginkgo.It("should reconcile the OS and Arch labels when restarted", func(ctx context.Context) {
			node := getLocalNode(ctx, f)
			e2enode.ExpectNodeHasLabel(ctx, f.ClientSet, node.Name, v1.LabelOSStable, runtime.GOOS)
			e2enode.ExpectNodeHasLabel(ctx, f.ClientSet, node.Name, v1.LabelArchStable, runtime.GOARCH)

			ginkgo.By("killing and restarting kubelet")
			// Let's kill the kubelet
			restartKubelet := mustStopKubelet(ctx, f)
			// Update labels
			newNode := node.DeepCopy()
			newNode.Labels[v1.LabelOSStable] = "dummyOS"
			newNode.Labels[v1.LabelArchStable] = "dummyArch"
			_, _, err := nodeutil.PatchNodeStatus(f.ClientSet.CoreV1(), types.NodeName(node.Name), node, newNode)
			framework.ExpectNoError(err)
			// Restart kubelet
			restartKubelet(ctx)
			framework.ExpectNoError(e2enode.WaitForAllNodesSchedulable(ctx, f.ClientSet, framework.RestartNodeReadyAgainTimeout))
			// If this happens right, node should have all the labels reset properly
			err = waitForNodeLabels(ctx, f.ClientSet.CoreV1(), node.Name, 5*time.Minute)
			framework.ExpectNoError(err)
		})
		ginkgo.It("should reconcile the OS and Arch labels when running", func(ctx context.Context) {

			node := getLocalNode(ctx, f)
			e2enode.ExpectNodeHasLabel(ctx, f.ClientSet, node.Name, v1.LabelOSStable, runtime.GOOS)
			e2enode.ExpectNodeHasLabel(ctx, f.ClientSet, node.Name, v1.LabelArchStable, runtime.GOARCH)

			// Update labels
			newNode := node.DeepCopy()
			newNode.Labels[v1.LabelOSStable] = "dummyOS"
			newNode.Labels[v1.LabelArchStable] = "dummyArch"
			_, _, err := nodeutil.PatchNodeStatus(f.ClientSet.CoreV1(), types.NodeName(node.Name), node, newNode)
			framework.ExpectNoError(err)
			err = waitForNodeLabels(ctx, f.ClientSet.CoreV1(), node.Name, 5*time.Minute)
			framework.ExpectNoError(err)
		})
	})
})

// waitForNodeLabels waits for the nodes to be have appropriate labels.
func waitForNodeLabels(ctx context.Context, c v1core.CoreV1Interface, nodeName string, timeout time.Duration) error {
	ginkgo.By(fmt.Sprintf("Waiting for node %v to have appropriate labels", nodeName))
	// Poll until the node has desired labels
	return wait.PollUntilContextTimeout(ctx, framework.Poll, timeout, false,
		func(ctx context.Context) (bool, error) {
			node, err := c.Nodes().Get(ctx, nodeName, metav1.GetOptions{})
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
