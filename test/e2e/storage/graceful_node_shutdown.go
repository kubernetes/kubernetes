/*
Copyright 2022 The Kubernetes Authors.

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

package storage

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe(nodefeature.GracefulNodeShutdown, func() {
	var (
		c  clientset.Interface
		ns string
	)
	f := framework.NewDefaultFramework("graceful-shutdown")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func(ctx context.Context) {
		c = f.ClientSet
		ns = f.Namespace.Name
		e2eskipper.SkipUnlessProviderIs("gce")
		nodeList, err := e2enode.GetReadySchedulableNodes(ctx, c)
		framework.ExpectNoError(err, "Failed to list nodes")
		if len(nodeList.Items) < 2 {
			ginkgo.Skip("At least 2 nodes are required to run the test")
		}
	})

	ginkgo.Describe("Pod that uses a persistent volume via gce pd driver", func() {
		ginkgo.It("should get immediately rescheduled to a different node after graceful node shutdown", func(ctx context.Context) {
			ginkgo.By("deploying csi gce-pd driver")
			driver := drivers.InitGcePDCSIDriver()
			config := driver.PrepareTest(ctx, f)
			dDriver, ok := driver.(storageframework.DynamicPVTestDriver)
			if !ok {
				e2eskipper.Skipf("csi driver expected DynamicPVTestDriver but got %v", driver)
			}
			ginkgo.By("Creating a gce-pd storage class")
			sc := dDriver.GetDynamicProvisionStorageClass(ctx, config, "")
			_, err := c.StorageV1().StorageClasses().Create(ctx, sc, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create a storageclass")
			scName := &sc.Name

			deploymentName := "sts-pod-gcepd"
			podLabels := map[string]string{"app": deploymentName}
			pod := createAndVerifyStatefulDeployment(ctx, scName, deploymentName, ns, podLabels, c)
			oldNodeName := pod.Spec.NodeName

			ginkgo.By("Retrieving the node object where the pod is running")
			node, err := c.CoreV1().Nodes().Get(ctx, oldNodeName, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to get node object")

			ginkgo.By("Emitting shutdown signal to the node where the pod is running")
			err = emitShutdownSignal(ctx, node)
			framework.ExpectNoError(err)

			ginkgo.By(fmt.Sprintf("Checking if the pod %s got rescheduled to a new node", pod.Name))
			labelSelectorStr := labels.SelectorFromSet(podLabels).String()
			podListOpts := metav1.ListOptions{
				LabelSelector: labelSelectorStr,
				FieldSelector: fields.OneTermNotEqualSelector("spec.nodeName", oldNodeName).String(),
			}
			_, err = e2epod.WaitForPods(ctx, c, ns, podListOpts, e2epod.Range{MinMatching: 1}, framework.PodStartTimeout, "be running and ready", e2epod.RunningReady)
			framework.ExpectNoError(err)

			// Verify that a pod gets scheduled to the older node that was terminated and now is back online
			newDeploymentName := "sts-pod-gcepd-new"
			newPodLabels := map[string]string{"app": newDeploymentName}
			createAndVerifyStatefulDeployment(ctx, scName, newDeploymentName, ns, newPodLabels, c)
		})
	})
})

func emitShutdownSignal(ctx context.Context, node *v1.Node) error {
	cmd := "sudo systemctl poweroff"
	framework.Logf("Sending shutdown signal to node %s", node.Name)
	return e2essh.IssueSSHCommand(ctx, cmd, framework.TestContext.Provider, node)
}
