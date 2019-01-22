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

package vsphere

import (
	"context"
	"os"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"github.com/vmware/govmomi/object"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("Node Unregister [Feature:vsphere] [Slow] [Disruptive]", func() {
	f := framework.NewDefaultFramework("node-unregister")
	var (
		client     clientset.Interface
		namespace  string
		workingDir string
		err        error
	)

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		namespace = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(client, framework.TestContext.NodeSchedulableTimeout))
		Expect(err).NotTo(HaveOccurred())
		workingDir = os.Getenv("VSPHERE_WORKING_DIR")
		Expect(workingDir).NotTo(BeEmpty())

	})

	It("node unregister", func() {
		By("Get total Ready nodes")
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(len(nodeList.Items) > 1).To(BeTrue(), "At least 2 nodes are required for this test")

		totalNodesCount := len(nodeList.Items)
		nodeVM := nodeList.Items[0]

		nodeInfo := TestContext.NodeMapper.GetNodeInfo(nodeVM.ObjectMeta.Name)
		vmObject := object.NewVirtualMachine(nodeInfo.VSphere.Client.Client, nodeInfo.VirtualMachineRef)

		// Find VM .vmx file path, host, resource pool.
		// They are required to register a node VM to VC
		vmxFilePath := getVMXFilePath(vmObject)

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		vmHost, err := vmObject.HostSystem(ctx)
		Expect(err).NotTo(HaveOccurred())

		vmPool, err := vmObject.ResourcePool(ctx)
		Expect(err).NotTo(HaveOccurred())

		// Unregister Node VM
		By("Unregister a node VM")
		unregisterNodeVM(nodeVM.ObjectMeta.Name, vmObject)

		// Ready nodes should be 1 less
		By("Verifying the ready node counts")
		Expect(verifyReadyNodeCount(f.ClientSet, totalNodesCount-1)).To(BeTrue(), "Unable to verify expected ready node count")

		nodeList = framework.GetReadySchedulableNodesOrDie(client)
		Expect(nodeList.Items).NotTo(BeEmpty(), "Unable to find ready and schedulable Node")

		var nodeNameList []string
		for _, node := range nodeList.Items {
			nodeNameList = append(nodeNameList, node.ObjectMeta.Name)
		}
		Expect(nodeNameList).NotTo(ContainElement(nodeVM.ObjectMeta.Name))

		// Register Node VM
		By("Register back the node VM")
		registerNodeVM(nodeVM.ObjectMeta.Name, workingDir, vmxFilePath, vmPool, vmHost)

		// Ready nodes should be equal to earlier count
		By("Verifying the ready node counts")
		Expect(verifyReadyNodeCount(f.ClientSet, totalNodesCount)).To(BeTrue(), "Unable to verify expected ready node count")

		nodeList = framework.GetReadySchedulableNodesOrDie(client)
		Expect(nodeList.Items).NotTo(BeEmpty(), "Unable to find ready and schedulable Node")

		nodeNameList = nodeNameList[:0]
		for _, node := range nodeList.Items {
			nodeNameList = append(nodeNameList, node.ObjectMeta.Name)
		}
		Expect(nodeNameList).To(ContainElement(nodeVM.ObjectMeta.Name))

		// Sanity test that pod provisioning works
		By("Sanity check for volume lifecycle")
		scParameters := make(map[string]string)
		storagePolicy := os.Getenv("VSPHERE_SPBM_GOLD_POLICY")
		Expect(storagePolicy).NotTo(BeEmpty(), "Please set VSPHERE_SPBM_GOLD_POLICY system environment")
		scParameters[SpbmStoragePolicy] = storagePolicy
		invokeValidPolicyTest(f, client, namespace, scParameters)
	})
})
