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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/vmware/govmomi/object"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("Node Unregister [Feature:vsphere] [Slow] [Disruptive]", func() {
	f := framework.NewDefaultFramework("node-unregister")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	var (
		client     clientset.Interface
		namespace  string
		workingDir string
		err        error
	)

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		namespace = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(client, framework.TestContext.NodeSchedulableTimeout))
		framework.ExpectNoError(err)
		workingDir = GetAndExpectStringEnvVar("VSPHERE_WORKING_DIR")
	})

	ginkgo.It("node unregister", func() {
		ginkgo.By("Get total Ready nodes")
		nodeList, err := e2enode.GetReadySchedulableNodes(f.ClientSet)
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(nodeList.Items) > 1, true, "At least 2 nodes are required for this test")

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
		framework.ExpectNoError(err)

		vmPool, err := vmObject.ResourcePool(ctx)
		framework.ExpectNoError(err)

		// Unregister Node VM
		ginkgo.By("Unregister a node VM")
		unregisterNodeVM(nodeVM.ObjectMeta.Name, vmObject)

		// Ready nodes should be 1 less
		ginkgo.By("Verifying the ready node counts")
		framework.ExpectEqual(verifyReadyNodeCount(f.ClientSet, totalNodesCount-1), true, "Unable to verify expected ready node count")

		nodeList, err = e2enode.GetReadySchedulableNodes(client)
		framework.ExpectNoError(err)

		var nodeNameList []string
		for _, node := range nodeList.Items {
			nodeNameList = append(nodeNameList, node.ObjectMeta.Name)
		}
		gomega.Expect(nodeNameList).NotTo(gomega.ContainElement(nodeVM.ObjectMeta.Name))

		// Register Node VM
		ginkgo.By("Register back the node VM")
		registerNodeVM(nodeVM.ObjectMeta.Name, workingDir, vmxFilePath, vmPool, vmHost)

		// Ready nodes should be equal to earlier count
		ginkgo.By("Verifying the ready node counts")
		framework.ExpectEqual(verifyReadyNodeCount(f.ClientSet, totalNodesCount), true, "Unable to verify expected ready node count")

		nodeList, err = e2enode.GetReadySchedulableNodes(client)
		framework.ExpectNoError(err)

		nodeNameList = nodeNameList[:0]
		for _, node := range nodeList.Items {
			nodeNameList = append(nodeNameList, node.ObjectMeta.Name)
		}
		gomega.Expect(nodeNameList).To(gomega.ContainElement(nodeVM.ObjectMeta.Name))

		// Sanity test that pod provisioning works
		ginkgo.By("Sanity check for volume lifecycle")
		scParameters := make(map[string]string)
		storagePolicy := GetAndExpectStringEnvVar("VSPHERE_SPBM_GOLD_POLICY")
		scParameters[SpbmStoragePolicy] = storagePolicy
		invokeValidPolicyTest(f, client, namespace, scParameters)
	})
})
