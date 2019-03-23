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
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

/*
	Tests to verify volume provisioning on a clustered datastore
	1. Static provisioning
	2. Dynamic provisioning
	3. Dynamic provisioning with spbm policy

	This test reads env
	1. CLUSTER_DATASTORE which should be set to clustered datastore
	2. VSPHERE_SPBM_POLICY_DS_CLUSTER which should be set to a tag based spbm policy tagged to a clustered datastore
*/
var _ = utils.SIGDescribe("Volume Provisioning On Clustered Datastore [Feature:vsphere]", func() {
	f := framework.NewDefaultFramework("volume-provision")

	var (
		client           clientset.Interface
		namespace        string
		scParameters     map[string]string
		clusterDatastore string
		nodeInfo         *NodeInfo
	)

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		namespace = f.Namespace.Name
		nodeInfo = GetReadySchedulableRandomNodeInfo()
		scParameters = make(map[string]string)
		clusterDatastore = GetAndExpectStringEnvVar(VCPClusterDatastore)
	})

	/*
		Steps:
		1. Create volume options with datastore to be a clustered datastore
		2. Create a vsphere volume
		3. Create podspec with volume path. Create a corresponding pod
		4. Verify disk is attached
		5. Delete the pod and wait for the disk to be detached
		6. Delete the volume
	*/

	It("verify static provisioning on clustered datastore", func() {
		var volumePath string

		By("creating a test vsphere volume")
		volumeOptions := new(VolumeOptions)
		volumeOptions.CapacityKB = 2097152
		volumeOptions.Name = "e2e-vmdk-" + namespace
		volumeOptions.Datastore = clusterDatastore

		volumePath, err := nodeInfo.VSphere.CreateVolume(volumeOptions, nodeInfo.DataCenterRef)
		Expect(err).NotTo(HaveOccurred())

		defer func() {
			By("Deleting the vsphere volume")
			nodeInfo.VSphere.DeleteVolume(volumePath, nodeInfo.DataCenterRef)
		}()

		podspec := getVSpherePodSpecWithVolumePaths([]string{volumePath}, nil, nil)

		By("Creating pod")
		pod, err := client.CoreV1().Pods(namespace).Create(podspec)
		Expect(err).NotTo(HaveOccurred())
		By("Waiting for pod to be ready")
		Expect(framework.WaitForPodNameRunningInNamespace(client, pod.Name, namespace)).To(Succeed())

		// get fresh pod info
		pod, err = client.CoreV1().Pods(namespace).Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		nodeName := pod.Spec.NodeName

		By("Verifying volume is attached")
		expectVolumeToBeAttached(nodeName, volumePath)

		By("Deleting pod")
		err = framework.DeletePodWithWait(f, client, pod)
		Expect(err).NotTo(HaveOccurred())

		By("Waiting for volumes to be detached from the node")
		err = waitForVSphereDiskToDetach(volumePath, nodeName)
		Expect(err).NotTo(HaveOccurred())
	})

	/*
		Steps:
		1. Create storage class parameter and specify datastore to be a clustered datastore name
		2. invokeValidPolicyTest - util to do e2e dynamic provision test
	*/
	It("verify dynamic provision with default parameter on clustered datastore", func() {
		scParameters[Datastore] = clusterDatastore
		invokeValidPolicyTest(f, client, namespace, scParameters)
	})

	/*
		Steps:
		1. Create storage class parameter and specify storage policy to be a tag based spbm policy
		2. invokeValidPolicyTest - util to do e2e dynamic provision test
	*/
	It("verify dynamic provision with spbm policy on clustered datastore", func() {
		policyDatastoreCluster := GetAndExpectStringEnvVar(SPBMPolicyDataStoreCluster)
		scParameters[SpbmStoragePolicy] = policyDatastoreCluster
		invokeValidPolicyTest(f, client, namespace, scParameters)
	})
})
