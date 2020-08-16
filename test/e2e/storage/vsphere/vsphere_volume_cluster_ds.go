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
	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
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

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("vsphere")
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

	ginkgo.It("verify static provisioning on clustered datastore", func() {
		var volumePath string

		ginkgo.By("creating a test vsphere volume")
		volumeOptions := new(VolumeOptions)
		volumeOptions.CapacityKB = 2097152
		volumeOptions.Name = "e2e-vmdk-" + namespace
		volumeOptions.Datastore = clusterDatastore

		volumePath, err := nodeInfo.VSphere.CreateVolume(volumeOptions, nodeInfo.DataCenterRef)
		framework.ExpectNoError(err)

		defer func() {
			ginkgo.By("Deleting the vsphere volume")
			nodeInfo.VSphere.DeleteVolume(volumePath, nodeInfo.DataCenterRef)
		}()

		podspec := getVSpherePodSpecWithVolumePaths([]string{volumePath}, nil, nil)

		ginkgo.By("Creating pod")
		pod, err := client.CoreV1().Pods(namespace).Create(context.TODO(), podspec, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		ginkgo.By("Waiting for pod to be ready")
		gomega.Expect(e2epod.WaitForPodNameRunningInNamespace(client, pod.Name, namespace)).To(gomega.Succeed())

		// get fresh pod info
		pod, err = client.CoreV1().Pods(namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		nodeName := pod.Spec.NodeName

		ginkgo.By("Verifying volume is attached")
		expectVolumeToBeAttached(nodeName, volumePath)

		ginkgo.By("Deleting pod")
		err = e2epod.DeletePodWithWait(client, pod)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for volumes to be detached from the node")
		err = waitForVSphereDiskToDetach(volumePath, nodeName)
		framework.ExpectNoError(err)
	})

	/*
		Steps:
		1. Create storage class parameter and specify datastore to be a clustered datastore name
		2. invokeValidPolicyTest - util to do e2e dynamic provision test
	*/
	ginkgo.It("verify dynamic provision with default parameter on clustered datastore", func() {
		scParameters[Datastore] = clusterDatastore
		invokeValidPolicyTest(f, client, namespace, scParameters)
	})

	/*
		Steps:
		1. Create storage class parameter and specify storage policy to be a tag based spbm policy
		2. invokeValidPolicyTest - util to do e2e dynamic provision test
	*/
	ginkgo.It("verify dynamic provision with spbm policy on clustered datastore", func() {
		policyDatastoreCluster := GetAndExpectStringEnvVar(SPBMPolicyDataStoreCluster)
		scParameters[SpbmStoragePolicy] = policyDatastoreCluster
		invokeValidPolicyTest(f, client, namespace, scParameters)
	})
})
