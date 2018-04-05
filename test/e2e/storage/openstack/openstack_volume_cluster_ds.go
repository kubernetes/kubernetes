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

package openstack

import (
	"fmt"
	"os"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	utils "k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("Volume Provisioning On Clustered Datastore [Feature:openstack]", func() {
	f := framework.NewDefaultFramework("volume-provision")

	var client clientset.Interface
	var namespace string
	var scParameters map[string]string
	var clusterDatastore string
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("openstack")
		client = f.ClientSet
		namespace = f.Namespace.Name
		scParameters = make(map[string]string)

		clusterDatastore = os.Getenv("CLUSTER_DATASTORE")
		Expect(clusterDatastore).NotTo(BeEmpty(), "Please set CLUSTER_DATASTORE system environment. eg: export CLUSTER_DATASTORE=<cluster_name>/<datastore_name")
	})

	It("verify static provisioning on clustered datastore", func() {
		var volumeID string
		osp, id, err := getOpenstack(client)
		Expect(err).NotTo(HaveOccurred())

		volumeID, err = createOpenstackVolume(osp)
		Expect(err).NotTo(HaveOccurred())

		defer func() {
			By("Deleting the openstack volume")
			err = osp.DeleteVolume(volumeID)
		}()

		podspec := getOpenstackPodSpecWithVolumeIDs([]string{volumeID}, nil, nil)

		By("Creating pod")
		pod, err := client.CoreV1().Pods(namespace).Create(podspec)
		Expect(err).NotTo(HaveOccurred())
		By("Waiting for pod to be ready")
		Expect(framework.WaitForPodNameRunningInNamespace(client, pod.Name, namespace)).To(Succeed())

		// get fresh pod info
		pod, err = client.CoreV1().Pods(namespace).Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		nodeName := types.NodeName(pod.Spec.NodeName)

		By("Verifying volume is attached")
		isAttached, err := verifyOpenstackDiskAttached(client, osp, id, volumeID, nodeName)
		Expect(err).NotTo(HaveOccurred())
		Expect(isAttached).To(BeTrue(), fmt.Sprintf("disk: %s is not attached with the node: %v", volumeID, nodeName))

		By("Deleting pod")
		err = framework.DeletePodWithWait(f, client, pod)
		Expect(err).NotTo(HaveOccurred())

		By("Waiting for volumes to be detached from the node")
		err = waitForOpenstackDiskToDetach(client, id, osp, volumeID, nodeName)
		Expect(err).NotTo(HaveOccurred())
	})

	It("verify dynamic provision with default parameter on clustered datastore", func() {
		scParameters[Datastore] = clusterDatastore
		invokeValidPolicyTest(f, client, namespace, scParameters)
	})

	It("verify dynamic provision with spbm policy on clustered datastore", func() {
		storagePolicy := os.Getenv("OPENSTACK_SPBM_POLICY_DS_CLUSTER")
		Expect(storagePolicy).NotTo(BeEmpty(), "Please set OPENSTACK_SPBM_POLICY_DS_CLUSTER system environment")
		scParameters[SpbmStoragePolicy] = storagePolicy
		invokeValidPolicyTest(f, client, namespace, scParameters)
	})
})

func invokeValidPolicyTest(f *framework.Framework, client clientset.Interface, namespace string, scParameters map[string]string) {
	By("Creating Storage Class With storage policy params")
	storageclass, err := client.StorageV1().StorageClasses().Create(getOpenstackStorageClassSpec("storagepolicysc", scParameters))
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create storage class with err: %v", err))
	defer client.StorageV1().StorageClasses().Delete(storageclass.Name, nil)

	By("Creating PVC using the Storage Class")
	pvclaim, err := framework.CreatePVC(client, namespace, getOpenstackClaimSpecWithStorageClassAnnotation(namespace, "2Gi", storageclass))
	Expect(err).NotTo(HaveOccurred())
	defer framework.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)

	var pvclaims []*v1.PersistentVolumeClaim
	pvclaims = append(pvclaims, pvclaim)
	By("Waiting for claim to be in bound phase")
	persistentvolumes, err := framework.WaitForPVClaimBoundPhase(client, pvclaims, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred())

	By("Creating pod to attach PV to the node")
	// Create pod to attach Volume to Node
	pod, err := framework.CreatePod(client, namespace, nil, pvclaims, false, "")
	Expect(err).NotTo(HaveOccurred())

	osp, id, err := getOpenstack(client)
	Expect(err).NotTo(HaveOccurred())
	By("Verify the volume is accessible and available in the pod")
	verifyOpenstackVolumesAccessible(client, pod, persistentvolumes, id, osp)

	By("Deleting pod")
	framework.DeletePodWithWait(f, client, pod)

	By("Waiting for volumes to be detached from the node")
	waitForOpenstackDiskToDetach(client, id, osp, persistentvolumes[0].Spec.Cinder.VolumeID, types.NodeName(pod.Spec.NodeName))
}
