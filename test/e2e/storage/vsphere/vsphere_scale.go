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
	"fmt"
	"strconv"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

/*
	Perform vsphere volume life cycle management at scale based on user configurable value for number of volumes.
	The following actions will be performed as part of this test.

	1. Create Storage Classes of 4 Categories (Default, SC with Non Default Datastore, SC with SPBM Policy, SC with VSAN Storage Capabilities.)
	2. Read VCP_SCALE_VOLUME_COUNT, VCP_SCALE_INSTANCES, VCP_SCALE_VOLUMES_PER_POD, VSPHERE_SPBM_POLICY_NAME, VSPHERE_DATASTORE from System Environment.
	3. Launch VCP_SCALE_INSTANCES goroutine for creating VCP_SCALE_VOLUME_COUNT volumes. Each goroutine is responsible for create/attach of VCP_SCALE_VOLUME_COUNT/VCP_SCALE_INSTANCES volumes.
	4. Read VCP_SCALE_VOLUMES_PER_POD from System Environment. Each pod will be have VCP_SCALE_VOLUMES_PER_POD attached to it.
	5. Once all the go routines are completed, we delete all the pods and volumes.
*/
const (
	NodeLabelKey = "vsphere_e2e_label"
)

// NodeSelector holds
type NodeSelector struct {
	labelKey   string
	labelValue string
}

var _ = utils.SIGDescribe("vcp at scale [Feature:vsphere] ", func() {
	f := framework.NewDefaultFramework("vcp-at-scale")

	var (
		client            clientset.Interface
		namespace         string
		nodeSelectorList  []*NodeSelector
		volumeCount       int
		numberOfInstances int
		volumesPerPod     int
		policyName        string
		datastoreName     string
		nodeVolumeMapChan chan map[string][]string
		nodes             *v1.NodeList
		scNames           = []string{storageclass1, storageclass2, storageclass3, storageclass4}
	)

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		namespace = f.Namespace.Name
		nodeVolumeMapChan = make(chan map[string][]string)

		// Read the environment variables
		volumeCount = GetAndExpectIntEnvVar(VCPScaleVolumeCount)
		volumesPerPod = GetAndExpectIntEnvVar(VCPScaleVolumesPerPod)

		numberOfInstances = GetAndExpectIntEnvVar(VCPScaleInstances)
		framework.ExpectNotEqual(numberOfInstances > 5, true, "Maximum allowed instances are 5")
		framework.ExpectNotEqual(numberOfInstances > volumeCount, true, "Number of instances should be less than the total volume count")

		policyName = GetAndExpectStringEnvVar(SPBMPolicyName)
		datastoreName = GetAndExpectStringEnvVar(StorageClassDatastoreName)

		var err error
		nodes, err = e2enode.GetReadySchedulableNodes(client)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf("Requires at least %d nodes (not %d)", 2, len(nodes.Items))
		}
		// Verify volume count specified by the user can be satisfied
		if volumeCount > volumesPerNode*len(nodes.Items) {
			e2eskipper.Skipf("Cannot attach %d volumes to %d nodes. Maximum volumes that can be attached on %d nodes is %d", volumeCount, len(nodes.Items), len(nodes.Items), volumesPerNode*len(nodes.Items))
		}
		nodeSelectorList = createNodeLabels(client, namespace, nodes)
	})

	/*
		Remove labels from all the nodes
	*/
	framework.AddCleanupAction(func() {
		// Cleanup actions will be called even when the tests are skipped and leaves namespace unset.
		if len(namespace) > 0 {
			for _, node := range nodes.Items {
				framework.RemoveLabelOffNode(client, node.Name, NodeLabelKey)
			}
		}
	})

	ginkgo.It("vsphere scale tests", func() {
		var pvcClaimList []string
		nodeVolumeMap := make(map[string][]string)
		// Volumes will be provisioned with each different types of Storage Class
		scArrays := make([]*storagev1.StorageClass, len(scNames))
		for index, scname := range scNames {
			// Create vSphere Storage Class
			ginkgo.By(fmt.Sprintf("Creating Storage Class : %q", scname))
			var sc *storagev1.StorageClass
			scParams := make(map[string]string)
			var err error
			switch scname {
			case storageclass1:
				scParams = nil
			case storageclass2:
				scParams[Policy_HostFailuresToTolerate] = "1"
			case storageclass3:
				scParams[SpbmStoragePolicy] = policyName
			case storageclass4:
				scParams[Datastore] = datastoreName
			}
			sc, err = client.StorageV1().StorageClasses().Create(getVSphereStorageClassSpec(scname, scParams, nil, ""))
			gomega.Expect(sc).NotTo(gomega.BeNil(), "Storage class is empty")
			framework.ExpectNoError(err, "Failed to create storage class")
			defer client.StorageV1().StorageClasses().Delete(scname, nil)
			scArrays[index] = sc
		}

		volumeCountPerInstance := volumeCount / numberOfInstances
		for instanceCount := 0; instanceCount < numberOfInstances; instanceCount++ {
			if instanceCount == numberOfInstances-1 {
				volumeCountPerInstance = volumeCount
			}
			volumeCount = volumeCount - volumeCountPerInstance
			go VolumeCreateAndAttach(client, namespace, scArrays, volumeCountPerInstance, volumesPerPod, nodeSelectorList, nodeVolumeMapChan)
		}

		// Get the list of all volumes attached to each node from the go routines by reading the data from the channel
		for instanceCount := 0; instanceCount < numberOfInstances; instanceCount++ {
			for node, volumeList := range <-nodeVolumeMapChan {
				nodeVolumeMap[node] = append(nodeVolumeMap[node], volumeList...)
			}
		}
		podList, err := client.CoreV1().Pods(namespace).List(metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list pods")
		for _, pod := range podList.Items {
			pvcClaimList = append(pvcClaimList, getClaimsForPod(&pod, volumesPerPod)...)
			ginkgo.By("Deleting pod")
			err = e2epod.DeletePodWithWait(client, &pod)
			framework.ExpectNoError(err)
		}
		ginkgo.By("Waiting for volumes to be detached from the node")
		err = waitForVSphereDisksToDetach(nodeVolumeMap)
		framework.ExpectNoError(err)

		for _, pvcClaim := range pvcClaimList {
			err = e2epv.DeletePersistentVolumeClaim(client, pvcClaim, namespace)
			framework.ExpectNoError(err)
		}
	})
})

// Get PVC claims for the pod
func getClaimsForPod(pod *v1.Pod, volumesPerPod int) []string {
	pvcClaimList := make([]string, volumesPerPod)
	for i, volumespec := range pod.Spec.Volumes {
		if volumespec.PersistentVolumeClaim != nil {
			pvcClaimList[i] = volumespec.PersistentVolumeClaim.ClaimName
		}
	}
	return pvcClaimList
}

// VolumeCreateAndAttach peforms create and attach operations of vSphere persistent volumes at scale
func VolumeCreateAndAttach(client clientset.Interface, namespace string, sc []*storagev1.StorageClass, volumeCountPerInstance int, volumesPerPod int, nodeSelectorList []*NodeSelector, nodeVolumeMapChan chan map[string][]string) {
	defer ginkgo.GinkgoRecover()
	nodeVolumeMap := make(map[string][]string)
	nodeSelectorIndex := 0
	for index := 0; index < volumeCountPerInstance; index = index + volumesPerPod {
		if (volumeCountPerInstance - index) < volumesPerPod {
			volumesPerPod = volumeCountPerInstance - index
		}
		pvclaims := make([]*v1.PersistentVolumeClaim, volumesPerPod)
		for i := 0; i < volumesPerPod; i++ {
			ginkgo.By("Creating PVC using the Storage Class")
			pvclaim, err := e2epv.CreatePVC(client, namespace, getVSphereClaimSpecWithStorageClass(namespace, "2Gi", sc[index%len(sc)]))
			framework.ExpectNoError(err)
			pvclaims[i] = pvclaim
		}

		ginkgo.By("Waiting for claim to be in bound phase")
		persistentvolumes, err := e2epv.WaitForPVClaimBoundPhase(client, pvclaims, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("Creating pod to attach PV to the node")
		nodeSelector := nodeSelectorList[nodeSelectorIndex%len(nodeSelectorList)]
		// Create pod to attach Volume to Node
		pod, err := e2epod.CreatePod(client, namespace, map[string]string{nodeSelector.labelKey: nodeSelector.labelValue}, pvclaims, false, "")
		framework.ExpectNoError(err)

		for _, pv := range persistentvolumes {
			nodeVolumeMap[pod.Spec.NodeName] = append(nodeVolumeMap[pod.Spec.NodeName], pv.Spec.VsphereVolume.VolumePath)
		}
		ginkgo.By("Verify the volume is accessible and available in the pod")
		verifyVSphereVolumesAccessible(client, pod, persistentvolumes)
		nodeSelectorIndex++
	}
	nodeVolumeMapChan <- nodeVolumeMap
	close(nodeVolumeMapChan)
}

func createNodeLabels(client clientset.Interface, namespace string, nodes *v1.NodeList) []*NodeSelector {
	var nodeSelectorList []*NodeSelector
	for i, node := range nodes.Items {
		labelVal := "vsphere_e2e_" + strconv.Itoa(i)
		nodeSelector := &NodeSelector{
			labelKey:   NodeLabelKey,
			labelValue: labelVal,
		}
		nodeSelectorList = append(nodeSelectorList, nodeSelector)
		framework.AddOrUpdateLabelOnNode(client, node.Name, NodeLabelKey, labelVal)
	}
	return nodeSelectorList
}
