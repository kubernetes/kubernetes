/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	"path"
)

func createStorageClass(ns string, c clientset.Interface) (*storage.StorageClass, error) {
	bindingMode := storage.VolumeBindingImmediate
	stKlass := getStorageClass("flex-expand", map[string]string{}, &bindingMode, ns, "resizing")
	allowExpansion := true
	stKlass.AllowVolumeExpansion = &allowExpansion

	var err error
	stKlass, err = c.StorageV1().StorageClasses().Create(stKlass)
	return stKlass, err
}

var _ = utils.SIGDescribe("Mounted flexvolume volume expand [Slow] [Feature:ExpandInUsePersistentVolumes]", func() {
	var (
		c                 clientset.Interface
		ns                string
		err               error
		pvc               *v1.PersistentVolumeClaim
		resizableSc       *storage.StorageClass
		nodeName          string
		isNodeLabeled     bool
		nodeKeyValueLabel map[string]string
		nodeLabelValue    string
		nodeKey           string
		nodeList          *v1.NodeList
	)

	f := framework.NewDefaultFramework("mounted-flexvolume-expand")
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("aws", "gce", "local")
		framework.SkipUnlessMasterOSDistroIs("debian", "ubuntu", "gci", "custom")
		framework.SkipUnlessNodeOSDistroIs("debian", "ubuntu", "gci", "custom")
		framework.SkipUnlessSSHKeyPresent()
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))

		nodeList = framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		if len(nodeList.Items) == 0 {
			framework.Failf("unable to find ready and schedulable Node")
		}
		nodeName = nodeList.Items[0].Name

		nodeKey = "mounted_flexvolume_expand"

		if !isNodeLabeled {
			nodeLabelValue = ns
			nodeKeyValueLabel = make(map[string]string)
			nodeKeyValueLabel[nodeKey] = nodeLabelValue
			framework.AddOrUpdateLabelOnNode(c, nodeName, nodeKey, nodeLabelValue)
			isNodeLabeled = true
		}

		resizableSc, err = createStorageClass(ns, c)
		if err != nil {
			fmt.Printf("storage class creation error: %v\n", err)
		}
		Expect(err).NotTo(HaveOccurred(), "Error creating resizable storage class: %v", err)
		Expect(*resizableSc.AllowVolumeExpansion).To(BeTrue())

		pvc = getClaim("2Gi", ns)
		pvc.Spec.StorageClassName = &resizableSc.Name
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		Expect(err).NotTo(HaveOccurred(), "Error creating pvc: %v", err)

	})

	framework.AddCleanupAction(func() {
		if len(nodeLabelValue) > 0 {
			framework.RemoveLabelOffNode(c, nodeName, nodeKey)
		}
	})

	AfterEach(func() {
		framework.Logf("AfterEach: Cleaning up resources for mounted volume resize")

		if c != nil {
			if errs := framework.PVPVCCleanup(c, ns, nil, pvc); len(errs) > 0 {
				framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
			}
			pvc, nodeName, isNodeLabeled, nodeLabelValue = nil, "", false, ""
			nodeKeyValueLabel = make(map[string]string)
		}
	})

	It("should be resizable when mounted", func() {
		driver := "dummy-attachable"

		node := nodeList.Items[0]
		By(fmt.Sprintf("installing flexvolume %s on node %s as %s", path.Join(driverDir, driver), node.Name, driver))
		installFlex(c, &node, "k8s", driver, path.Join(driverDir, driver))
		By(fmt.Sprintf("installing flexvolume %s on (master) node %s as %s", path.Join(driverDir, driver), node.Name, driver))
		installFlex(c, nil, "k8s", driver, path.Join(driverDir, driver))

		pv := framework.MakePersistentVolume(framework.PersistentVolumeConfig{
			PVSource: v1.PersistentVolumeSource{
				FlexVolume: &v1.FlexPersistentVolumeSource{
					Driver: "k8s/" + driver,
				}},
			NamePrefix:       "pv-",
			StorageClassName: resizableSc.Name,
			VolumeMode:       pvc.Spec.VolumeMode,
		})

		pv, err = framework.CreatePV(c, pv)
		Expect(err).NotTo(HaveOccurred(), "Error creating pv %v", err)

		By("Waiting for PVC to be in bound phase")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		var pvs []*v1.PersistentVolume

		pvs, err = framework.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred(), "Failed waiting for PVC to be bound %v", err)
		Expect(len(pvs)).To(Equal(1))

		var pod *v1.Pod
		By("Creating pod")
		pod, err = framework.CreateNginxPod(c, ns, nodeKeyValueLabel, pvcClaims)
		Expect(err).NotTo(HaveOccurred(), "Failed to create pod %v", err)
		defer framework.DeletePodWithWait(f, c, pod)

		By("Waiting for pod to go to 'running' state")
		err = f.WaitForPodRunning(pod.ObjectMeta.Name)
		Expect(err).NotTo(HaveOccurred(), "Pod didn't go to 'running' state %v", err)

		By("Expanding current pvc")
		newSize := resource.MustParse("6Gi")
		pvc, err = expandPVCSize(pvc, newSize, c)
		Expect(err).NotTo(HaveOccurred(), "While updating pvc for more size")
		Expect(pvc).NotTo(BeNil())

		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcSize.Cmp(newSize) != 0 {
			framework.Failf("error updating pvc size %q", pvc.Name)
		}

		By("Waiting for cloudprovider resize to finish")
		err = waitForControllerVolumeResize(pvc, c)
		Expect(err).NotTo(HaveOccurred(), "While waiting for pvc resize to finish")

		By("Waiting for file system resize to finish")
		pvc, err = waitForFSResize(pvc, c)
		Expect(err).NotTo(HaveOccurred(), "while waiting for fs resize to finish")

		pvcConditions := pvc.Status.Conditions
		Expect(len(pvcConditions)).To(Equal(0), "pvc should not have conditions")
	})
})
