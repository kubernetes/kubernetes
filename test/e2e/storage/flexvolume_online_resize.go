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
	"path"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("Mounted flexvolume volume expand [Slow] [Feature:ExpandInUsePersistentVolumes]", func() {
	var (
		c                 clientset.Interface
		ns                string
		err               error
		pvc               *v1.PersistentVolumeClaim
		resizableSc       *storagev1.StorageClass
		nodeName          string
		isNodeLabeled     bool
		nodeKeyValueLabel map[string]string
		nodeLabelValue    string
		nodeKey           string
		nodeList          *v1.NodeList
	)

	f := framework.NewDefaultFramework("mounted-flexvolume-expand")
	ginkgo.BeforeEach(func() {
		framework.SkipUnlessProviderIs("aws", "gce", "local")
		framework.SkipUnlessMasterOSDistroIs("debian", "ubuntu", "gci", "custom")
		framework.SkipUnlessNodeOSDistroIs("debian", "ubuntu", "gci", "custom")
		framework.SkipUnlessSSHKeyPresent()
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))

		nodeList = framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		if len(nodeList.Items) == 0 {
			e2elog.Failf("unable to find ready and schedulable Node")
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

		test := testsuites.StorageClassTest{
			Name:                 "flexvolume-resize",
			ClaimSize:            "2Gi",
			AllowVolumeExpansion: true,
			Provisioner:          "flex-expand",
		}

		resizableSc, err = c.StorageV1().StorageClasses().Create(newStorageClass(test, ns, "resizing"))
		if err != nil {
			fmt.Printf("storage class creation error: %v\n", err)
		}
		framework.ExpectNoError(err, "Error creating resizable storage class: %v", err)
		gomega.Expect(*resizableSc.AllowVolumeExpansion).To(gomega.BeTrue())

		pvc = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
			StorageClassName: &(resizableSc.Name),
			ClaimSize:        "2Gi",
		}, ns)
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		framework.ExpectNoError(err, "Error creating pvc: %v", err)

	})

	framework.AddCleanupAction(func() {
		if len(nodeLabelValue) > 0 {
			framework.RemoveLabelOffNode(c, nodeName, nodeKey)
		}
	})

	ginkgo.AfterEach(func() {
		e2elog.Logf("AfterEach: Cleaning up resources for mounted volume resize")

		if c != nil {
			if errs := framework.PVPVCCleanup(c, ns, nil, pvc); len(errs) > 0 {
				e2elog.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
			}
			pvc, nodeName, isNodeLabeled, nodeLabelValue = nil, "", false, ""
			nodeKeyValueLabel = make(map[string]string)
		}
	})

	ginkgo.It("should be resizable when mounted", func() {
		framework.SkipUnlessSSHKeyPresent()

		driver := "dummy-attachable"

		node := nodeList.Items[0]
		ginkgo.By(fmt.Sprintf("installing flexvolume %s on node %s as %s", path.Join(driverDir, driver), node.Name, driver))
		installFlex(c, &node, "k8s", driver, path.Join(driverDir, driver))
		ginkgo.By(fmt.Sprintf("installing flexvolume %s on (master) node %s as %s", path.Join(driverDir, driver), node.Name, driver))
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
		framework.ExpectNoError(err, "Error creating pv %v", err)

		ginkgo.By("Waiting for PVC to be in bound phase")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		var pvs []*v1.PersistentVolume

		pvs, err = framework.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err, "Failed waiting for PVC to be bound %v", err)
		framework.ExpectEqual(len(pvs), 1)

		var pod *v1.Pod
		ginkgo.By("Creating pod")
		pod, err = framework.CreateNginxPod(c, ns, nodeKeyValueLabel, pvcClaims)
		framework.ExpectNoError(err, "Failed to create pod %v", err)
		defer framework.DeletePodWithWait(f, c, pod)

		ginkgo.By("Waiting for pod to go to 'running' state")
		err = f.WaitForPodRunning(pod.ObjectMeta.Name)
		framework.ExpectNoError(err, "Pod didn't go to 'running' state %v", err)

		ginkgo.By("Expanding current pvc")
		newSize := resource.MustParse("6Gi")
		pvc, err = testsuites.ExpandPVCSize(pvc, newSize, c)
		framework.ExpectNoError(err, "While updating pvc for more size")
		gomega.Expect(pvc).NotTo(gomega.BeNil())

		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcSize.Cmp(newSize) != 0 {
			e2elog.Failf("error updating pvc size %q", pvc.Name)
		}

		ginkgo.By("Waiting for cloudprovider resize to finish")
		err = testsuites.WaitForControllerVolumeResize(pvc, c, totalResizeWaitPeriod)
		framework.ExpectNoError(err, "While waiting for pvc resize to finish")

		ginkgo.By("Waiting for file system resize to finish")
		pvc, err = testsuites.WaitForFSResize(pvc, c)
		framework.ExpectNoError(err, "while waiting for fs resize to finish")

		pvcConditions := pvc.Status.Conditions
		framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")
	})
})
