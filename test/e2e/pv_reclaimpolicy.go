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

package e2e

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
	"k8s.io/kubernetes/test/e2e/framework"
	"strconv"
	"time"
)

var _ = framework.KubeDescribe("persistentvolumereclaim", func() {
	f := framework.NewDefaultFramework("persistentvolumereclaim")
	var c clientset.Interface
	var ns string

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
	})

	/*
		This test verifies persistent volume should be deleted when reclaimPolicy on the PV is set to delete and
		associated claim is deleted

		Test Steps:
		1. Create vmdk
		2. Create PV Spec with volume path set to VMDK file created in Step-1, and PersistentVolumeReclaimPolicy is set to Delete
		3. Create PVC with the storage request set to PV's storage capacity.
		4. Wait for PV and PVC to bound.
		5. Delete PVC
		6. Verify PV is deleted automatically.
	*/
	framework.KubeDescribe("persistentvolumereclaim:delete", func() {
		var (
			volumePath    string
			pv            *v1.PersistentVolume
			pvc           *v1.PersistentVolumeClaim
			volumeoptions vsphere.VolumeOptions
		)

		It("should delete persistent volume when reclaimPolicy set to delete and associated claim is deleted", func() {

			By("creating vmdk")
			vsp, err := vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())

			volumeoptions.CapacityKB = 2097152
			volumeoptions.Name = "e2e-vmdk-" + strconv.FormatInt(time.Now().UnixNano(), 10)
			volumeoptions.DiskFormat = "thin"
			volumePath, err = vsp.CreateVolume(&volumeoptions)

			Expect(err).NotTo(HaveOccurred())
			pv = getVSpherePersistentVolumeSpec(volumePath, v1.PersistentVolumeReclaimDelete, nil)
			pv, err := c.CoreV1().PersistentVolumes().Create(pv)
			Expect(err).NotTo(HaveOccurred())

			By("creating the pv")
			pvc = getVSpherePersistentVolumeClaimSpec(ns, nil)

			pvc, err := c.CoreV1().PersistentVolumeClaims(ns).Create(pvc)
			Expect(err).NotTo(HaveOccurred())

			By("wait for the pv and pvc")
			waitOnPVandPVC(c, ns, pv, pvc)

			By("delete pvc")
			deletePersistentVolumeClaim(c, pvc.Name, ns)

			By("verify pv is deleted")
			err = framework.WaitForPersistentVolumeDeleted(c, pv.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	/*
		This test Verify persistent volume should be retained when reclaimPolicy on the PV is set to retain
		and associated claim is deleted

		Test Steps:
		1. Create vmdk
		2. Create PV Spec with volume path set to VMDK file created in Step-1, and PersistentVolumeReclaimPolicy is set to Retain
		3. Create PVC with the storage request set to PV's storage capacity.
		4. Wait for PV and PVC to bound.
		5. Delete PVC
		6. Verify PV is retained.
		7. Delete PV.
	*/

	framework.KubeDescribe("persistentvolumereclaim:retain", func() {
		var (
			volumePath    string
			pv            *v1.PersistentVolume
			pvc           *v1.PersistentVolumeClaim
			volumeoptions vsphere.VolumeOptions
		)

		It("should retain persistent volume when reclaimPolicy set to retain when associated claim is deleted", func() {

			By("creating vmdk")
			vsp, err := vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())

			volumeoptions.CapacityKB = 2097152
			volumeoptions.Name = "e2e-vmdk-" + strconv.FormatInt(time.Now().UnixNano(), 10)
			volumeoptions.DiskFormat = "thin"
			volumePath, err = vsp.CreateVolume(&volumeoptions)
			Expect(err).NotTo(HaveOccurred())

			By("creating the pv")
			pv = getVSpherePersistentVolumeSpec(volumePath, v1.PersistentVolumeReclaimRetain, nil)
			pv, err := c.CoreV1().PersistentVolumes().Create(pv)
			Expect(err).NotTo(HaveOccurred())

			By("creating the pvc")
			pvc = getVSpherePersistentVolumeClaimSpec(ns, nil)

			pvc, err := c.CoreV1().PersistentVolumeClaims(ns).Create(pvc)
			Expect(err).NotTo(HaveOccurred())

			By("wait for the pv and pvc")
			waitOnPVandPVC(c, ns, pv, pvc)

			By("delete pvc")
			deletePersistentVolumeClaim(c, pvc.Name, ns)

			By("verify pv is retained")
			framework.Logf("Waiting for PV %v to become Released", pv.Name)
			err = framework.WaitForPersistentVolumePhase(v1.VolumeReleased, c, pv.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())

			By("delete pv")
			deletePersistentVolume(c, pv.Name)
			err = framework.WaitForPersistentVolumeDeleted(c, pv.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())
		})
	})
})

func getVSpherePersistentVolumeSpec(volumePath string, persistentVolumeReclaimPolicy v1.PersistentVolumeReclaimPolicy, labels map[string]string) *v1.PersistentVolume {
	var (
		pvConfig persistentVolumeConfig
		pv       *v1.PersistentVolume
		claimRef *v1.ObjectReference
	)
	pvConfig = persistentVolumeConfig{
		namePrefix: "vspherepv-",
		pvSource: v1.PersistentVolumeSource{
			VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
				VolumePath: volumePath,
				FSType:     "ext4",
			},
		},
		prebind: nil,
	}

	pv = &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: pvConfig.namePrefix,
			Annotations: map[string]string{
				volumehelper.VolumeGidAnnotationKey: "777",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: persistentVolumeReclaimPolicy,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("2Gi"),
			},
			PersistentVolumeSource: pvConfig.pvSource,
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			ClaimRef: claimRef,
		},
	}
	if labels != nil {
		pv.Labels = labels
	}
	return pv
}

func getVSpherePersistentVolumeClaimSpec(namespace string, labels map[string]string) *v1.PersistentVolumeClaim {
	var (
		pvc *v1.PersistentVolumeClaim
	)
	pvc = &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    namespace,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("2Gi"),
				},
			},
		},
	}
	if labels != nil {
		pvc.Spec.Selector = &metav1.LabelSelector{MatchLabels: labels}
	}

	return pvc
}
