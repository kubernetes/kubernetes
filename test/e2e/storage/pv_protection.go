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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/util/slice"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("PV Protection", func() {
	var (
		client    clientset.Interface
		nameSpace string
		err       error
		pvc       *v1.PersistentVolumeClaim
		pv        *v1.PersistentVolume
		pvConfig  framework.PersistentVolumeConfig
		pvcConfig framework.PersistentVolumeClaimConfig
		volLabel  labels.Set
		selector  *metav1.LabelSelector
	)

	f := framework.NewDefaultFramework("pv-protection")
	BeforeEach(func() {
		client = f.ClientSet
		nameSpace = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(client, framework.TestContext.NodeSchedulableTimeout))

		// Enforce binding only within test space via selector labels
		volLabel = labels.Set{framework.VolumeSelectorKey: nameSpace}
		selector = metav1.SetAsLabelSelector(volLabel)

		pvConfig = framework.PersistentVolumeConfig{
			NamePrefix: "hostpath-",
			Labels:     volLabel,
			PVSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: "/tmp/data",
				},
			},
		}

		emptyStorageClass := ""
		pvcConfig = framework.PersistentVolumeClaimConfig{
			Selector:         selector,
			StorageClassName: &emptyStorageClass,
		}

		By("Creating a PV")
		// make the pv definitions
		pv = framework.MakePersistentVolume(pvConfig)
		// create the PV
		pv, err = client.CoreV1().PersistentVolumes().Create(pv)
		Expect(err).NotTo(HaveOccurred(), "Error creating PV")

		By("Waiting for PV to enter phase Available")
		framework.ExpectNoError(framework.WaitForPersistentVolumePhase(v1.VolumeAvailable, client, pv.Name, 1*time.Second, 30*time.Second))

		By("Checking that PV Protection finalizer is set")
		pv, err = client.CoreV1().PersistentVolumes().Get(pv.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "While getting PV status")
		Expect(slice.ContainsString(pv.ObjectMeta.Finalizers, volumeutil.PVProtectionFinalizer, nil)).To(BeTrue())
	})

	AfterEach(func() {
		framework.Logf("AfterEach: Cleaning up test resources.")
		if errs := framework.PVPVCCleanup(client, nameSpace, pv, pvc); len(errs) > 0 {
			framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
		}
	})

	It("Verify \"immediate\" deletion of a PV that is not bound to a PVC", func() {
		By("Deleting the PV")
		err = client.CoreV1().PersistentVolumes().Delete(pv.Name, metav1.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred(), "Error deleting PV")
		framework.WaitForPersistentVolumeDeleted(client, pv.Name, framework.Poll, framework.PVDeletingTimeout)
	})

	It("Verify that PV bound to a PVC is not removed immediately", func() {
		By("Creating a PVC")
		pvc = framework.MakePersistentVolumeClaim(pvcConfig, nameSpace)
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		Expect(err).NotTo(HaveOccurred(), "Error creating PVC")

		By("Waiting for PVC to become Bound")
		err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, nameSpace, pvc.Name, framework.Poll, framework.ClaimBindingTimeout)
		Expect(err).NotTo(HaveOccurred(), "Failed waiting for PVC to be bound %v", err)

		By("Deleting the PV, however, the PV must not be removed from the system as it's bound to a PVC")
		err = client.CoreV1().PersistentVolumes().Delete(pv.Name, metav1.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred(), "Error deleting PV")

		By("Checking that the PV status is Terminating")
		pv, err = client.CoreV1().PersistentVolumes().Get(pv.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "While checking PV status")
		Expect(pv.ObjectMeta.DeletionTimestamp).NotTo(Equal(nil))

		By("Deleting the PVC that is bound to the PV")
		err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, metav1.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred(), "Error deleting PVC")

		By("Checking that the PV is automatically removed from the system because it's no longer bound to a PVC")
		framework.WaitForPersistentVolumeDeleted(client, pv.Name, framework.Poll, framework.PVDeletingTimeout)
	})
})
