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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	utils "k8s.io/kubernetes/test/e2e/storage/utils"
)

// Testing configurations of single a PV/PVC pair attached to a openstack Disk
var _ = utils.SIGDescribe("PersistentVolumes:openstack", func() {
	var (
		c         clientset.Interface
		ns        string
		volumeID  string
		pv        *v1.PersistentVolume
		pvc       *v1.PersistentVolumeClaim
		clientPod *v1.Pod
		pvConfig  framework.PersistentVolumeConfig
		pvcConfig framework.PersistentVolumeClaimConfig
		err       error
		node      types.NodeName
		volLabel  labels.Set
		selector  *metav1.LabelSelector
	)

	f := framework.NewDefaultFramework("pv")
	os, _, err := getOpenstack(c)
	volumeID, err = createOpenstackVolume(os)
	Expect(err).NotTo(HaveOccurred())

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("openstack")
		c = f.ClientSet
		ns = f.Namespace.Name
		clientPod = nil
		pvc = nil
		pv = nil

		volLabel = labels.Set{framework.VolumeSelectorKey: ns}
		selector = metav1.SetAsLabelSelector(volLabel)

		pvConfig = framework.PersistentVolumeConfig{
			NamePrefix: "openstack-",
			Labels:     volLabel,
			PVSource: v1.PersistentVolumeSource{
				Cinder: &v1.CinderVolumeSource{
					VolumeID: volumeID,
					FSType:   "ext3",
					ReadOnly: false,
				},
			},
			Prebind: nil,
		}
		pvcConfig = framework.PersistentVolumeClaimConfig{
			Annotations: map[string]string{
				v1.BetaStorageClassAnnotation: "",
			},
			Selector: selector,
		}

		By("Creating the PV and PVC")
		pv, pvc, err = framework.CreatePVPVC(c, pvConfig, pvcConfig, ns, false)
		Expect(err).NotTo(HaveOccurred())
		framework.ExpectNoError(framework.WaitOnPVandPVC(c, ns, pv, pvc))

		By("Creating the Client Pod")
		clientPod, err = framework.CreateClientPod(c, ns, pvc)
		Expect(err).NotTo(HaveOccurred())
		node = types.NodeName(clientPod.Spec.NodeName)

	})

	AfterEach(func() {
		framework.Logf("AfterEach: Cleaning up test resources")
		if c != nil {
			framework.ExpectNoError(framework.DeletePodWithWait(f, c, clientPod), "AfterEach: failed to delete pod ", clientPod.Name)

			if pv != nil {
				framework.ExpectNoError(framework.DeletePersistentVolume(c, pv.Name), "AfterEach: failed to delete PV ", pv.Name)
			}
			if pvc != nil {
				framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvc.Name, ns), "AfterEach: failed to delete PVC ", pvc.Name)
			}
		}
	})

	framework.AddCleanupAction(func() {
		if len(ns) > 0 {
			_, err = framework.LoadClientset()
			if err != nil {
				return
			}
			os.DeleteVolume(volumeID)
		}
	})

	It("should test that deleting a PVC before the pod does not cause pod deletion to fail on openstack volume detach", func() {
		By("Deleting the Claim")
		framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
		pvc = nil

		By("Deleting the Pod")
		framework.ExpectNoError(framework.DeletePodWithWait(f, c, clientPod), "Failed to delete pod ", clientPod.Name)
	})

	It("should test that deleting the PV before the pod does not cause pod deletion to fail on openstack volume detach", func() {
		By("Deleting the Persistent Volume")
		framework.ExpectNoError(framework.DeletePersistentVolume(c, pv.Name), "Failed to delete PV ", pv.Name)
		pv = nil

		By("Deleting the pod")
		framework.ExpectNoError(framework.DeletePodWithWait(f, c, clientPod), "Failed to delete pod ", clientPod.Name)
	})

	It("should test that a file written to the openstack volume mount before kubelet restart can be read after restart [Disruptive]", func() {
		utils.TestKubeletRestartsAndRestoresMount(c, f, clientPod, pvc, pv)
	})

	It("should test that a openstack volume mounted to a pod that is deleted while the kubelet is down unmounts when the kubelet returns [Disruptive]", func() {
		utils.TestVolumeUnmountsFromDeletedPod(c, f, clientPod, pvc, pv)
	})

	It("should test that deleting the Namespace of a PVC and Pod causes the successful detach of openstack volume", func() {
		By("Deleting the Namespace")
		err := c.CoreV1().Namespaces().Delete(ns, nil)
		Expect(err).NotTo(HaveOccurred())

		err = framework.WaitForNamespacesDeleted(c, []string{ns}, 3*time.Minute)
		Expect(err).NotTo(HaveOccurred())

		By("Verifying Persistent Disk detaches")
		getVol, err := os.GetVolume(volumeID)
		Expect(err).NotTo(HaveOccurred())
		Expect(getVol.Status).To(Equal(VolumeAvailableStatus))

	})
})
