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
	"context"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("PV Protection", func() {
	var (
		client    clientset.Interface
		nameSpace string
		err       error
		pvc       *v1.PersistentVolumeClaim
		pv        *v1.PersistentVolume
		pvConfig  e2epv.PersistentVolumeConfig
		pvcConfig e2epv.PersistentVolumeClaimConfig
		volLabel  labels.Set
		selector  *metav1.LabelSelector
	)

	f := framework.NewDefaultFramework("pv-protection")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.BeforeEach(func(ctx context.Context) {
		client = f.ClientSet
		nameSpace = f.Namespace.Name
		framework.ExpectNoError(e2enode.WaitForAllNodesSchedulable(ctx, client, f.Timeouts.NodeSchedulable))

		// Enforce binding only within test space via selector labels
		volLabel = labels.Set{e2epv.VolumeSelectorKey: nameSpace}
		selector = metav1.SetAsLabelSelector(volLabel)

		pvConfig = e2epv.PersistentVolumeConfig{
			NamePrefix: "hostpath-",
			Labels:     volLabel,
			PVSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: "/tmp/data",
				},
			},
		}

		emptyStorageClass := ""
		pvcConfig = e2epv.PersistentVolumeClaimConfig{
			Selector:         selector,
			StorageClassName: &emptyStorageClass,
		}

		ginkgo.By("Creating a PV")
		// make the pv definitions
		pv = e2epv.MakePersistentVolume(pvConfig)
		// create the PV
		pv, err = client.CoreV1().PersistentVolumes().Create(ctx, pv, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating PV")

		ginkgo.By("Waiting for PV to enter phase Available")
		framework.ExpectNoError(e2epv.WaitForPersistentVolumePhase(ctx, v1.VolumeAvailable, client, pv.Name, 1*time.Second, 30*time.Second))

		ginkgo.By("Checking that PV Protection finalizer is set")
		pv, err = client.CoreV1().PersistentVolumes().Get(ctx, pv.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "While getting PV status")
		gomega.Expect(pv.ObjectMeta.Finalizers).Should(gomega.ContainElement(volumeutil.PVProtectionFinalizer), "PV Protection finalizer(%v) is not set in %v", volumeutil.PVProtectionFinalizer, pv.ObjectMeta.Finalizers)
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		framework.Logf("AfterEach: Cleaning up test resources.")
		if errs := e2epv.PVPVCCleanup(ctx, client, nameSpace, pv, pvc); len(errs) > 0 {
			framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
		}
	})

	ginkgo.It("Verify \"immediate\" deletion of a PV that is not bound to a PVC", func(ctx context.Context) {
		ginkgo.By("Deleting the PV")
		err = client.CoreV1().PersistentVolumes().Delete(ctx, pv.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err, "Error deleting PV")
		err = e2epv.WaitForPersistentVolumeDeleted(ctx, client, pv.Name, framework.Poll, f.Timeouts.PVDelete)
		framework.ExpectNoError(err, "waiting for PV to be deleted")
	})

	ginkgo.It("Verify that PV bound to a PVC is not removed immediately", func(ctx context.Context) {
		ginkgo.By("Creating a PVC")
		pvc = e2epv.MakePersistentVolumeClaim(pvcConfig, nameSpace)
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(ctx, pvc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating PVC")

		ginkgo.By("Waiting for PVC to become Bound")
		err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, client, nameSpace, pvc.Name, framework.Poll, f.Timeouts.ClaimBound)
		framework.ExpectNoError(err, "Failed waiting for PVC to be bound %v", err)

		ginkgo.By("Deleting the PV, however, the PV must not be removed from the system as it's bound to a PVC")
		err = client.CoreV1().PersistentVolumes().Delete(ctx, pv.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err, "Error deleting PV")

		ginkgo.By("Checking that the PV status is Terminating")
		pv, err = client.CoreV1().PersistentVolumes().Get(ctx, pv.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "While checking PV status")
		gomega.Expect(pv.ObjectMeta.DeletionTimestamp).ToNot(gomega.BeNil())

		ginkgo.By("Deleting the PVC that is bound to the PV")
		err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(ctx, pvc.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err, "Error deleting PVC")

		ginkgo.By("Checking that the PV is automatically removed from the system because it's no longer bound to a PVC")
		err = e2epv.WaitForPersistentVolumeDeleted(ctx, client, pv.Name, framework.Poll, f.Timeouts.PVDelete)
		framework.ExpectNoError(err, "waiting for PV to be deleted")
	})
})
