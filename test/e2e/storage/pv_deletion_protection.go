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
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// pvDeletionInTreeProtectionFinalizer is the finalizer added to protect PV deletion for in-tree volumes.
	pvDeletionInTreeProtectionFinalizer = "kubernetes.io/pv-controller"
)

var _ = utils.SIGDescribe("PV Deletion Protection", func() {
	var (
		client    clientset.Interface
		nameSpace string
		err       error
		pvc       *v1.PersistentVolumeClaim
		pv        *v1.PersistentVolume
		pvcConfig e2epv.PersistentVolumeClaimConfig
		volLabel  labels.Set
		selector  *metav1.LabelSelector
	)

	f := framework.NewDefaultFramework("pv-deletion-protection")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.BeforeEach(func(ctx context.Context) {
		client = f.ClientSet
		nameSpace = f.Namespace.Name
		framework.ExpectNoError(e2enode.WaitForAllNodesSchedulable(ctx, client, f.Timeouts.NodeSchedulable))

		// Enforce binding only within test space via selector labels
		volLabel = labels.Set{e2epv.VolumeSelectorKey: nameSpace}
		selector = metav1.SetAsLabelSelector(volLabel)

		emptyStorageClass := ""
		pvcConfig = e2epv.PersistentVolumeClaimConfig{
			Selector:         selector,
			StorageClassName: &emptyStorageClass,
		}

	})

	ginkgo.AfterEach(func(ctx context.Context) {
		framework.Logf("AfterEach: Cleaning up test resources.")
		if errs := e2epv.PVPVCCleanup(ctx, client, nameSpace, pv, pvc); len(errs) > 0 {
			framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
		}
	})

	ginkgo.It("HonorPVReclaimPolicy delete pv prior", func(ctx context.Context) {
		ginkgo.By("Creating a PVC")
		pvc = e2epv.MakePersistentVolumeClaim(pvcConfig, nameSpace)
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(ctx, pvc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating PVC")

		ginkgo.By("Waiting for PVC to become Bound")
		err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, client, nameSpace, pvc.Name, framework.Poll, f.Timeouts.ClaimBound)
		framework.ExpectNoError(err, "Failed waiting for PVC to be bound %v", err)

		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(ctx, pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Error retrieving PVC")

		pvName := pvc.Spec.VolumeName
		ginkgo.By(fmt.Sprintf("Wait for finalizer %s to be added to pv %s", pvDeletionInTreeProtectionFinalizer, pv.Name))
		err = e2epv.WaitForPVFinalizer(ctx, client, pvName, pvDeletionInTreeProtectionFinalizer, 1*time.Millisecond, 1*time.Minute)
		framework.ExpectNoError(err)

		pv, err = client.CoreV1().PersistentVolumes().Get(ctx, pvName, metav1.GetOptions{})
		framework.ExpectNoError(err, "Error retrieving PV")

		ginkgo.By("Delete pv")
		err = e2epv.DeletePersistentVolume(ctx, client, pvName)
		framework.ExpectNoError(err)

		ginkgo.By("Delete pvc")
		err = e2epv.DeletePersistentVolumeClaim(ctx, client, pvc.Name, pvc.Namespace)
		framework.ExpectNoError(err)

		ginkgo.By("Wating for the pvc to be deleted")
		framework.ExpectNoError(waitForPersistentVolumeClaimDeleted(ctx, client, pvc.Namespace, pvc.Name, 2*time.Second, 60*time.Second),
			"Failed to delete PVC", pvc.Name)

		ginkgo.By("check the pv whether it's exists")
		exists, err := e2epv.CheckPVExists(ctx, client, pvName, 2*time.Second, 60*time.Second)
		framework.ExpectNoError(err)
		if !exists {
			framework.ExpectNoError(fmt.Errorf("pv was deleted"))
		}

		ginkgo.By("Remove the finalizer on pv")
		err = e2epv.RemovePVFinalizer(ctx, client, pvName, pvDeletionInTreeProtectionFinalizer, 1*time.Millisecond, 1*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Wating for the pv to be deleted")
		framework.ExpectNoError(e2epv.WaitForPersistentVolumeDeleted(ctx, client, pvName, 2*time.Second, 60*time.Second),
			"Failed to delete PV ", pv.Name)
	})
})
