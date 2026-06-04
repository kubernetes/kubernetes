/*
Copyright 2019 The Kubernetes Authors.

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

package node

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/gomega"
)

func getPatchBytes(oldLease, newLease *coordinationv1.Lease) ([]byte, error) {
	oldData, err := json.Marshal(oldLease)
	if err != nil {
		return nil, fmt.Errorf("failed to Marshal oldData: %w", err)
	}
	newData, err := json.Marshal(newLease)
	if err != nil {
		return nil, fmt.Errorf("failed to Marshal newData: %w", err)
	}
	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, coordinationv1.Lease{})
	if err != nil {
		return nil, fmt.Errorf("failed to CreateTwoWayMergePatch: %w", err)
	}
	return patchBytes, nil
}

var _ = SIGDescribe("Lease", func() {
	f := framework.NewDefaultFramework("lease-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
		Release: v1.17
		Testname: lease API should be available
		Description: Create Lease object, and get it; create and get MUST be successful and Spec of the
		read Lease MUST match Spec of original Lease. Update the Lease and get it; update and get MUST
		be successful	and Spec of the read Lease MUST match Spec of updated Lease. Patch the Lease and
		get it; patch and get MUST be successful and Spec of the read Lease MUST match Spec of patched
		Lease. Create a second Lease with labels and list Leases; create and list MUST be successful and
		list MUST return both leases. Delete the labels lease via delete collection; the delete MUST be
		successful and MUST delete only the labels lease. List leases; list MUST be successful and MUST
		return just the remaining lease. Delete the lease; delete MUST be successful. Get the lease; get
		MUST return not found error.
	*/
	framework.ConformanceIt("lease API should be available", func(ctx context.Context) {
		leaseClient := f.ClientSet.CoordinationV1().Leases(f.Namespace.Name)

		name := "lease"
		lease := &coordinationv1.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: coordinationv1.LeaseSpec{
				HolderIdentity:       ptr.To("holder"),
				LeaseDurationSeconds: ptr.To[int32](30),
				AcquireTime:          &metav1.MicroTime{Time: time.Time{}.Add(2 * time.Second)},
				RenewTime:            &metav1.MicroTime{Time: time.Time{}.Add(5 * time.Second)},
				LeaseTransitions:     ptr.To[int32](0),
			},
		}

		createdLease, err := leaseClient.Create(ctx, lease, metav1.CreateOptions{})
		framework.ExpectNoError(err, "creating Lease failed")
		gomega.Expect(createdLease).To(apimachineryutils.HaveValidResourceVersion())

		readLease, err := leaseClient.Get(ctx, name, metav1.GetOptions{})
		framework.ExpectNoError(err, "couldn't read Lease")
		if !apiequality.Semantic.DeepEqual(lease.Spec, readLease.Spec) {
			framework.Failf("Leases don't match. Diff (- for expected, + for actual):\n%s", cmp.Diff(lease.Spec, readLease.Spec))
		}

		createdLease.Spec = coordinationv1.LeaseSpec{
			HolderIdentity:       ptr.To("holder2"),
			LeaseDurationSeconds: ptr.To[int32](30),
			AcquireTime:          &metav1.MicroTime{Time: time.Time{}.Add(20 * time.Second)},
			RenewTime:            &metav1.MicroTime{Time: time.Time{}.Add(50 * time.Second)},
			LeaseTransitions:     ptr.To[int32](1),
		}

		_, err = leaseClient.Update(ctx, createdLease, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "updating Lease failed")

		readLease, err = leaseClient.Get(ctx, name, metav1.GetOptions{})
		framework.ExpectNoError(err, "couldn't read Lease")
		if !apiequality.Semantic.DeepEqual(createdLease.Spec, readLease.Spec) {
			framework.Failf("Leases don't match. Diff (- for expected, + for actual):\n%s", cmp.Diff(createdLease.Spec, readLease.Spec))
		}

		patchedLease := readLease.DeepCopy()
		patchedLease.Spec = coordinationv1.LeaseSpec{
			HolderIdentity:       ptr.To("holder3"),
			LeaseDurationSeconds: ptr.To[int32](60),
			AcquireTime:          &metav1.MicroTime{Time: time.Time{}.Add(50 * time.Second)},
			RenewTime:            &metav1.MicroTime{Time: time.Time{}.Add(70 * time.Second)},
			LeaseTransitions:     ptr.To[int32](2),
		}
		patchBytes, err := getPatchBytes(readLease, patchedLease)
		framework.ExpectNoError(err, "creating patch failed")

		_, err = leaseClient.Patch(ctx, name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
		framework.ExpectNoError(err, "patching Lease failed")

		readLease, err = leaseClient.Get(ctx, name, metav1.GetOptions{})
		framework.ExpectNoError(err, "couldn't read Lease")
		if !apiequality.Semantic.DeepEqual(patchedLease.Spec, readLease.Spec) {
			framework.Failf("Leases don't match. Diff (- for expected, + for actual):\n%s", cmp.Diff(patchedLease.Spec, readLease.Spec))
		}
		gomega.Expect(resourceversion.CompareResourceVersion(createdLease.ResourceVersion, readLease.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

		name2 := "lease2"
		lease2 := &coordinationv1.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name:   name2,
				Labels: map[string]string{"deletecollection": "true"},
			},
			Spec: coordinationv1.LeaseSpec{
				HolderIdentity:       ptr.To("holder"),
				LeaseDurationSeconds: ptr.To[int32](30),
				AcquireTime:          &metav1.MicroTime{Time: time.Time{}.Add(2 * time.Second)},
				RenewTime:            &metav1.MicroTime{Time: time.Time{}.Add(5 * time.Second)},
				LeaseTransitions:     ptr.To[int32](0),
			},
		}
		_, err = leaseClient.Create(ctx, lease2, metav1.CreateOptions{})
		framework.ExpectNoError(err, "creating Lease failed")

		leases, err := leaseClient.List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "couldn't list Leases")
		gomega.Expect(leases.Items).To(gomega.HaveLen(2))

		selector := labels.Set(map[string]string{"deletecollection": "true"}).AsSelector()
		err = leaseClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: selector.String()})
		framework.ExpectNoError(err, "couldn't delete collection")

		leases, err = leaseClient.List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "couldn't list Leases")
		gomega.Expect(leases.Items).To(gomega.HaveLen(1))

		err = leaseClient.Delete(ctx, name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "deleting Lease failed")

		_, err = leaseClient.Get(ctx, name, metav1.GetOptions{})
		if !apierrors.IsNotFound(err) {
			framework.Failf("expected IsNotFound error, got %#v", err)
		}

		leaseClient = f.ClientSet.CoordinationV1().Leases(metav1.NamespaceAll)
		// Number of leases may be high in large clusters, as Lease object is
		// created for every node by the corresponding Kubelet.
		// That said, the objects themselves are small (~300B), so even with 5000
		// of them, that gives ~1.5MB, which is acceptable.
		_, err = leaseClient.List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "couldn't list Leases from all namespace")
	})
})
