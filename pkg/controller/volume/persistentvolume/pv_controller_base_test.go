/*
Copyright 2026 The Kubernetes Authors.

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

package persistentvolume

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes/fake"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	storagehelpers "k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2/ktesting"
)

func TestResyncEnqueuesOnlyObjectsNeedingReconciliation(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	claimHealthy := newClaim("claim-healthy", "uid-healthy", "1Gi", "pv-healthy", v1.ClaimBound, nil, storagehelpers.AnnBindCompleted)
	pvHealthy := newVolume("pv-healthy", "1Gi", "uid-healthy", "claim-healthy", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty)

	claimUnbound := newClaim("claim-unbound", "uid-unbound", "1Gi", "", v1.ClaimPending, nil)

	claimPrebound := newClaim("claim-prebound", "uid-prebound", "1Gi", "pv-prebound", v1.ClaimPending, nil)
	pvPrebound := newVolume("pv-prebound", "1Gi", "uid-prebound", "claim-prebound", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)

	claimMissingPV := newClaim("claim-missing-pv", "uid-missing-pv", "1Gi", "pv-missing", v1.ClaimBound, nil, storagehelpers.AnnBindCompleted)

	pvMissingClaim := newVolume("pv-missing-claim", "1Gi", "uid-missing-claim", "claim-missing-claim", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty)

	claimMisbound := newClaim("claim-misbound", "uid-misbound", "1Gi", "pv-misbound", v1.ClaimBound, nil, storagehelpers.AnnBindCompleted)
	claimOther := newClaim("claim-other", "uid-other", "1Gi", "pv-other", v1.ClaimBound, nil, storagehelpers.AnnBindCompleted)
	pvOther := newVolume("pv-other", "1Gi", "uid-other", "claim-other", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty)
	pvMisbound := newVolume("pv-misbound", "1Gi", "uid-other", "claim-other", v1.VolumeBound, v1.PersistentVolumeReclaimRetain, classEmpty)

	pvAvailable := newVolume("pv-available", "1Gi", "", "", v1.VolumeAvailable, v1.PersistentVolumeReclaimRetain, classEmpty)
	pvReleased := newVolume("pv-released", "1Gi", "", "", v1.VolumeReleased, v1.PersistentVolumeReclaimRetain, classEmpty)

	ctrl := newResyncTestController(
		t,
		ctx,
		[]*v1.PersistentVolumeClaim{
			claimHealthy,
			claimUnbound,
			claimPrebound,
			claimMissingPV,
			claimMisbound,
			claimOther,
		},
		[]*v1.PersistentVolume{
			pvHealthy,
			pvPrebound,
			pvMissingClaim,
			pvMisbound,
			pvOther,
			pvAvailable,
			pvReleased,
		},
	)

	ctrl.resync(ctx)

	gotClaimKeys := drainQueueKeys(ctrl.claimQueue)
	wantClaimKeys := sets.New[string](
		claimToClaimKey(claimUnbound),
		claimToClaimKey(claimPrebound),
		claimToClaimKey(claimMissingPV),
		claimToClaimKey(claimMisbound),
	)
	if !gotClaimKeys.Equal(wantClaimKeys) {
		t.Fatalf("unexpected claims enqueued by resync: got %v, want %v", sets.List(gotClaimKeys), sets.List(wantClaimKeys))
	}

	gotVolumeKeys := drainQueueKeys(ctrl.volumeQueue)
	wantVolumeKeys := sets.New[string](
		pvPrebound.Name,
		pvMissingClaim.Name,
		pvMisbound.Name,
		pvReleased.Name,
	)
	if !gotVolumeKeys.Equal(wantVolumeKeys) {
		t.Fatalf("unexpected volumes enqueued by resync: got %v, want %v", sets.List(gotVolumeKeys), sets.List(wantVolumeKeys))
	}
}

func newResyncTestController(t *testing.T, ctx context.Context, claims []*v1.PersistentVolumeClaim, volumes []*v1.PersistentVolume) *PersistentVolumeController {
	t.Helper()

	ctrl, err := newTestController(ctx, fake.NewSimpleClientset(), nil, true)
	if err != nil {
		t.Fatalf("failed to construct test controller: %v", err)
	}

	claimIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	for _, claim := range claims {
		if err := claimIndexer.Add(claim.DeepCopy()); err != nil {
			t.Fatalf("failed to add claim %q to indexer: %v", claim.Name, err)
		}
	}
	volumeIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	for _, volume := range volumes {
		if err := volumeIndexer.Add(volume.DeepCopy()); err != nil {
			t.Fatalf("failed to add volume %q to indexer: %v", volume.Name, err)
		}
	}

	ctrl.claimLister = corelisters.NewPersistentVolumeClaimLister(claimIndexer)
	ctrl.volumeLister = corelisters.NewPersistentVolumeLister(volumeIndexer)

	return ctrl
}

func drainQueueKeys(queue workqueue.TypedRateLimitingInterface[string]) sets.Set[string] {
	keys := sets.New[string]()
	for queue.Len() > 0 {
		key, shutdown := queue.Get()
		if shutdown {
			break
		}
		keys.Insert(key)
		queue.Done(key)
		queue.Forget(key)
	}
	return keys
}
