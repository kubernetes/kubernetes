/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"testing"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/conversion"
)

// Test the real controller methods (add/update/delete claim/volume) with
// a fake API server.
// There is no controller API to 'initiate syncAll now', therefore these tests
// can't reliably simulate periodic sync of volumes/claims - it would be
// either very timing-sensitive or slow to wait for real periodic sync.
func TestControllerSync(t *testing.T) {
	expectedChanges := []int{1, 4, 1, 1}
	tests := []controllerTest{
		// [Unit test set 5] - controller tests.
		// We test the controller as if
		// it was connected to real API server, i.e. we call add/update/delete
		// Claim/Volume methods. Also, all changes to volumes and claims are
		// sent to add/update/delete Claim/Volume as real controller would do.
		{
			// addVolume gets a new volume. Check it's marked as Available and
			// that it's not bound to any claim - we bind volumes on periodic
			// syncClaim, not on addVolume.
			"5-1 - addVolume",
			novolumes, /* added in testCall below */
			newVolumeArray("volume5-1", "10Gi", "", "", api.VolumeAvailable, api.PersistentVolumeReclaimRetain),
			newClaimArray("claim5-1", "uid5-1", "1Gi", "", api.ClaimPending),
			newClaimArray("claim5-1", "uid5-1", "1Gi", "", api.ClaimPending),
			noevents, noerrors,
			// Custom test function that generates an add event
			func(ctrl *PersistentVolumeController, reactor *volumeReactor, test controllerTest) error {
				volume := newVolume("volume5-1", "10Gi", "", "", api.VolumePending, api.PersistentVolumeReclaimRetain)
				reactor.volumes[volume.Name] = volume
				reactor.volumeSource.Add(volume)
				return nil
			},
		},
		{
			// addClaim gets a new claim. Check it's bound to a volume.
			"5-2 - complete bind",
			newVolumeArray("volume5-2", "10Gi", "", "", api.VolumeAvailable, api.PersistentVolumeReclaimRetain),
			newVolumeArray("volume5-2", "10Gi", "uid5-2", "claim5-2", api.VolumeBound, api.PersistentVolumeReclaimRetain, annBoundByController),
			noclaims, /* added in testAddClaim5_2 */
			newClaimArray("claim5-2", "uid5-2", "1Gi", "volume5-2", api.ClaimBound, annBoundByController, annBindCompleted),
			noevents, noerrors,
			// Custom test function that generates an add event
			func(ctrl *PersistentVolumeController, reactor *volumeReactor, test controllerTest) error {
				claim := newClaim("claim5-2", "uid5-2", "1Gi", "", api.ClaimPending)
				reactor.claims[claim.Name] = claim
				reactor.claimSource.Add(claim)
				return nil
			},
		},
		{
			// deleteClaim with a bound claim makes bound volume released.
			"5-3 - delete claim",
			newVolumeArray("volume5-3", "10Gi", "uid5-3", "claim5-3", api.VolumeBound, api.PersistentVolumeReclaimRetain, annBoundByController),
			newVolumeArray("volume5-3", "10Gi", "uid5-3", "claim5-3", api.VolumeReleased, api.PersistentVolumeReclaimRetain, annBoundByController),
			newClaimArray("claim5-3", "uid5-3", "1Gi", "volume5-3", api.ClaimBound, annBoundByController, annBindCompleted),
			noclaims,
			noevents, noerrors,
			// Custom test function that generates a delete event
			func(ctrl *PersistentVolumeController, reactor *volumeReactor, test controllerTest) error {
				obj := ctrl.claims.List()[0]
				claim := obj.(*api.PersistentVolumeClaim)
				// Remove the claim from list of resulting claims.
				delete(reactor.claims, claim.Name)
				// Poke the controller with deletion event. Cloned claim is
				// needed to prevent races (and we would get a clone from etcd
				// too).
				clone, _ := conversion.NewCloner().DeepCopy(claim)
				claimClone := clone.(*api.PersistentVolumeClaim)
				reactor.claimSource.Delete(claimClone)
				return nil
			},
		},
		{
			// deleteVolume with a bound volume. Check the claim is Lost.
			"5-4 - delete volume",
			newVolumeArray("volume5-4", "10Gi", "uid5-4", "claim5-4", api.VolumeBound, api.PersistentVolumeReclaimRetain),
			novolumes,
			newClaimArray("claim5-4", "uid5-4", "1Gi", "volume5-4", api.ClaimBound, annBoundByController, annBindCompleted),
			newClaimArray("claim5-4", "uid5-4", "1Gi", "volume5-4", api.ClaimLost, annBoundByController, annBindCompleted),
			[]string{"Warning ClaimLost"}, noerrors,
			// Custom test function that generates a delete event
			func(ctrl *PersistentVolumeController, reactor *volumeReactor, test controllerTest) error {
				obj := ctrl.volumes.store.List()[0]
				volume := obj.(*api.PersistentVolume)
				// Remove the volume from list of resulting volumes.
				delete(reactor.volumes, volume.Name)
				// Poke the controller with deletion event. Cloned volume is
				// needed to prevent races (and we would get a clone from etcd
				// too).
				clone, _ := conversion.NewCloner().DeepCopy(volume)
				volumeClone := clone.(*api.PersistentVolume)
				reactor.volumeSource.Delete(volumeClone)
				return nil
			},
		},
	}

	for ix, test := range tests {
		glog.V(4).Infof("starting test %q", test.name)

		// Initialize the controller
		client := &fake.Clientset{}
		volumeSource := framework.NewFakeControllerSource()
		claimSource := framework.NewFakeControllerSource()
		ctrl := newTestController(client, volumeSource, claimSource)
		reactor := newVolumeReactor(client, ctrl, volumeSource, claimSource, test.errors)
		for _, claim := range test.initialClaims {
			claimSource.Add(claim)
			reactor.claims[claim.Name] = claim
		}
		for _, volume := range test.initialVolumes {
			volumeSource.Add(volume)
			reactor.volumes[volume.Name] = volume
		}

		// Start the controller
		defer ctrl.Stop()
		go ctrl.Run()

		// Wait for the controller to pass initial sync.
		for !ctrl.isFullySynced() {
			time.Sleep(10 * time.Millisecond)
		}

		count := reactor.getChangeCount()

		// Call the tested function
		err := test.test(ctrl, reactor, test)
		if err != nil {
			t.Errorf("Test %q initial test call failed: %v", test.name, err)
		}

		for reactor.getChangeCount() < count+expectedChanges[ix] {
			reactor.waitTest()
		}

		evaluateTestResults(ctrl, reactor, test, t)
	}
}
