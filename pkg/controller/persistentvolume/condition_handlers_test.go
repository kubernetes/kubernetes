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

	"fmt"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/volume"
)

func TestContextIsObservable(t *testing.T) {
	volume := makeTestVolume()
	claim := makeTestClaim()
	ctx := makeTestTaskContext()

	// volume not found in local store
	if ctx.isValidVolume(volume.Name) {
		t.Error("Expected volume to be valid")
	}

	// volume not found in local store
	if ctx.isValidClaim(claimToClaimKey(claim)) {
		t.Error("Expected claim to be valid")
	}

	// a watch adds the volume and claim to the local store
	ctx.host.volumeStore.Add(volume)
	ctx.host.claimStore.Add(claim)

	if !ctx.isValidVolume(volume.Name) {
		t.Error("Expected volume to be valid")
	}

	if !ctx.isValidClaim(claimToClaimKey(claim)) {
		t.Error("Expected claim to be valid")
	}
}

func TestVolumeBindFixing(t *testing.T) {
	volume := makeTestVolume()
	claim := makeTestClaim()

	// make claim and volume bound to each other
	claim.Spec.VolumeName = volume.Name
	volume.Spec.ClaimRef = &api.ObjectReference{
		Name:      claim.Name,
		Namespace: claim.Namespace,
		UID:       claim.UID,
	}

	ctx := makeTestTaskContext()
	ctx.volume = volume
	ctx.claim = claim
	ctx.claimExists = true
	ctx.host.volumeStore.Add(volume)
	ctx.host.volumeStore.Add(claim)

	if !ctx.isValidVolume(volume.Name) {
		t.Error("Expected bind to be valid")
	}

	// make claim and volume mis-bind
	ctx.volume.Spec.ClaimRef = &api.ObjectReference{
		Name:      "notTheTestClaim",
		Namespace: "somewhereNotHere",
		UID:       "def567",
	}

	if ctx.isVolumeBound() {
		t.Error("Expected volume to be bound")
	}

	err := ctx.removeInvalidClaimRef()
	if err != nil {
		t.Errorf("Unexpected error removing invalid claimRef: %v", err)
	}

	mockClient, _ := ctx.host.client.(taskClient)
	pv, _ := mockClient.GetPersistentVolume("anything")
	if pv.Spec.ClaimRef != nil {
		t.Errorf("Expected nil claimRef on volume")
	}
}

func TestVolumeBoundConditionSatisfied(t *testing.T) {
	volume := makeTestVolume()
	claim := makeTestClaim()

	// make claim and volume bound to each other
	claim.Spec.VolumeName = volume.Name
	volume.Spec.ClaimRef = &api.ObjectReference{
		Name:      claim.Name,
		Namespace: claim.Namespace,
		UID:       claim.UID,
	}

	ctx := makeTestTaskContext()
	ctx.host.volumeStore.Add(volume)
	ctx.host.claimStore.Add(claim)

	ctx.syncVolumeBoundCondition(volume.Name)

	volume, _ = ctx.host.client.GetPersistentVolume("anything")
	if len(volume.Status.Conditions) != 1 {
		t.Errorf("Expected 1 condition (bound) got got %d", len(volume.Status.Conditions))
	}
	if volume.Status.Conditions[0].Status != api.ConditionTrue {
		t.Errorf("Expected %v but got %v", api.ConditionTrue, volume.Status.Conditions[0].Status)
	}
	if volume.Status.Conditions[0].Reason != "Bound" {
		t.Errorf("Expected %v but got %v", "Bound", volume.Status.Conditions[0].Reason)
	}
	if volume.Status.Conditions[0].Message == "" {
		t.Errorf("Expected non-empty Message")
	}
}

func TestVolumeBoundConditionUnsatisfied(t *testing.T) {
	volume := makeTestVolume()
	claim := makeTestClaim()

	// definitely not bound
	claim.Spec.VolumeName = ""
	volume.Spec.ClaimRef = nil

	ctx := makeTestTaskContext()
	ctx.host.volumeStore.Add(volume)
	ctx.host.claimStore.Add(claim)

	ctx.syncVolumeBoundCondition(volume.Name)

	if len(ctx.volume.Status.Conditions) != 1 {
		t.Errorf("Expected 1 condition (bound) got got %d", len(ctx.volume.Status.Conditions))
	}
	if ctx.volume.Status.Conditions[0].Status != api.ConditionFalse {
		t.Errorf("Expected %v but got %v", api.ConditionFalse, ctx.volume.Status.Conditions[0].Status)
	}
	if ctx.volume.Status.Conditions[0].Reason != "Unbound" {
		t.Errorf("Expected %v but got %v", "Unbound", ctx.volume.Status.Conditions[0].Reason)
	}
	if ctx.volume.Status.Conditions[0].Message != "" {
		t.Error("Expected empty Message")
	}
}

func TestVolumeRecycledConditionUnsatisfied(t *testing.T) {
	volume := makeTestVolume()

	// bound but missing
	volume.Spec.ClaimRef = &api.ObjectReference{
		Name:      "foo",
		Namespace: "bar",
		UID:       "123abc",
	}

	ctx := makeTestTaskContext()
	ctx.host.volumeStore.Add(volume)

	if !ctx.isValidVolume(volume.Name) {
		t.Errorf("Unexpected error")
	}

	if !ctx.isBindReleased() {
		t.Errorf("Expected bind to be released. volume has claimRef but claim is missing")
	}

	err := ctx.syncVolumeBoundCondition(volume.Name)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if len(ctx.volume.Status.Conditions) != 2 {
		t.Errorf("Expected 2 conditions (bound + recycled) got got %d", len(ctx.volume.Status.Conditions))
	}

	volume, err = ctx.host.client.GetPersistentVolume("anything")
	if volume == nil {
		t.Fatal("Unexpected nil volume")
	}

	for _, condition := range volume.Status.Conditions {
		if condition.Type == api.PersistentVolumeBound {
			if condition.Status != api.ConditionFalse {
				t.Errorf("Expected %v but got %v", api.ConditionFalse, condition.Status)
			}
			if condition.Reason != "Released" {
				t.Errorf("Expected %v but got %v", "Released", condition.Reason)
			}
			if condition.Message != "Bound claim is missing" {
				t.Error("Expected bound claim message")
			}
		}
		if condition.Type == api.PersistentVolumeRecycled {
			if condition.Status != api.ConditionFalse {
				t.Errorf("Expected %v but got %v", api.ConditionFalse, condition.Status)
			}
			if condition.Reason != "ClaimNotFound" {
				t.Errorf("Expected %v but got %v", "New", condition.Reason)
			}
			if condition.Message != "Reclaim policy not yet applied" {
				t.Error("Expected reclaim policy message")
			}
		}
	}
}

func TestClaimBoundConditionSatisfied(t *testing.T) {
	volume := makeTestVolume()
	claim := makeTestClaim()

	// definitely not bound
	claim.Spec.VolumeName = ""
	volume.Spec.ClaimRef = nil

	ctx := makeTestTaskContext()
	ctx.host.volumeStore.Add(volume)
	ctx.host.claimStore.Add(claim)

	err := ctx.syncClaimBoundCondition(claimToClaimKey(claim))
	if err != nil {
		t.Errorf("Unexpected error syncing claim bound condition: %v", err)
	}

	if len(ctx.claim.Status.Conditions) != 1 {
		t.Errorf("Expected 1 condition (bound) got got %d", len(ctx.claim.Status.Conditions))
	}
	if ctx.claim.Status.Conditions[0].Status != api.ConditionTrue {
		t.Errorf("Expected %v but got %v", api.ConditionTrue, ctx.claim.Status.Conditions[0].Status)
	}
	if ctx.claim.Status.Conditions[0].Reason != "Bound" {
		t.Errorf("Expected %v but got %v", "Bound", ctx.claim.Status.Conditions[0].Reason)
	}
	if ctx.claim.Status.Conditions[0].Message == "" {
		t.Error("Unexpected empty Message")
	}
	if ctx.claim.Spec.VolumeName != volume.Name {
		t.Errorf("Expected %v but got %v", volume.Name, ctx.claim.Spec.VolumeName)
	}
}

func TestVolumeProvisionedSatisfied(t *testing.T) {
	claim := makeTestClaim()
	pv := makeTestVolume()

	// make claim and volume bound to each other
	claim.Spec.VolumeName = pv.Name
	pv.Spec.ClaimRef = &api.ObjectReference{
		Name:      claim.Name,
		Namespace: claim.Namespace,
		UID:       claim.UID,
	}

	pv.Status.Conditions = append(pv.Status.Conditions, api.PersistentVolumeCondition{
		Type:   api.PersistentVolumeProvisioned,
		Status: api.ConditionFalse,
	})

	ctx := makeTestTaskContext()
	ctx.host.volumeStore.Add(pv)
	ctx.host.volumeStore.Add(claim)

	err := ctx.syncVolumeProvisionedCondition(pv.Name)
	if err != nil {
		t.Error("Unexpected error syncing provisioned condition: %v", err)
	}

	for _, condition := range ctx.volume.Status.Conditions {
		if condition.Type == api.PersistentVolumeProvisioned {
			if condition.Status != api.ConditionTrue {
				t.Errorf("Expected %v but got %v", api.ConditionTrue, condition.Status)
			}
			if condition.Reason != "Provisioned" {
				t.Errorf("Expected %v but got %v", "Provisioned", condition.Reason)
			}
			if condition.Message != "" {
				t.Error("Unexpected non-empty Message")
			}
		}
	}
}

func TestVolumeProvisionedUnsatisfied(t *testing.T) {
	claim := makeTestClaim()
	pv := makeTestVolume()

	// definitely not bound to each other
	claim.Spec.VolumeName = ""
	pv.Spec.ClaimRef = nil

	pv.Status.Conditions = append(pv.Status.Conditions, api.PersistentVolumeCondition{
		Type:   api.PersistentVolumeProvisioned,
		Status: api.ConditionFalse,
	})

	ctx := makeTestTaskContext()
	ctx.host.volumeStore.Add(pv)
	ctx.host.volumeStore.Add(claim)

	// no plugins, so no provisioners
	pluginMgr := &volume.VolumePluginMgr{}
	_ = pluginMgr.InitPlugins([]volume.VolumePlugin{}, ctx.host.volumePluginHost)
	ctx.host.pluginMgr = *pluginMgr

	err := ctx.syncVolumeProvisionedCondition(pv.Name)
	if err == nil {
		t.Error("Unexpected nil error. Was expecting an error due to missing plugins")
	}

	for _, condition := range ctx.volume.Status.Conditions {
		if condition.Type == api.PersistentVolumeProvisioned {
			if condition.Status != api.ConditionFalse {
				t.Errorf("Expected %v but got %v", api.ConditionFalse, condition.Status)
			}
			if condition.Reason != "Failed" {
				t.Errorf("Expected %v but got %v", "Failed", condition.Reason)
			}
			if condition.Message == "" {
				t.Error("Expected error message but got empty")
			}
		}
	}
}

func TestVolumeRecycledCondition(t *testing.T) {
	volume := makeTestVolume()
	volume.Spec.ClaimRef = &api.ObjectReference{
		Name:      "noclaim",
		Namespace: "deletedbyuser",
	}

	ctx := makeTestTaskContext()
	ctx.host.volumeStore.Add(volume)
	ctx.volume = volume

	if err := ctx.addVolumeRecycledCondition(); err != nil {
		t.Errorf("Unexpected error setting volume recycled condition: %v", err)
	}

	if len(ctx.volume.Status.Conditions) != 2 {
		t.Errorf("Expected 2 conditions (bound + recycled) got got %d", len(ctx.volume.Status.Conditions))
	}

	for _, condition := range ctx.volume.Status.Conditions {
		if condition.Type == api.PersistentVolumeRecycled {
			if condition.Status != api.ConditionFalse {
				t.Errorf("Expected %v but got %v", api.ConditionFalse, condition.Status)
			}
			if condition.Reason != "ClaimNotFound" {
				t.Errorf("Expected %v but got %v", "ClaimNotFound", condition.Reason)
			}
			if condition.Message == "" {
				t.Error("Unexpected empty Message")
			}
		}
	}

	policies := []api.PersistentVolumeReclaimPolicy{
		api.PersistentVolumeReclaimRetain,
		api.PersistentVolumeReclaimRecycle,
		api.PersistentVolumeReclaimDelete,
	}

	for _, policy := range policies {
		volume.Spec.PersistentVolumeReclaimPolicy = policy
		ctx.volume = volume

		if err := ctx.setVolumeRecycledConditionUnsatisfied(); err != nil {
			t.Errorf("Unexpected error setting volume recycled condition: %v", err)
		}

		if len(ctx.volume.Status.Conditions) != 2 {
			t.Errorf("Expected 2 conditions (bound + recycled) got got %d", len(ctx.volume.Status.Conditions))
		}

		ctx.host.volumeStore.Add(ctx.volume)
		err := ctx.syncVolumeRecycledCondition(volume.Name)
		if err != nil {
			t.Errorf("Unexpected error syncing recycled condition: %v", err)
		}

		// volume has default reclaim policy: Retain
		pv, err := ctx.host.client.GetPersistentVolume(volume.Name)
		if err != nil {
			t.Errorf("Unexpected error getting volume: %v", err)
		}

		// delete policy removes the the volume from the system
		if pv == nil && policy == api.PersistentVolumeReclaimDelete {
			continue

		}
		for _, condition := range pv.Status.Conditions {
			if condition.Type == api.PersistentVolumeRecycled {
				if condition.Status != api.ConditionTrue {
					t.Errorf("For policy %v, expected %v but got %v", policy, api.ConditionTrue, condition.Status)
				}
				if condition.Reason != "Recycled" {
					t.Errorf("For policy %v, expected %v but got %v", policy, "Recycled", condition.Reason)
				}
				message := fmt.Sprintf("Reclaim policy is %v", policy)
				if condition.Message != message {
					t.Errorf("For policy %v, expected '%v' but got '%v'", policy, message, condition.Message)
				}
			}
		}
	}
}
