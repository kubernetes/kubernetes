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
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/types"
	ioutil "k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

/*

Syncing PersistentVolumes (volumes) and PersistentVolumeClaims (claims) requires establishing a
bi-directional bind between the two objects.  Neither is considered fully bound until both accept their binds.

API Conditions are used to express the status of a volume or claim and are persisted as a subresource in the API.
Using Conditions w/ Status, Type, LastProbeTime, Reason, and Message is more expressive than Phase and allows an
effective means of locking by relying on resource version errors.

Single function handlers are assigned to 1 type of Condition. Each time an object syncs in the controller, one task
per condition is spawned. Each condition is handled concurrently as separate tasks.

Reconciling an object follows this lifecycle:

  Task             When                 Action
  syncObject       every watch event    for each condition, run a Condition handler
  syncCondition    every watch event    observe current status of the object
                                          if status matches spec, nothing to do, return
                                          else save current status
*/

// syncVolumeBoundCondition inspects the state of the volume's bind to a claim.
// only 3 things needed for the Bound condition to reflect reality
// 1. Validate the bi-directional bind.
// 2. Persist updates to Condition as needed.
// 3. Add Recycled condition when needed (arguably at object birth w/ Status=false, Reason=New)
func (c *taskContext) syncVolumeBoundCondition(key interface{}) error {
	glog.V(5).Infof("PersistentVolume[%s] syncing %v condition\n", key, api.PersistentVolumeBound)

	// ensures the volume is found in the local store
	// also initializes the context, as needed.
	if !c.isValidVolume(key) {
		// nothing to observe
		return nil
	}

	// not much to do if the volume is not bound.
	if !c.isVolumeBound() {
		glog.V(5).Infof("PersistentVolume[%s] is unbound\n", c.VolumeKey())
		// save pv.Status.Condition, as needed. No change = no-op.
		if err := c.setVolumeBoundConditionUnsatisfied(); err != nil {
			return err
		}
	}

	// Remove any invalid binds that may exist. This is possible if the same claim is being sync'd concurrently.
	// A claim could take ownership of more than one volume before the winning claim persists the single volumeName.
	// The rest are orphaned but are unavailable to other volumes until the bind is removed.
	if c.isVolumeAndClaimBindMismatched() {
		glog.V(5).Infof("PersistentVolume[%s] has binding mismatch with claim %s\n", c.ClaimKey())
		if err := c.removeInvalidClaimRef(); err != nil {
			return err
		}
	}

	// a volume has a satisfied Bound Condition when the bind to its claim is consummated.
	if c.isVolumeBound() {
		// save pv.Status.Condition, as needed. No change = no-op.
		if err := c.setVolumeBoundConditionSatisfied(); err != nil {
			return err
		}
	}

	// volumes are released of their claims when the user deletes the claim.
	// the volume would have a claimRef but the claim would 404 in the API and not be found in the local watch store.
	if c.isBindReleased() {
		// add pv.Status.Condition, as needed. No change = no-op.
		// creation of this condition triggers the reclamation of the resource per its retention policy
		if err := c.addVolumeRecycledCondition(); err != nil {
			return err
		}
		// a watch event was triggered by persisting the volume above.
		// the next sync will have the correct Recycled condition.
	}

	glog.V(5).Infof("PersistentVolume[%s] Bound condition reconciled", key)
	return nil
}

func (c *taskContext) syncClaimBoundCondition(key interface{}) error {
	glog.V(5).Infof("PersistentVolumeClaim[%s] syncing %v condition\n", key, api.PersistentVolumeClaimBound)

	if !c.isValidClaim(key) {
		// nothing to observe
		return nil
	}

	// the same claim syncing concurrently can bind many volumes to one claim.
	// volumes with matching claimRef are always returned.
	foundMatch, err := c.attemptMatch()
	if err != nil {
		glog.V(5).Infof("PersistentVolumeClaim[%s] err matching volume: %v", claimToClaimKey(c.claim), err)
		return nil
	}

	if !foundMatch {
		glog.V(5).Infof("PersistentVolumeClaim[%s] no match found", claimToClaimKey(c.claim))
		if err := c.setClaimBoundConditionUnsatisfied(); err != nil {
			return err
		}
		// better luck next time
		return nil
	}

	// saving pv.Spec.ClaimRef is the binding transaction.
	// this will no-op if already set
	if err := c.bindVolume(); err != nil {
		return fmt.Errorf("PersistentVolumeClaim[%s] err binding volume: %#v", err)
	}

	// put the volume name on the claim to complete the bind.
	// when dupes happen concurrently, only one racer will persist claim.Spec.VolumeName
	// De-duping those claimed volumes happens when the volume syncs.
	// this will no-op if already set
	if err := c.bindClaim(); err != nil {
		return fmt.Errorf("PersistentVolumeClaim[%s] err binding claim: %#v", err)
	}

	// save claim.Status.Condition, as needed. No change = no-op.
	if err := c.setClaimBoundConditionSatisfied(); err != nil {
		return fmt.Errorf("error saving claim.Status: %v", err)
	}

	glog.V(5).Infof("PersistentVolumeClaim[%s] Bound condition reconciled", claimToClaimKey(c.claim))

	return nil
}

func (c *taskContext) syncVolumeRecycledCondition(key interface{}) error {
	glog.V(5).Infof("PersistentVolumeClaim[%s] syncing %v condition\n", key, api.PersistentVolumeClaimBound)

	if !c.isValidVolume(key) {
		// nothing to observe
		return nil
	}

	if c.volumeRequiresRecycling() {
		// sets the Condition specific to the reclaim policy
		if err := c.recycleVolume(); err != nil {
			return fmt.Errorf("PersistentVolume[%s] error recycling: %v", c.VolumeKey(), err)
		}
	}

	glog.V(5).Infof("PersistentVolumeClaim[%s] Bound condition reconciled")
	return nil
}

func (c *taskContext) syncVolumeProvisionedCondition(key interface{}) error {
	glog.V(5).Infof("PersistentVolume[%s] syncing %v condition\n", key, api.PersistentVolumeProvisioned)

	if !c.isValidVolume(key) {
		// nothing to observe
		return nil
	}

	if c.volumeRequiresProvisioning() {
		if err := c.provisionVolume(); err != nil {
			return fmt.Errorf("PersistentVolume[%s] error provisioning: %v", c.VolumeKey(), err)
		}
	}

	glog.V(5).Infof("PersistentVolumeClaim[%s] Bound condition reconciled")
	return nil
}

// volumeRequiresProvisioning returns true if the volume has an unsatisfied provisioning condition
func (c *taskContext) volumeRequiresProvisioning() bool {
	for _, condition := range c.volume.Status.Conditions {
		if condition.Type == api.PersistentVolumeProvisioned && condition.Status == api.ConditionFalse {
			return true
		}
	}

	// add Jan's re-try stuff here or
	// perhaps something timebased using condition.LastProbeTime and condition.LastTransitionTime (i.e, don't try forever)
	return false
}

// volumeRequiresRecycling returns true if the volume has an unsatisfied recycled condition
func (c *taskContext) volumeRequiresRecycling() bool {
	if !c.isBindReleased() {
		return false
	}
	for _, condition := range c.volume.Status.Conditions {
		if condition.Type == api.PersistentVolumeRecycled && condition.Status == api.ConditionFalse {
			return true
		}
	}

	// add Jan's re-try stuff here or
	// perhaps something timebased using condition.LastProbeTime and condition.LastTransitionTime (i.e, don't try forever)
	return false
}

func (c *taskContext) doReclaimPolicyRecycle() error {
	spec := volume.NewSpecFromPersistentVolume(c.volume, false)
	plugin, err := c.host.pluginMgr.FindRecyclablePluginBySpec(spec)
	if err != nil {
		return fmt.Errorf("PersistentVolume[%s] err getting recyclable plugin: %v", c.volume.Name, err)
	}
	if plugin == nil {
		return fmt.Errorf("PersistentVolume[%s] nil recyclable plugin", c.volume.Name)
	}

	volRecycler, err := plugin.NewRecycler(spec)
	if err != nil {
		return fmt.Errorf("Could not obtain Recycler for spec: %#v  error: %v", spec, err)
	}
	// blocks until completion
	if err := volRecycler.Recycle(); err != nil {
		return fmt.Errorf("PersistentVolume[%s] failed recycling: %+v", c.volume.Name, err)
	}

	// success!
	return nil
}

func (c *taskContext) doReclaimPolicyDelete() error {
	spec := volume.NewSpecFromPersistentVolume(c.volume, false)
	plugin, err := c.host.pluginMgr.FindDeletablePluginBySpec(spec)
	if err != nil {
		return fmt.Errorf("PersistentVolume[%s] err getting deletable plugin: %v", c.volume.Name, err)
	}

	deleter, err := plugin.NewDeleter(spec)
	if err != nil {
		return fmt.Errorf("PersistentVolume[%s] could not obtain Deleter error: %v", c.volume.Name, err)
	}
	// blocks until completion
	err = deleter.Delete()
	if err != nil {
		return fmt.Errorf("PersistentVolume[%s] failed deletion: %+v", c.volume.Name, err)
	}

	glog.V(5).Infof("PersistentVolume[%s] successfully deleted through plugin\n", c.volume.Name)
	if err := c.host.client.DeletePersistentVolume(c.volume); err != nil {
		return fmt.Errorf("error deleting persistent volume: %+v", err)
	}

	return nil
}

func (c *taskContext) doReclaimPolicyRetain() error {
	if err := c.setVolumeCondition(api.PersistentVolumeRecycled, api.ConditionTrue, "Recycled", "Reclaim policy is Retain"); err != nil {
		return err
	}
	glog.V(5).Infof("PersistentVolume[%s] is retained.\n", c.volume.Name)
	return nil
}

func (c *taskContext) provisionVolume() error {
	if err := c.doProvisionVolume(); err != nil {
		// save failed status w/ error message
		if err := c.setVolumeCondition(api.PersistentVolumeProvisioned, api.ConditionFalse, "Failed", err.Error()); err != nil {
			return fmt.Errorf("PersistentVolume[%s] failed provisioning status update: %v", err)
		}
		glog.Errorf("PersistentVolume[%s] failed provisioning: %v", c.VolumeKey(), err)
		return err
	}

	if err := c.setVolumeCondition(api.PersistentVolumeProvisioned, api.ConditionTrue, "Provisioned", ""); err != nil {
		return fmt.Errorf("PersistentVolume[%s] failed saving status: %v", err)
	}

	glog.V(5).Infof("PersistentVolume[%s] is provisioned.\n", c.volume.Name)
	return nil
}

func (c *taskContext) doProvisionVolume() error {
	spec := volume.NewSpecFromPersistentVolume(c.volume, false)
	plugin, err := c.host.pluginMgr.FindCreatablePluginBySpec(spec)
	if err != nil {
		return fmt.Errorf("PersistentVolume[%s] err getting provisionable plugin: %v", c.volume.Name, err)
	}

	tags := make(map[string]string)
	tags[cloudVolumeCreatedForVolumeNameTag] = c.volume.Name
	if c.volume.Spec.ClaimRef != nil {
		tags[cloudVolumeCreatedForClaimNameTag] = c.volume.Spec.ClaimRef.Name
		tags[cloudVolumeCreatedForClaimNamespaceTag] = c.volume.Spec.ClaimRef.Namespace
	}

	volumeOptions := volume.VolumeOptions{
		Capacity:                      c.volume.Spec.Capacity[api.ResourceName(api.ResourceStorage)],
		AccessModes:                   c.volume.Spec.AccessModes,
		PersistentVolumeReclaimPolicy: c.volume.Spec.PersistentVolumeReclaimPolicy,
		CloudTags:                     &tags,
	}

	provisioner, err := plugin.NewProvisioner(volumeOptions)
	if err != nil {
		return fmt.Errorf("PersistentVolume[%s] could not obtain Provisioner error: %v", c.volume.Name, err)
	}

	// volume is mutated by provisioner with IDs from provider or other errata, requires saving afterwards.
	clone, _ := conversion.NewCloner().DeepCopy(c.volume)
	volumeClone, _ := clone.(*api.PersistentVolume)

	// blocks until completion.
	err = provisioner.Provision(volumeClone)
	if err != nil {
		return fmt.Errorf("PersistentVolume[%s] failed provisioning with provider: %+v", c.volume.Name, err)
	}

	glog.V(5).Infof("PersistentVolume[%s] successfully provisioned through plugin\n", c.volume.Name)
	updatedVolume, err := c.host.client.UpdatePersistentVolume(volumeClone)
	if err != nil {
		// each provisioner should implement its own lookup by volume name to prevent resource leakage.
		// volume health checks and reconcilation with the provider would help.
		return err
	}

	c.volume = updatedVolume

	return nil
}

func (c *taskContext) recycleVolume() error {
	var err error
	var policy api.PersistentVolumeReclaimPolicy
	switch c.volume.Spec.PersistentVolumeReclaimPolicy {
	case api.PersistentVolumeReclaimRecycle:
		err, policy = c.doReclaimPolicyRecycle(), api.PersistentVolumeReclaimRecycle
	case api.PersistentVolumeReclaimDelete:
		err, policy = c.doReclaimPolicyDelete(), api.PersistentVolumeReclaimDelete
	case api.PersistentVolumeReclaimRetain:
		err, policy = c.doReclaimPolicyRetain(), api.PersistentVolumeReclaimRetain
	}
	if err != nil {
		return fmt.Errorf("PersistentVolume[%s] failed recycling: %v", c.volume.Name, err)
	}
	if err := c.setVolumeRecycleConditionSatisfied("Recycled", fmt.Sprintf("Reclaim policy is %v", policy)); err != nil {
		return fmt.Errorf("PersistentVolume[%s] error saving status: %v", c.VolumeKey(), err)
	}
	return nil
}

func (c *taskContext) attemptMatch() (bool, error) {
	volume, e := c.host.volumeStore.findBestMatchForClaim(c.claim)
	if e != nil {
		return false, fmt.Errorf("PersistentVolumeClaim[%s] error finding match")
	}
	if volume != nil {
		c.volume = volume
		return true, nil
	}
	return false, nil
}

func (c *taskContext) bindVolume() error {
	if c.volume.Spec.ClaimRef == nil {
		clone, _ := conversion.NewCloner().DeepCopy(c.volume)
		volumeClone, _ := clone.(*api.PersistentVolume)
		claimRef, _ := api.GetReference(c.claim)
		volumeClone.Spec.ClaimRef = claimRef
		updatedVolume, err := c.host.client.UpdatePersistentVolume(volumeClone)
		if err != nil {
			return err
		}
		c.volume = updatedVolume
	}
	return nil
}

func (c *taskContext) bindClaim() error {
	if c.claim.Spec.VolumeName == "" {
		clone, _ := conversion.NewCloner().DeepCopy(c.claim)
		claimClone, _ := clone.(*api.PersistentVolumeClaim)
		claimClone.Spec.VolumeName = c.volume.Name
		updatedClaim, err := c.host.client.UpdatePersistentVolumeClaim(claimClone)
		if err != nil {
			return err
		}
		c.claim = updatedClaim
	}
	return nil
}

func (c *taskContext) isBindReleased() bool {
	return c.volume != nil && c.volume.Spec.ClaimRef != nil && !c.claimExists

}
func (c *taskContext) removeInvalidClaimRef() error {
	clone, _ := conversion.NewCloner().DeepCopy(c.volume)
	volumeClone, _ := clone.(*api.PersistentVolume)
	volumeClone.Spec.ClaimRef = nil
	updatedVolume, err := c.host.client.UpdatePersistentVolume(volumeClone)
	if err != nil {
		return err
	}
	c.volume = updatedVolume
	return nil
}

func (c *taskContext) setVolumeBoundConditionSatisfied() error {
	for i, condition := range c.volume.Status.Conditions {
		if condition.Type == api.PersistentVolumeBound {
			if condition.Status == api.ConditionTrue {
				return nil
			}
			clone, _ := conversion.NewCloner().DeepCopy(c.volume)
			volumeClone, _ := clone.(*api.PersistentVolume)
			volumeClone.Status.Conditions[i].Status = api.ConditionTrue
			volumeClone.Status.Conditions[i].Reason = "Bound"
			volumeClone.Status.Conditions[i].Message = fmt.Sprintf("Bound to %s", c.ClaimKey())
			volumeClone.Status.Conditions[i].LastProbeTime = unversioned.Now()
			volumeClone.Status.Conditions[i].LastTransitionTime = unversioned.Now()

			updatedVolume, err := c.host.client.UpdatePersistentVolumeStatus(volumeClone)
			if err != nil {
				return err
			}
			c.volume = updatedVolume
		}
	}
	return nil
}

func (c *taskContext) setClaimBoundConditionSatisfied() error {
	for i, condition := range c.claim.Status.Conditions {
		if condition.Type == api.PersistentVolumeClaimBound {
			if condition.Status == api.ConditionTrue {
				return nil
			}
			clone, _ := conversion.NewCloner().DeepCopy(c.claim)
			claimClone, _ := clone.(*api.PersistentVolumeClaim)
			claimClone.Status.Conditions[i].Status = api.ConditionTrue
			claimClone.Status.Conditions[i].Reason = "Bound"
			claimClone.Status.Conditions[i].Message = fmt.Sprintf("Bound to %s", c.volume.Name)
			claimClone.Status.Conditions[i].LastProbeTime = unversioned.Now()
			claimClone.Status.Conditions[i].LastTransitionTime = unversioned.Now()

			updatedClaim, err := c.host.client.UpdatePersistentVolumeClaimStatus(claimClone)
			if err != nil {
				return err
			}
			c.claim = updatedClaim
		}
	}
	return nil
}

func (c *taskContext) setVolumeBoundConditionUnsatisfied() error {
	for i, condition := range c.volume.Status.Conditions {
		if condition.Type == api.PersistentVolumeBound {
			if condition.Status == api.ConditionFalse && condition.Reason == "Unbound" {
				return nil
			}
			clone, _ := conversion.NewCloner().DeepCopy(c.volume)
			volumeClone, _ := clone.(*api.PersistentVolume)
			volumeClone.Status.Conditions[i].Status = api.ConditionFalse
			volumeClone.Status.Conditions[i].Reason = "Unbound"
			volumeClone.Status.Conditions[i].Message = ""
			volumeClone.Status.Conditions[i].LastProbeTime = unversioned.Now()
			volumeClone.Status.Conditions[i].LastTransitionTime = unversioned.Now()

			updatedVolume, err := c.host.client.UpdatePersistentVolumeStatus(volumeClone)
			if err != nil {
				return err
			}
			c.volume = updatedVolume
		}
	}
	return nil
}

func (c *taskContext) setClaimBoundConditionUnsatisfied() error {
	for i, condition := range c.claim.Status.Conditions {
		if condition.Type == api.PersistentVolumeClaimBound {
			if condition.Status == api.ConditionFalse && condition.Reason == "Unbound" {
				// no-op! final no change required.
				return nil
			}
			clone, _ := conversion.NewCloner().DeepCopy(c.claim)
			claimClone, _ := clone.(*api.PersistentVolumeClaim)

			claimClone.Status.Conditions[i].Status = api.ConditionFalse
			claimClone.Status.Conditions[i].Reason = "Unbound"
			claimClone.Status.Conditions[i].Message = ""
			claimClone.Status.Conditions[i].LastProbeTime = unversioned.Now()
			claimClone.Status.Conditions[i].LastTransitionTime = unversioned.Now()

			updatedClone, err := c.host.client.UpdatePersistentVolumeClaimStatus(claimClone)
			if err != nil {
				return err
			}
			c.claim = updatedClone
		}
	}
	return nil
}

// addVolumeRecycledCondition only adds a Recycled condition if one does not already exist
func (c *taskContext) addVolumeRecycledCondition() error {
	// no need to add the condition if we already have it
	for _, condition := range c.volume.Status.Conditions {
		if condition.Type == api.PersistentVolumeRecycled {
			return nil
		}
	}
	return c.setVolumeRecycledConditionUnsatisfied()
}

// setVolumeRecycledConditionUnsatisfied sets the volume's Conditions
// to be in the correct state for recycling.
func (c *taskContext) setVolumeRecycledConditionUnsatisfied() error {
	clone, _ := conversion.NewCloner().DeepCopy(c.volume)
	volumeClone, _ := clone.(*api.PersistentVolume)

	volumeClone.Status.Conditions = []api.PersistentVolumeCondition{
		{
			Type:               api.PersistentVolumeBound,
			Status:             api.ConditionFalse,
			Reason:             "Released",
			Message:            "Bound claim is missing",
			LastProbeTime:      unversioned.Now(),
			LastTransitionTime: unversioned.Now(),
		},
		{
			Type:               api.PersistentVolumeRecycled,
			Status:             api.ConditionFalse,
			Reason:             "ClaimNotFound",
			Message:            "Reclaim policy not yet applied",
			LastProbeTime:      unversioned.Now(),
			LastTransitionTime: unversioned.Now(),
		},
	}

	updatedVolume, err := c.host.client.UpdatePersistentVolumeStatus(volumeClone)
	if err != nil {
		return err
	}
	c.volume = updatedVolume

	return nil
}

func (c *taskContext) setVolumeRecycleConditionSatisfied(reason, message string) error {
	for i, condition := range c.volume.Status.Conditions {
		if condition.Type == api.PersistentVolumeRecycled {
			if condition.Status == api.ConditionTrue {
				return nil
			}
			clone, _ := conversion.NewCloner().DeepCopy(c.volume)
			volumeClone, _ := clone.(*api.PersistentVolume)
			volumeClone.Status.Conditions[i].Status = api.ConditionTrue
			volumeClone.Status.Conditions[i].Reason = reason
			volumeClone.Status.Conditions[i].Message = message
			volumeClone.Status.Conditions[i].LastProbeTime = unversioned.Now()
			volumeClone.Status.Conditions[i].LastTransitionTime = unversioned.Now()

			updatedVolume, err := c.host.client.UpdatePersistentVolumeStatus(volumeClone)
			if err != nil {
				return err
			}
			c.volume = updatedVolume
		}
	}
	return nil
}

func (c *taskContext) setVolumeCondition(conditionType api.PersistentVolumeConditionType, status api.ConditionStatus, reason, message string) error {
	for i, condition := range c.volume.Status.Conditions {
		if condition.Type == conditionType {
			if condition.Status == api.ConditionTrue {
				return nil
			}
			clone, _ := conversion.NewCloner().DeepCopy(c.volume)
			volumeClone, _ := clone.(*api.PersistentVolume)
			volumeClone.Status.Conditions[i].Status = status
			volumeClone.Status.Conditions[i].Reason = reason
			volumeClone.Status.Conditions[i].Message = message
			volumeClone.Status.Conditions[i].LastProbeTime = unversioned.Now()
			volumeClone.Status.Conditions[i].LastTransitionTime = unversioned.Now()

			updatedVolume, err := c.host.client.UpdatePersistentVolumeStatus(volumeClone)
			if err != nil {
				return err
			}
			c.volume = updatedVolume
		}
	}
	return nil
}

func (c *taskContext) isClaimUnbound() bool {
	return c.claim.Spec.VolumeName == ""
}

// isVolumeAndClaimBindMismatched verifies that both volume and claim are bound, but not to each other.
// pv.Spec.ClaimRef and pvc.Spec.ClaimName must be set but mismatched.
// All other conditions return false.
func (c *taskContext) isVolumeAndClaimBindMismatched() bool {
	// consummation is bi-directional
	if c.claim == nil || !c.claimExists || c.claim.Spec.VolumeName == "" || c.volume == nil || c.volume.Spec.ClaimRef == nil {
		return false
	}

	if c.claim.Spec.VolumeName != c.volume.Name ||
		c.claim.UID != c.volume.Spec.ClaimRef.UID {
		return true
	}

	return false
}

// pv.Spec.ClaimRef and pvc.Spec.ClaimName must match. All other conditions return false.
func (c *taskContext) isVolumeBound() bool {
	// consummation is bi-directional
	if c.claim == nil || c.volume == nil || c.volume.Spec.ClaimRef == nil {
		return false
	}
	if c.claim.Spec.VolumeName == c.volume.Name && c.claim.UID == c.volume.Spec.ClaimRef.UID {
		return true
	}
	return false
}

// isValidVolume returns true if the object is in the local store.
// also acts as an initializer for the taskContext.
func (c *taskContext) isValidVolume(key interface{}) bool {
	obj, exists, err := c.host.volumeStore.GetByKey(key.(string))
	if err != nil {
		glog.V(5).Infof("PersistentVolume[%s] not found in local store: %v", key, err)
		return false
	}

	if !exists {
		glog.V(5).Infof("PersistentVolume[%s] has been deleted", key)
		return false
	}
	c.volume = obj.(*api.PersistentVolume)

	if c.volume.Spec.ClaimRef != nil {
		key := fmt.Sprintf("%s/%s", c.volume.Spec.ClaimRef.Namespace, c.volume.Spec.ClaimRef.Name)
		obj, c.claimExists, err = c.host.claimStore.GetByKey(key)
		if c.claimExists {
			c.claim = obj.(*api.PersistentVolumeClaim)
		}
	}

	return true
}

// isValidClaim returns true if the object is in the local store.
// also acts as an initializer for the taskContext.
func (c *taskContext) isValidClaim(key interface{}) bool {
	obj, exists, err := c.host.claimStore.GetByKey(key.(string))
	if err != nil {
		glog.V(5).Infof("PersistentVolumeClaim[%s] not found in local store: %v", key, err)
		return false
	}

	if !exists {
		glog.V(5).Infof("PersistentVolumeClaim[%s] has been deleted", key)
		return false
	}
	c.claim = obj.(*api.PersistentVolumeClaim)

	if c.claim.Spec.VolumeName != "" {
		obj, c.volumeExists, err = c.host.volumeStore.GetByKey(c.claim.Spec.VolumeName)
		if err != nil {
			glog.V(5).Infof("PersistentVolumeClaim[%s] not found in local store: %v", c.claim.Spec.VolumeName, err)
			return false
		}
		if c.volumeExists {
			c.volume = obj.(*api.PersistentVolume)
		}
	}
	return true
}

// provide taskContext with services to the outside world
type taskHost struct {
	client      taskClient
	volumeStore *persistentVolumeOrderedIndex
	claimStore  cache.Store
	queue       *workqueue.Type
	// concurrency not support. lock by volume or claim key for parallelism.
	lock keymutex.KeyMutex
	// required to act as VolumeHost to plugins
	cloud                   cloudprovider.Interface
	pluginMgr               volume.VolumePluginMgr
	volumePluginHost        *volumePluginHost
	volumeConditionHandlers map[api.PersistentVolumeConditionType]func(obj interface{}) error
	claimConditionHandlers  map[api.PersistentVolumeClaimConditionType]func(obj interface{}) error
}

type taskContext struct {
	host         *taskHost
	volume       *api.PersistentVolume
	claim        *api.PersistentVolumeClaim
	claimExists  bool
	volumeExists bool
}

func newTaskContext(host *taskHost) *taskContext {
	ctx := &taskContext{
		host: host,
	}
	ctx.host.volumeConditionHandlers = map[api.PersistentVolumeConditionType]func(obj interface{}) error{
		api.PersistentVolumeBound:       ctx.syncVolumeBoundCondition,
		api.PersistentVolumeRecycled:    ctx.syncVolumeRecycledCondition,
		api.PersistentVolumeProvisioned: ctx.syncVolumeProvisionedCondition,
	}
	ctx.host.claimConditionHandlers = map[api.PersistentVolumeClaimConditionType]func(obj interface{}) error{
		api.PersistentVolumeClaimBound: ctx.syncClaimBoundCondition,
	}
	return ctx
}

func newTaskHost(client taskClient, volumeStore *persistentVolumeOrderedIndex, claimStore cache.Store, queue *workqueue.Type, plugins []volume.VolumePlugin, cloud cloudprovider.Interface) (*taskHost, error) {
	taskHost := &taskHost{
		client:           client,
		volumeStore:      volumeStore,
		claimStore:       claimStore,
		queue:            queue,
		lock:             keymutex.NewKeyMutex(),
		cloud:            cloud,
		volumePluginHost: &volumePluginHost{cloud},
	}

	if err := taskHost.pluginMgr.InitPlugins(plugins, newVolumePluginHost(cloud)); err != nil {
		return nil, fmt.Errorf("Failed to init plugins: %v", err)
	}

	return taskHost, nil
}

// workerTask applies the sync function using the src as the argument
type workerTask struct {
	sync func(obj interface{}) error
	key  interface{}
}

func newWorkerTask(syncFunc func(obj interface{}) error, key interface{}) *workerTask {
	return &workerTask{
		sync: syncFunc,
		key:  key,
	}
}

func worker(queue *workqueue.Type) {
	running := true
	for running {
		func() {
			obj, quit := queue.Get()
			if obj == nil {
				return
			}
			if quit {
				running = false
				return
			}
			defer queue.Done(obj)
			if task, ok := obj.(workerTask); ok {
				fmt.Printf("running\n", task.key)
				err := task.sync(task.key)
				if err != nil {
					glog.Errorf("Error in worker task: %v %v", err, task.key)
					return
				}
			}
		}()
	}
	fmt.Printf("Exiting gracefully")
}

func (c *taskContext) VolumeKey() string {
	if key, err := controller.KeyFunc(c.volume); err == nil {
		return key
	}
	return ""
}

func (c *taskContext) ClaimKey() string {
	if key, err := controller.KeyFunc(c.claim); err == nil {
		return key
	}
	return ""
}

// syncVolume runs a handler func for each pv.Status.Condition
func (c *taskContext) syncVolume(obj interface{}) error {
	if !c.isValidVolume(obj) {
		glog.V(5).Infof("No volume to sync")
		return nil
	}
	for _, condition := range c.volume.Status.Conditions {
		if handler, exists := c.host.volumeConditionHandlers[condition.Type]; exists {
			glog.V(5).Infof("PersistentVolume[%s] syncing condition %s", c.VolumeKey(), condition.Type)
			if err := handler(c.VolumeKey()); err != nil {
				return err
			}
		}
	}
	return nil
}

// syncClaim queues a syncCondition task for each claim.Status.Condition
func (c *taskContext) syncClaim(obj interface{}) error {
	if !c.isValidClaim(obj) {
		glog.V(5).Infof("No claim to sync")
		return nil
	}
	for _, condition := range c.claim.Status.Conditions {
		if handler, exists := c.host.claimConditionHandlers[condition.Type]; exists {
			if err := handler(c.ClaimKey()); err != nil {
				return err
			}
		}
	}

	// changes to the claim should be reflected in the bound volume
	// triggering a sync here ensures updates, if any, are shown before
	// the next sync period.
	// objects where the status matches the spec will no-op in these sync calls
	if !c.isClaimUnbound() && c.isValidVolume(c.VolumeKey()) {
		c.syncVolume(c.VolumeKey())
	}

	return nil
}

// taskClient abstracts access to PVs and PVCs.  Easy to mock for testing and wrap for real client.
type taskClient interface {
	CreatePersistentVolume(pv *api.PersistentVolume) (*api.PersistentVolume, error)
	ListPersistentVolumes(options api.ListOptions) (*api.PersistentVolumeList, error)
	WatchPersistentVolumes(options api.ListOptions) (watch.Interface, error)
	GetPersistentVolume(name string) (*api.PersistentVolume, error)
	UpdatePersistentVolume(volume *api.PersistentVolume) (*api.PersistentVolume, error)
	DeletePersistentVolume(volume *api.PersistentVolume) error
	UpdatePersistentVolumeStatus(volume *api.PersistentVolume) (*api.PersistentVolume, error)

	GetPersistentVolumeClaim(namespace, name string) (*api.PersistentVolumeClaim, error)
	ListPersistentVolumeClaims(namespace string, options api.ListOptions) (*api.PersistentVolumeClaimList, error)
	WatchPersistentVolumeClaims(namespace string, options api.ListOptions) (watch.Interface, error)
	UpdatePersistentVolumeClaim(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error)
	UpdatePersistentVolumeClaimStatus(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error)
}

func newTaskClient(c clientset.Interface) taskClient {
	return &realTaskClient{c}
}

var _ taskClient = &realTaskClient{}

type realTaskClient struct {
	client clientset.Interface
}

func (c *realTaskClient) GetPersistentVolume(name string) (*api.PersistentVolume, error) {
	return c.client.Core().PersistentVolumes().Get(name)
}

func (c *realTaskClient) ListPersistentVolumes(options api.ListOptions) (*api.PersistentVolumeList, error) {
	return c.client.Core().PersistentVolumes().List(options)
}

func (c *realTaskClient) WatchPersistentVolumes(options api.ListOptions) (watch.Interface, error) {
	return c.client.Core().PersistentVolumes().Watch(options)
}

func (c *realTaskClient) CreatePersistentVolume(pv *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.client.Core().PersistentVolumes().Create(pv)
}

func (c *realTaskClient) UpdatePersistentVolume(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.client.Core().PersistentVolumes().Update(volume)
}

func (c *realTaskClient) DeletePersistentVolume(volume *api.PersistentVolume) error {
	return c.client.Core().PersistentVolumes().Delete(volume.Name, nil)
}

func (c *realTaskClient) UpdatePersistentVolumeStatus(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.client.Core().PersistentVolumes().UpdateStatus(volume)
}

func (c *realTaskClient) GetPersistentVolumeClaim(namespace, name string) (*api.PersistentVolumeClaim, error) {
	return c.client.Core().PersistentVolumeClaims(namespace).Get(name)
}

func (c *realTaskClient) ListPersistentVolumeClaims(namespace string, options api.ListOptions) (*api.PersistentVolumeClaimList, error) {
	return c.client.Core().PersistentVolumeClaims(namespace).List(options)
}

func (c *realTaskClient) WatchPersistentVolumeClaims(namespace string, options api.ListOptions) (watch.Interface, error) {
	return c.client.Core().PersistentVolumeClaims(namespace).Watch(options)
}

func (c *realTaskClient) UpdatePersistentVolumeClaim(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	return c.client.Core().PersistentVolumeClaims(claim.Namespace).Update(claim)
}

func (c *realTaskClient) UpdatePersistentVolumeClaimStatus(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	return c.client.Core().PersistentVolumeClaims(claim.Namespace).UpdateStatus(claim)
}

var _ volume.VolumeHost = &volumePluginHost{}

// This controller has to host volume plugins, but does not actually mount any volumes.
// Because no mounting is performed, most of its methods are not implemented.
type volumePluginHost struct {
	cloud cloudprovider.Interface
}

func newVolumePluginHost(cloud cloudprovider.Interface) *volumePluginHost {
	return &volumePluginHost{cloud}
}

func (h *volumePluginHost) GetCloudProvider() cloudprovider.Interface {
	return h.cloud
}

func (h *volumePluginHost) GetPluginDir(podUID string) string {
	return ""
}

func (h *volumePluginHost) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return ""
}

func (h *volumePluginHost) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return ""
}

func (h *volumePluginHost) GetKubeClient() clientset.Interface {
	return nil
}

func (h *volumePluginHost) NewWrapperBuilder(volName string, spec volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Builder, error) {
	return nil, fmt.Errorf("NewWrapperBuilder not supported by PVClaimBinder's VolumeHost implementation")
}

func (h *volumePluginHost) NewWrapperCleaner(volName string, spec volume.Spec, podUID types.UID) (volume.Cleaner, error) {
	return nil, fmt.Errorf("NewWrapperCleaner not supported by PVClaimBinder's VolumeHost implementation")
}

func (h *volumePluginHost) GetMounter() mount.Interface {
	return nil
}

func (h *volumePluginHost) GetWriter() ioutil.Writer {
	return nil
}

func (h *volumePluginHost) GetHostName() string {
	return ""
}
