/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package pool

import (
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// Repair is a controller loop that periodically examines all allocations
// and logs any errors, and repairs any misallocations.
//
// Handles:
// * Duplicate pool assignments caused by operator action or undetected race conditions
// * Assignments that do not match the current owner network
// * Allocations to owners that were not actually created due to a crash or powerloss
// * Migrates old versions of Kubernetes services into the pool model automatically
//
// Can be run at infrequent intervals, and is best performed on startup of the master.
type Repair struct {
	interval time.Duration
	primary  OwnerRegistry
	pool     PoolAllocator
}

type OwnerRegistry interface {
	ListAllOwnedKeys() ([]PoolOwner, error)
	GetOwnedKeys(ownerId string) (*PoolOwner, uint64, error)
}

type PoolOwner struct {
	Owner string
	Keys  []string
}

// NewRepair creates a controller that periodically ensures that pool resources are uniquely allocated across the cluster
// and generates informational warnings for a cluster that is not in sync.
func NewRepair(interval time.Duration, primary OwnerRegistry, pool PoolAllocator) *Repair {
	return &Repair{
		interval: interval,
		primary:  primary,
		pool:     pool,
	}
}

// RunUntil starts the controller until the provided ch is closed.
func (c *Repair) RunUntil(ch chan struct{}) {
	util.Until(func() {
		if err := c.RunOnce(); err != nil {
			util.HandleError(err)
		}
	}, c.interval, ch)
}

func contains(haystack []string, needle string) bool {
	for i := range haystack {
		if haystack[i] == needle {
			return true
		}
	}
	return false
}

// Performs a repair that is a forced release: the owner has no record of an allocation
func (c *Repair) repairRelease(poolKey string, ownerId string) bool {
	// re-read to ensure we get a linearized view...

	// Get the latest version of the pool to avoid unnecessary alerts
	allocation, t1, err := c.pool.ReadAllocation(poolKey)
	if err != nil {
		util.HandleError(fmt.Errorf("error retrieving updated pool status: %s", ownerId))
		return false
	}
	if allocation == nil {
		// our view of the pool was out of date
		return false
	}

	// Get the latest version of the owner to avoid unnecessary alerts
	poolOwner, t2, err := c.primary.GetOwnedKeys(ownerId)
	if err != nil {
		util.HandleError(fmt.Errorf("error retrieving updated owner: %s", allocation.Owner))
		return false
	}
	if poolOwner != nil && contains(poolOwner.Keys, poolKey) {
		// owner concurrently modified
		return false
	}

	// This established a consistent ordering:
	// The write will fail if allocation.Version is not up-to-date
	// Therefore if the write succeeds, allocation.Version is current
	// poolOwner is at least as new, and we write allocations first,
	// so poolOwner is also up to date
	if t2 < t1 {
		// TODO: Return error?
		glog.Infof("Found pool key %s marked as in-use but not assigned by owner %s, but got non-linear read during fix-attempt; won't fix", poolKey, ownerId)
		return false
	}

	// Problem detected: the key is marked as used in the pool, but not by the owner
	glog.Infof("Releasing pool key %s marked as in-use but not assigned by owner: %s", poolKey, ownerId)

	// We want to release the pool key, but only if it has not been touched since the snapshot.
	// Note that if the owner happened to concurrently add the resource we are now removing,
	// then the version of the pool key allocation will have been bumped.
	done, err := c.pool.ReleaseForRepair(*allocation)
	if err != nil {
		util.HandleError(fmt.Errorf("unable to persist the updated pool repair-release: %v", err))
		return false
	}

	if !done {
		glog.Infof("Concurrent modification on pool key %s during release", poolKey)
	}

	return done
}

// Performs a repair that is a forced allocation: the owner has a record of an allocation, but it isn't marked as used
func (c *Repair) repairAllocate(poolKey string, ownerId string) bool {
	// re-read to ensure we get a linearized view...

	allocation, t1, err := c.pool.ReadAllocation(poolKey)
	if err != nil {
		util.HandleError(fmt.Errorf("error retrieving updated pool status: %s", ownerId))
		return false
	}

	if allocation != nil {
		// our view of the pool was out of date
		return false
	}

	// Get the latest version of the owner to avoid unnecessary alerts
	owner, t2, err := c.primary.GetOwnedKeys(ownerId)
	if err != nil {
		util.HandleError(fmt.Errorf("error retrieving updated owner: %s", ownerId))
		return false
	}
	if owner == nil {
		// owner concurrently deleted
		return false
	}
	if !contains(owner.Keys, poolKey) {
		// owner concurrently modified
		return false
	}

	if t2 < t1 {
		// TODO: Return error?
		glog.Infof("Found unlocked pool resource %s marked assigned in the owner (%s), but got non-linear read during fix-attempt; won't fix", poolKey, ownerId)
		return false
	}

	// Problem detected: the key is marked as used in the owner, but not in the pool
	// This is a big problem if left unfixed, because we could end up double-allocating the resource
	// We don't expect this to actually happen; that would be a bug
	glog.Errorf("Found unlocked pool resource %s marked assigned in the owner (%s)", poolKey, ownerId)

	// It actually isn't a big problem if we accidentally mark something as in-use that isn't; it will get repaired.
	// (the big problem is what we think we have detected, where e.g. a service thinks it owns a port, but doesn't)
	locked, err := c.pool.Allocate(poolKey, ownerId)
	if err != nil {
		util.HandleError(fmt.Errorf("unable to persist the updated pool repair-allocation: %v", err))
		return false
	}

	if !locked {
		// TODO: This might be panic-worthy?
		util.HandleError(fmt.Errorf("the pool resource %s for owner %s was assigned to multiple owners; please recreate", poolKey, ownerId))
	}

	return locked
}

// RunOnce verifies the state of the portal IP allocations and returns an error if an unrecoverable problem occurs.
func (c *Repair) RunOnce() error {
	ownedKeys, err := c.primary.ListAllOwnedKeys()
	if err != nil {
		return fmt.Errorf("pool repair; unable to refresh the primary: %v", err)
	}

	ownedKeyMap := map[string]*PoolOwner{}
	for i := range ownedKeys {
		poolOwner := &ownedKeys[i]
		ownedKeyMap[poolOwner.Owner] = poolOwner
	}

	allocations, err := c.pool.ListAllocations()
	if err != nil {
		return fmt.Errorf("unable to refresh the service IP block; could not retrieve allocations: %v", err)
	}

	// Map of resources marked as in-use
	allocated := map[string]bool{}

	// Look for extra allocations: the resource is marked as owned, but the owner does not think it owns it
	// We have to be really sure before we release the resource, because double-allocation is end-of-the-world
	for _, allocation := range allocations {
		ownerId := allocation.Owner
		poolKey := allocation.Key

		// For the next phase
		allocated[allocation.Key] = true

		poolOwner := ownedKeyMap[ownerId]
		if poolOwner != nil && contains(poolOwner.Keys, poolKey) {
			continue
		}

		c.repairRelease(poolKey, ownerId)
	}

	// Look for missing allocations; the port is marked as in-use by a service, but this is not marked in the pool.
	// We can make mistakes here; failure to release a resource is just going to cause requests to fail that could have
	// succeeded
	for ownerId, poolOwner := range ownedKeyMap {
		for _, poolKey := range poolOwner.Keys {
			if allocated[poolKey] {
				continue
			}

			c.repairAllocate(poolKey, ownerId)
		}
	}

	return nil
}
