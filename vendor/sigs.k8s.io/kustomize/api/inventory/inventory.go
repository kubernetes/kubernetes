// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package inventory

import (
	"encoding/json"

	"sigs.k8s.io/kustomize/api/resid"
)

//Refs is a reference map.  Each key is the id
//of a k8s resource, and each value is a list of
//object ids that refer back to the object in the
//key.

//For example, the key could correspond to a
//ConfigMap, and the list of values might include
//several different Deployments that get data from
//that ConfigMap (and thus refer to it).

//References are important in inventory management
//because one may not delete an object before all
//objects referencing it have been removed.
type Refs map[resid.ResId][]resid.ResId

func NewRefs() Refs {
	return Refs{}
}

// Merge merges a Refs into an existing Refs
func (rf Refs) Merge(b Refs) Refs {
	for key, value := range b {
		_, ok := rf[key]
		if ok {
			rf[key] = append(rf[key], value...)
		} else {
			rf[key] = value
		}
	}
	return rf
}

// removeIfContains removes the reference relationship
//  a --> b
// from the Refs if it exists
func (rf Refs) RemoveIfContains(a, b resid.ResId) {
	refs, ok := rf[a]
	if !ok {
		return
	}
	for i, ref := range refs {
		if ref.Equals(b) {
			rf[a] = append(refs[:i], refs[i+1:]...)
			break
		}
	}
}

//Inventory is a an object intended for
//serialization into the annotations of a so-called
//apply-root object (a ConfigMap, an Application,
//etc.) living in the cluster.  This apply-root
//object is written as part of an apply operation as
//a means to record overall cluster state changes.

//At the end of a successful apply, the "current"
//field in Inventory will be a map whose keys all
//correspond to an object in the cluster, and
//"previous" will be the previous such set (an empty
//set on the first apply).

//An Inventory allows the Prune method to work.
type Inventory struct {
	Current  Refs `json:"current,omitempty"`
	Previous Refs `json:"previous,omitempty"`
}

// NewInventory returns an Inventory object
func NewInventory() *Inventory {
	return &Inventory{
		Current:  NewRefs(),
		Previous: NewRefs(),
	}
}

// UpdateCurrent updates the Inventory given a
// new current Refs
// The existing Current refs is merged into
// the Previous refs
func (a *Inventory) UpdateCurrent(curref Refs) *Inventory {
	if len(a.Previous) > 0 {
		a.Previous.Merge(a.Current)
	} else {
		a.Previous = a.Current
	}
	a.Current = curref
	return a
}

func (a *Inventory) removeNewlyOrphanedItemsFromPrevious() []resid.ResId {
	var results []resid.ResId
	for item, refs := range a.Previous {
		if _, ok := a.Current[item]; ok {
			delete(a.Previous, item)
			continue
		}

		var newRefs []resid.ResId
		toDelete := true
		for _, ref := range refs {
			if _, ok := a.Current[ref]; ok {
				toDelete = false
				newRefs = append(newRefs, ref)
			}
		}
		if toDelete {
			results = append(results, item)
			delete(a.Previous, item)
		} else {
			a.Previous[item] = newRefs
		}
	}
	return results
}

func (a *Inventory) removeOrphanedItemsFromPreviousThatAreNotInCurrent() []resid.ResId {
	var results []resid.ResId
	for item, refs := range a.Previous {
		if _, ok := a.Current[item]; ok {
			continue
		}
		if len(refs) == 0 {
			results = append(results, item)
			delete(a.Previous, item)
		}
	}
	return results
}

func (a *Inventory) removeOrphanedItemsFromPreviousThatAreInCurrent() {
	//Remove references from Previous that are already in Current refs
	for item, refs := range a.Current {
		for _, ref := range refs {
			a.Previous.RemoveIfContains(item, ref)
		}
	}
	//Remove items from Previous that are already in Current refs
	for item, refs := range a.Previous {
		if len(refs) == 0 {
			if _, ok := a.Current[item]; ok {
				delete(a.Previous, item)
			}
		}
	}
}

// Prune computes the diff of Current refs and Previous refs
// and returns a list of Items that can be pruned.
// An item that can be pruned shows up only in Previous refs.
// Prune also updates the Previous refs with those items removed
func (a *Inventory) Prune() []resid.ResId {
	a.removeOrphanedItemsFromPreviousThatAreInCurrent()

	// These are candidates for deletion from the cluster.
	removable1 := a.removeOrphanedItemsFromPreviousThatAreNotInCurrent()
	removable2 := a.removeNewlyOrphanedItemsFromPrevious()
	return append(removable1, removable2...)
}

// inventory is the internal type used for serialization
type inventory struct {
	Current  map[string][]resid.ResId `json:"current,omitempty"`
	Previous map[string][]resid.ResId `json:"previous,omitempty"`
}

func (a *Inventory) toInternalType() inventory {
	prev := map[string][]resid.ResId{}
	curr := map[string][]resid.ResId{}
	for id, refs := range a.Current {
		curr[id.String()] = refs
	}
	for id, refs := range a.Previous {
		prev[id.String()] = refs
	}
	return inventory{
		Current:  curr,
		Previous: prev,
	}
}

func (a *Inventory) fromInternalType(i *inventory) {
	for s, refs := range i.Previous {
		a.Previous[resid.FromString(s)] = refs
	}
	for s, refs := range i.Current {
		a.Current[resid.FromString(s)] = refs
	}
}

func (a *Inventory) marshal() ([]byte, error) {
	return json.Marshal(a.toInternalType())
}

func (a *Inventory) unMarshal(data []byte) error {
	inv := &inventory{
		Current:  map[string][]resid.ResId{},
		Previous: map[string][]resid.ResId{},
	}
	err := json.Unmarshal(data, inv)
	if err != nil {
		return err
	}
	a.fromInternalType(inv)
	return nil
}

// UpdateAnnotations update the annotation map
func (a *Inventory) UpdateAnnotations(annot map[string]string) error {
	data, err := a.marshal()
	if err != nil {
		return err
	}
	annot[ContentAnnotation] = string(data)
	return nil
}

// LoadFromAnnotation loads the Inventory date from the annotation map
func (a *Inventory) LoadFromAnnotation(annot map[string]string) error {
	value, ok := annot[ContentAnnotation]
	if ok {
		return a.unMarshal([]byte(value))
	}
	return nil
}
