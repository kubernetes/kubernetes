/*
Copyright 2020 The Kubernetes Authors.

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

package endpointslicemirroring

import (
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	endpointsliceutil "k8s.io/endpointslice/util"
)

// slicesByAction includes lists of slices to create, update, or delete.
type slicesByAction struct {
	toCreate, toUpdate, toDelete []*discovery.EndpointSlice
}

// append appends slices from another slicesByAction struct.
func (s *slicesByAction) append(slices slicesByAction) {
	s.toCreate = append(s.toCreate, slices.toCreate...)
	s.toUpdate = append(s.toUpdate, slices.toUpdate...)
	s.toDelete = append(s.toDelete, slices.toDelete...)
}

// totalsByAction includes total numbers for added and removed.
type totalsByAction struct {
	added, updated, removed int
}

// add adds totals from another totalsByAction struct.
func (t *totalsByAction) add(totals totalsByAction) {
	t.added += totals.added
	t.updated += totals.updated
	t.removed += totals.removed
}

// newDesiredCalc initializes and returns a new desiredCalc.
func newDesiredCalc() *desiredCalc {
	return &desiredCalc{
		portsByKey:          map[addrTypePortMapKey][]discovery.EndpointPort{},
		endpointsByKey:      map[addrTypePortMapKey]endpointsliceutil.EndpointSet{},
		numDesiredEndpoints: 0,
	}
}

// desiredCalc helps calculate desired endpoints and ports.
type desiredCalc struct {
	portsByKey          map[addrTypePortMapKey][]discovery.EndpointPort
	endpointsByKey      map[addrTypePortMapKey]endpointsliceutil.EndpointSet
	numDesiredEndpoints int
}

// multiAddrTypePortMapKey stores addrTypePortMapKey for different address
// types.
type multiAddrTypePortMapKey map[discovery.AddressType]addrTypePortMapKey

// initPorts initializes ports for a subset and address type and returns the
// corresponding addrTypePortMapKey.
func (d *desiredCalc) initPorts(subsetPorts []v1.EndpointPort) multiAddrTypePortMapKey {
	endpointPorts := epPortsToEpsPorts(subsetPorts)
	addrTypes := []discovery.AddressType{discovery.AddressTypeIPv4, discovery.AddressTypeIPv6}
	multiKey := multiAddrTypePortMapKey{}

	for _, addrType := range addrTypes {
		multiKey[addrType] = newAddrTypePortMapKey(endpointPorts, addrType)
		if _, ok := d.endpointsByKey[multiKey[addrType]]; !ok {
			d.endpointsByKey[multiKey[addrType]] = endpointsliceutil.EndpointSet{}
		}
		d.portsByKey[multiKey[addrType]] = endpointPorts
	}

	return multiKey
}

// addAddress adds an EndpointAddress to the desired state if it is valid. It
// returns false if the address was invalid.
func (d *desiredCalc) addAddress(address v1.EndpointAddress, multiKey multiAddrTypePortMapKey, ready bool) bool {
	endpoint := addressToEndpoint(address, ready)
	addrType := getAddressType(address.IP)
	if addrType == nil {
		return false
	}

	d.endpointsByKey[multiKey[*addrType]].Insert(endpoint)
	d.numDesiredEndpoints++
	return true
}

type slicesByAddrType map[discovery.AddressType][]*discovery.EndpointSlice

// recycleSlices will recycle the slices marked for deletion by replacing
// creates with updates of slices that would otherwise be deleted.
func recycleSlices(slices *slicesByAction) {
	toCreateByAddrType := toSlicesByAddrType(slices.toCreate)
	toDeleteByAddrType := toSlicesByAddrType(slices.toDelete)

	for addrType, slicesToCreate := range toCreateByAddrType {
		slicesToDelete := toDeleteByAddrType[addrType]
		for i := 0; len(slicesToCreate) > i && len(slicesToDelete) > i; i++ {
			slices.toCreate = removeSlice(slices.toCreate, slicesToCreate[i])
			slices.toDelete = removeSlice(slices.toDelete, slicesToDelete[i])
			slice := slicesToCreate[i]
			slice.Name = slicesToDelete[i].Name
			slices.toUpdate = append(slices.toUpdate, slice)
		}
	}
}

// removeSlice removes an EndpointSlice from a list of EndpointSlices.
func removeSlice(slices []*discovery.EndpointSlice, sliceToRemove *discovery.EndpointSlice) []*discovery.EndpointSlice {
	for i, slice := range slices {
		if slice.Name == sliceToRemove.Name {
			return append(slices[:i], slices[i+1:]...)
		}
	}
	return slices
}

// toSliceByAddrType returns lists of EndpointSlices grouped by address.
func toSlicesByAddrType(slices []*discovery.EndpointSlice) slicesByAddrType {
	byAddrType := slicesByAddrType{}
	for _, slice := range slices {
		byAddrType[slice.AddressType] = append(byAddrType[slice.AddressType], slice)
	}
	return byAddrType
}
