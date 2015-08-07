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

package endpoints

import (
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"hash"
	"sort"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
)

// RepackSubsets takes a slice of EndpointSubset objects, expands it to the full
// representation, and then repacks that into the canonical layout.  This
// ensures that code which operates on these objects can rely on the common
// form for things like comparison.  The result is a newly allocated slice.
func RepackSubsets(subsets []api.EndpointSubset) []api.EndpointSubset {
	// First map each unique port definition to the sets of hosts that
	// offer it.  The sets of hosts must be de-duped, using IP+UID as the key.
	type addressKey struct {
		ip  string
		uid types.UID
	}
	allAddrs := map[addressKey]*api.EndpointAddress{}
	portsToAddrs := map[api.EndpointPort]addressSet{}
	for i := range subsets {
		for j := range subsets[i].Ports {
			epp := &subsets[i].Ports[j]
			for k := range subsets[i].Addresses {
				epa := &subsets[i].Addresses[k]
				ak := addressKey{ip: epa.IP}
				if epa.TargetRef != nil {
					ak.uid = epa.TargetRef.UID
				}
				// Accumulate the address.
				if allAddrs[ak] == nil {
					// Make a copy so we don't write to the
					// input args of this function.
					p := &api.EndpointAddress{}
					*p = *epa
					allAddrs[ak] = p
				}
				// Remember that this port maps to this address.
				if _, found := portsToAddrs[*epp]; !found {
					portsToAddrs[*epp] = addressSet{}
				}
				portsToAddrs[*epp].Insert(allAddrs[ak])
			}
		}
	}

	// Next, map the sets of hosts to the sets of ports they offer.
	// Go does not allow maps or slices as keys to maps, so we have
	// to synthesize and artificial key and do a sort of 2-part
	// associative entity.
	type keyString string
	addrSets := map[keyString]addressSet{}
	addrSetsToPorts := map[keyString][]api.EndpointPort{}
	for epp, addrs := range portsToAddrs {
		key := keyString(hashAddresses(addrs))
		addrSets[key] = addrs
		addrSetsToPorts[key] = append(addrSetsToPorts[key], epp)
	}

	// Next, build the N-to-M association the API wants.
	final := []api.EndpointSubset{}
	for key, ports := range addrSetsToPorts {
		addrs := []api.EndpointAddress{}
		for k := range addrSets[key] {
			addrs = append(addrs, *k)
		}
		final = append(final, api.EndpointSubset{Addresses: addrs, Ports: ports})
	}

	// Finally, sort it.
	return SortSubsets(final)
}

type addressSet map[*api.EndpointAddress]struct{}

func (set addressSet) Insert(addr *api.EndpointAddress) {
	set[addr] = struct{}{}
}

func hashAddresses(addrs addressSet) string {
	// Flatten the list of addresses into a string so it can be used as a
	// map key.  Unfortunately, DeepHashObject is implemented in terms of
	// spew, and spew does not handle non-primitive map keys well.  So
	// first we collapse it into a slice, sort the slice, then hash that.
	slice := []*api.EndpointAddress{}
	for k := range addrs {
		slice = append(slice, k)
	}
	sort.Sort(addrPtrsByIpAndUID(slice))
	hasher := md5.New()
	util.DeepHashObject(hasher, slice)
	return hex.EncodeToString(hasher.Sum(nil)[0:])
}

func LessEndpointAddress(a, b *api.EndpointAddress) bool {
	ipComparison := bytes.Compare([]byte(a.IP), []byte(b.IP))
	if ipComparison != 0 {
		return ipComparison < 0
	}
	if b.TargetRef == nil {
		return false
	}
	if a.TargetRef == nil {
		return true
	}
	return a.TargetRef.UID < b.TargetRef.UID
}

type addrPtrsByIpAndUID []*api.EndpointAddress

func (sl addrPtrsByIpAndUID) Len() int      { return len(sl) }
func (sl addrPtrsByIpAndUID) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl addrPtrsByIpAndUID) Less(i, j int) bool {
	return LessEndpointAddress(sl[i], sl[j])
}

// SortSubsets sorts an array of EndpointSubset objects in place.  For ease of
// use it returns the input slice.
func SortSubsets(subsets []api.EndpointSubset) []api.EndpointSubset {
	for i := range subsets {
		ss := &subsets[i]
		sort.Sort(addrsByIpAndUID(ss.Addresses))
		sort.Sort(portsByHash(ss.Ports))
	}
	sort.Sort(subsetsByHash(subsets))
	return subsets
}

func hashObject(hasher hash.Hash, obj interface{}) []byte {
	util.DeepHashObject(hasher, obj)
	return hasher.Sum(nil)
}

type subsetsByHash []api.EndpointSubset

func (sl subsetsByHash) Len() int      { return len(sl) }
func (sl subsetsByHash) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl subsetsByHash) Less(i, j int) bool {
	hasher := md5.New()
	h1 := hashObject(hasher, sl[i])
	h2 := hashObject(hasher, sl[j])
	return bytes.Compare(h1, h2) < 0
}

type addrsByIpAndUID []api.EndpointAddress

func (sl addrsByIpAndUID) Len() int      { return len(sl) }
func (sl addrsByIpAndUID) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl addrsByIpAndUID) Less(i, j int) bool {
	return LessEndpointAddress(&sl[i], &sl[j])
}

type portsByHash []api.EndpointPort

func (sl portsByHash) Len() int      { return len(sl) }
func (sl portsByHash) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl portsByHash) Less(i, j int) bool {
	hasher := md5.New()
	h1 := hashObject(hasher, sl[i])
	h2 := hashObject(hasher, sl[j])
	return bytes.Compare(h1, h2) < 0
}
