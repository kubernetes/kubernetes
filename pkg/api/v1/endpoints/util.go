/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/types"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

const (
	// TODO: to be deleted after v1.3 is released
	// Its value is the json representation of map[string(IP)][HostRecord]
	// example: '{"10.245.1.6":{"HostName":"my-webserver"}}'
	PodHostnamesAnnotation = "endpoints.beta.kubernetes.io/hostnames-map"
)

// TODO: to be deleted after v1.3 is released
type HostRecord struct {
	HostName string
}

// RepackSubsets takes a slice of EndpointSubset objects, expands it to the full
// representation, and then repacks that into the canonical layout.  This
// ensures that code which operates on these objects can rely on the common
// form for things like comparison.  The result is a newly allocated slice.
func RepackSubsets(subsets []v1.EndpointSubset) []v1.EndpointSubset {
	// First map each unique port definition to the sets of hosts that
	// offer it.
	allAddrs := map[addressKey]*v1.EndpointAddress{}
	portToAddrReadyMap := map[v1.EndpointPort]addressSet{}
	for i := range subsets {
		for _, port := range subsets[i].Ports {
			for k := range subsets[i].Addresses {
				mapAddressByPort(&subsets[i].Addresses[k], port, true, allAddrs, portToAddrReadyMap)
			}
			for k := range subsets[i].NotReadyAddresses {
				mapAddressByPort(&subsets[i].NotReadyAddresses[k], port, false, allAddrs, portToAddrReadyMap)
			}
		}
	}

	// Next, map the sets of hosts to the sets of ports they offer.
	// Go does not allow maps or slices as keys to maps, so we have
	// to synthesize an artificial key and do a sort of 2-part
	// associative entity.
	type keyString string
	keyToAddrReadyMap := map[keyString]addressSet{}
	addrReadyMapKeyToPorts := map[keyString][]v1.EndpointPort{}
	for port, addrs := range portToAddrReadyMap {
		key := keyString(hashAddresses(addrs))
		keyToAddrReadyMap[key] = addrs
		addrReadyMapKeyToPorts[key] = append(addrReadyMapKeyToPorts[key], port)
	}

	// Next, build the N-to-M association the API wants.
	final := []v1.EndpointSubset{}
	for key, ports := range addrReadyMapKeyToPorts {
		var readyAddrs, notReadyAddrs []v1.EndpointAddress
		for addr, ready := range keyToAddrReadyMap[key] {
			if ready {
				readyAddrs = append(readyAddrs, *addr)
			} else {
				notReadyAddrs = append(notReadyAddrs, *addr)
			}
		}
		final = append(final, v1.EndpointSubset{Addresses: readyAddrs, NotReadyAddresses: notReadyAddrs, Ports: ports})
	}

	// Finally, sort it.
	return SortSubsets(final)
}

// The sets of hosts must be de-duped, using IP+UID as the key.
type addressKey struct {
	ip  string
	uid types.UID
}

// mapAddressByPort adds an address into a map by its ports, registering the address with a unique pointer, and preserving
// any existing ready state.
func mapAddressByPort(addr *v1.EndpointAddress, port v1.EndpointPort, ready bool, allAddrs map[addressKey]*v1.EndpointAddress, portToAddrReadyMap map[v1.EndpointPort]addressSet) *v1.EndpointAddress {
	// use addressKey to distinguish between two endpoints that are identical addresses
	// but may have come from different hosts, for attribution. For instance, Mesos
	// assigns pods the node IP, but the pods are distinct.
	key := addressKey{ip: addr.IP}
	if addr.TargetRef != nil {
		key.uid = addr.TargetRef.UID
	}

	// Accumulate the address. The full EndpointAddress structure is preserved for use when
	// we rebuild the subsets so that the final TargetRef has all of the necessary data.
	existingAddress := allAddrs[key]
	if existingAddress == nil {
		// Make a copy so we don't write to the
		// input args of this function.
		existingAddress = &v1.EndpointAddress{}
		*existingAddress = *addr
		allAddrs[key] = existingAddress
	}

	// Remember that this port maps to this address.
	if _, found := portToAddrReadyMap[port]; !found {
		portToAddrReadyMap[port] = addressSet{}
	}
	// if we have not yet recorded this port for this address, or if the previous
	// state was ready, write the current ready state. not ready always trumps
	// ready.
	if wasReady, found := portToAddrReadyMap[port][existingAddress]; !found || wasReady {
		portToAddrReadyMap[port][existingAddress] = ready
	}
	return existingAddress
}

type addressSet map[*v1.EndpointAddress]bool

type addrReady struct {
	addr  *v1.EndpointAddress
	ready bool
}

func hashAddresses(addrs addressSet) string {
	// Flatten the list of addresses into a string so it can be used as a
	// map key.  Unfortunately, DeepHashObject is implemented in terms of
	// spew, and spew does not handle non-primitive map keys well.  So
	// first we collapse it into a slice, sort the slice, then hash that.
	slice := make([]addrReady, 0, len(addrs))
	for k, ready := range addrs {
		slice = append(slice, addrReady{k, ready})
	}
	sort.Sort(addrsReady(slice))
	hasher := md5.New()
	hashutil.DeepHashObject(hasher, slice)
	return hex.EncodeToString(hasher.Sum(nil)[0:])
}

func lessAddrReady(a, b addrReady) bool {
	// ready is not significant to hashing since we can't have duplicate addresses
	return LessEndpointAddress(a.addr, b.addr)
}

type addrsReady []addrReady

func (sl addrsReady) Len() int      { return len(sl) }
func (sl addrsReady) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl addrsReady) Less(i, j int) bool {
	return lessAddrReady(sl[i], sl[j])
}

func LessEndpointAddress(a, b *v1.EndpointAddress) bool {
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

type addrPtrsByIpAndUID []*v1.EndpointAddress

func (sl addrPtrsByIpAndUID) Len() int      { return len(sl) }
func (sl addrPtrsByIpAndUID) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl addrPtrsByIpAndUID) Less(i, j int) bool {
	return LessEndpointAddress(sl[i], sl[j])
}

// SortSubsets sorts an array of EndpointSubset objects in place.  For ease of
// use it returns the input slice.
func SortSubsets(subsets []v1.EndpointSubset) []v1.EndpointSubset {
	for i := range subsets {
		ss := &subsets[i]
		sort.Sort(addrsByIpAndUID(ss.Addresses))
		sort.Sort(addrsByIpAndUID(ss.NotReadyAddresses))
		sort.Sort(portsByHash(ss.Ports))
	}
	sort.Sort(subsetsByHash(subsets))
	return subsets
}

func hashObject(hasher hash.Hash, obj interface{}) []byte {
	hashutil.DeepHashObject(hasher, obj)
	return hasher.Sum(nil)
}

type subsetsByHash []v1.EndpointSubset

func (sl subsetsByHash) Len() int      { return len(sl) }
func (sl subsetsByHash) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl subsetsByHash) Less(i, j int) bool {
	hasher := md5.New()
	h1 := hashObject(hasher, sl[i])
	h2 := hashObject(hasher, sl[j])
	return bytes.Compare(h1, h2) < 0
}

type addrsByIpAndUID []v1.EndpointAddress

func (sl addrsByIpAndUID) Len() int      { return len(sl) }
func (sl addrsByIpAndUID) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl addrsByIpAndUID) Less(i, j int) bool {
	return LessEndpointAddress(&sl[i], &sl[j])
}

type portsByHash []v1.EndpointPort

func (sl portsByHash) Len() int      { return len(sl) }
func (sl portsByHash) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl portsByHash) Less(i, j int) bool {
	hasher := md5.New()
	h1 := hashObject(hasher, sl[i])
	h2 := hashObject(hasher, sl[j])
	return bytes.Compare(h1, h2) < 0
}
