/*
 *
 * Copyright 2021 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package resolver

type addressMapEntry struct {
	addr  Address
	value any
}

// AddressMap is a map of addresses to arbitrary values taking into account
// Attributes.  BalancerAttributes are ignored, as are Metadata and Type.
// Multiple accesses may not be performed concurrently.  Must be created via
// NewAddressMap; do not construct directly.
type AddressMap struct {
	// The underlying map is keyed by an Address with fields that we don't care
	// about being set to their zero values. The only fields that we care about
	// are `Addr`, `ServerName` and `Attributes`. Since we need to be able to
	// distinguish between addresses with same `Addr` and `ServerName`, but
	// different `Attributes`, we cannot store the `Attributes` in the map key.
	//
	// The comparison operation for structs work as follows:
	//  Struct values are comparable if all their fields are comparable. Two
	//  struct values are equal if their corresponding non-blank fields are equal.
	//
	// The value type of the map contains a slice of addresses which match the key
	// in their `Addr` and `ServerName` fields and contain the corresponding value
	// associated with them.
	m map[Address]addressMapEntryList
}

func toMapKey(addr *Address) Address {
	return Address{Addr: addr.Addr, ServerName: addr.ServerName}
}

type addressMapEntryList []*addressMapEntry

// NewAddressMap creates a new AddressMap.
func NewAddressMap() *AddressMap {
	return &AddressMap{m: make(map[Address]addressMapEntryList)}
}

// find returns the index of addr in the addressMapEntry slice, or -1 if not
// present.
func (l addressMapEntryList) find(addr Address) int {
	for i, entry := range l {
		// Attributes are the only thing to match on here, since `Addr` and
		// `ServerName` are already equal.
		if entry.addr.Attributes.Equal(addr.Attributes) {
			return i
		}
	}
	return -1
}

// Get returns the value for the address in the map, if present.
func (a *AddressMap) Get(addr Address) (value any, ok bool) {
	addrKey := toMapKey(&addr)
	entryList := a.m[addrKey]
	if entry := entryList.find(addr); entry != -1 {
		return entryList[entry].value, true
	}
	return nil, false
}

// Set updates or adds the value to the address in the map.
func (a *AddressMap) Set(addr Address, value any) {
	addrKey := toMapKey(&addr)
	entryList := a.m[addrKey]
	if entry := entryList.find(addr); entry != -1 {
		entryList[entry].value = value
		return
	}
	a.m[addrKey] = append(entryList, &addressMapEntry{addr: addr, value: value})
}

// Delete removes addr from the map.
func (a *AddressMap) Delete(addr Address) {
	addrKey := toMapKey(&addr)
	entryList := a.m[addrKey]
	entry := entryList.find(addr)
	if entry == -1 {
		return
	}
	if len(entryList) == 1 {
		entryList = nil
	} else {
		copy(entryList[entry:], entryList[entry+1:])
		entryList = entryList[:len(entryList)-1]
	}
	a.m[addrKey] = entryList
}

// Len returns the number of entries in the map.
func (a *AddressMap) Len() int {
	ret := 0
	for _, entryList := range a.m {
		ret += len(entryList)
	}
	return ret
}

// Keys returns a slice of all current map keys.
func (a *AddressMap) Keys() []Address {
	ret := make([]Address, 0, a.Len())
	for _, entryList := range a.m {
		for _, entry := range entryList {
			ret = append(ret, entry.addr)
		}
	}
	return ret
}

// Values returns a slice of all current map values.
func (a *AddressMap) Values() []any {
	ret := make([]any, 0, a.Len())
	for _, entryList := range a.m {
		for _, entry := range entryList {
			ret = append(ret, entry.value)
		}
	}
	return ret
}

type endpointNode struct {
	addrs map[string]struct{}
}

// Equal returns whether the unordered set of addrs are the same between the
// endpoint nodes.
func (en *endpointNode) Equal(en2 *endpointNode) bool {
	if len(en.addrs) != len(en2.addrs) {
		return false
	}
	for addr := range en.addrs {
		if _, ok := en2.addrs[addr]; !ok {
			return false
		}
	}
	return true
}

func toEndpointNode(endpoint Endpoint) endpointNode {
	en := make(map[string]struct{})
	for _, addr := range endpoint.Addresses {
		en[addr.Addr] = struct{}{}
	}
	return endpointNode{
		addrs: en,
	}
}

// EndpointMap is a map of endpoints to arbitrary values keyed on only the
// unordered set of address strings within an endpoint. This map is not thread
// safe, thus it is unsafe to access concurrently. Must be created via
// NewEndpointMap; do not construct directly.
type EndpointMap struct {
	endpoints map[*endpointNode]any
}

// NewEndpointMap creates a new EndpointMap.
func NewEndpointMap() *EndpointMap {
	return &EndpointMap{
		endpoints: make(map[*endpointNode]any),
	}
}

// Get returns the value for the address in the map, if present.
func (em *EndpointMap) Get(e Endpoint) (value any, ok bool) {
	en := toEndpointNode(e)
	if endpoint := em.find(en); endpoint != nil {
		return em.endpoints[endpoint], true
	}
	return nil, false
}

// Set updates or adds the value to the address in the map.
func (em *EndpointMap) Set(e Endpoint, value any) {
	en := toEndpointNode(e)
	if endpoint := em.find(en); endpoint != nil {
		em.endpoints[endpoint] = value
		return
	}
	em.endpoints[&en] = value
}

// Len returns the number of entries in the map.
func (em *EndpointMap) Len() int {
	return len(em.endpoints)
}

// Keys returns a slice of all current map keys, as endpoints specifying the
// addresses present in the endpoint keys, in which uniqueness is determined by
// the unordered set of addresses. Thus, endpoint information returned is not
// the full endpoint data (drops duplicated addresses and attributes) but can be
// used for EndpointMap accesses.
func (em *EndpointMap) Keys() []Endpoint {
	ret := make([]Endpoint, 0, len(em.endpoints))
	for en := range em.endpoints {
		var endpoint Endpoint
		for addr := range en.addrs {
			endpoint.Addresses = append(endpoint.Addresses, Address{Addr: addr})
		}
		ret = append(ret, endpoint)
	}
	return ret
}

// Values returns a slice of all current map values.
func (em *EndpointMap) Values() []any {
	ret := make([]any, 0, len(em.endpoints))
	for _, val := range em.endpoints {
		ret = append(ret, val)
	}
	return ret
}

// find returns a pointer to the endpoint node in em if the endpoint node is
// already present. If not found, nil is returned. The comparisons are done on
// the unordered set of addresses within an endpoint.
func (em EndpointMap) find(e endpointNode) *endpointNode {
	for endpoint := range em.endpoints {
		if e.Equal(endpoint) {
			return endpoint
		}
	}
	return nil
}

// Delete removes the specified endpoint from the map.
func (em *EndpointMap) Delete(e Endpoint) {
	en := toEndpointNode(e)
	if entry := em.find(en); entry != nil {
		delete(em.endpoints, entry)
	}
}
