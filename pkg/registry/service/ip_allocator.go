/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package service

import (
	"fmt"
	math_rand "math/rand"
	"net"
	"sync"
	"time"

	"github.com/golang/glog"
)

type ipAllocator struct {
	lock sync.Mutex // protects 'used'

	subnet         net.IPNet
	ipSpaceSize    int64 // Size of subnet, or -1 if it does not fit in an int64
	used           ipAddrSet
	randomAttempts int

	random *math_rand.Rand
}

type ipAddrSet struct {
	// We are pretty severely restricted in the types of things we can use as a key
	ips map[string]bool
}

func (s *ipAddrSet) Init() {
	s.ips = map[string]bool{}
}

// Gets the number of IPs in the set
func (s *ipAddrSet) Size() int {
	return len(s.ips)
}

// Tests whether the set holds a given IP
func (s *ipAddrSet) Contains(ip net.IP) bool {
	key := ip.String()
	exists := s.ips[key]
	return exists
}

// Adds to the ipAddrSet; returns true iff it was added (was not already in set)
func (s *ipAddrSet) Add(ip net.IP) bool {
	key := ip.String()
	exists := s.ips[key]
	if exists {
		return false
	}
	s.ips[key] = true
	return true
}

// Removes from the ipAddrSet; returns true iff it was removed (was already in set)
func (s *ipAddrSet) Remove(ip net.IP) bool {
	key := ip.String()
	exists := s.ips[key]
	if !exists {
		return false
	}
	delete(s.ips, key)
	// TODO: We probably should add this IP to an 'embargo' list for a limited amount of time

	return true
}

// The smallest number of IPs we accept.
const minIPSpace = 8

// newIPAllocator creates and intializes a new ipAllocator object.
func newIPAllocator(subnet *net.IPNet) *ipAllocator {
	if subnet == nil || subnet.IP == nil || subnet.Mask == nil {
		return nil
	}

	seed := time.Now().UTC().UnixNano()
	r := math_rand.New(math_rand.NewSource(seed))

	ipSpaceSize := int64(-1)
	ones, bits := subnet.Mask.Size()
	if (bits - ones) < 63 {
		ipSpaceSize = int64(1) << uint(bits-ones)

		if ipSpaceSize < minIPSpace {
			glog.Errorf("IPAllocator requires at least %d IPs", minIPSpace)
			return nil
		}
	}

	ipa := &ipAllocator{
		subnet:         *subnet,
		ipSpaceSize:    ipSpaceSize,
		random:         r,
		randomAttempts: 1000,
	}
	ipa.used.Init()

	network := make(net.IP, len(subnet.IP), len(subnet.IP))
	for i := 0; i < len(subnet.IP); i++ {
		network[i] = subnet.IP[i] & subnet.Mask[i]
	}
	ipa.used.Add(network) // block the network addr

	broadcast := make(net.IP, len(subnet.IP), len(subnet.IP))
	for i := 0; i < len(subnet.IP); i++ {
		broadcast[i] = subnet.IP[i] | ^subnet.Mask[i]
	}
	ipa.used.Add(broadcast) // block the broadcast addr

	return ipa
}

// Allocate allocates a specific IP.  This is useful when recovering saved state.
func (ipa *ipAllocator) Allocate(ip net.IP) error {
	ipa.lock.Lock()
	defer ipa.lock.Unlock()

	if !ipa.subnet.Contains(ip) {
		return fmt.Errorf("IP %s does not fall within subnet %s", ip, ipa.subnet)
	}

	if !ipa.used.Add(ip) {
		return fmt.Errorf("IP %s is already allocated", ip)
	}

	return nil
}

// AllocateNext allocates and returns a new IP.
func (ipa *ipAllocator) AllocateNext() (net.IP, error) {
	ipa.lock.Lock()
	defer ipa.lock.Unlock()

	if int64(ipa.used.Size()) == ipa.ipSpaceSize {
		return nil, fmt.Errorf("can't find a free IP in %s", ipa.subnet)
	}

	// Try randomly first
	for i := 0; i < ipa.randomAttempts; i++ {
		ip := ipa.createRandomIp()

		if ipa.used.Add(ip) {
			return ip, nil
		}
	}

	// If that doesn't work, try a linear search
	ip := copyIP(ipa.subnet.IP)
	for ipa.subnet.Contains(ip) {
		ip = ipAdd(ip, 1)
		if ipa.used.Add(ip) {
			return ip, nil
		}
	}

	return nil, fmt.Errorf("can't find a free IP in %s", ipa.subnet)
}

// Returns the index-th IP from the specified subnet range.
// For example, subnet "10.0.0.0/24" with index "2" will return the IP "10.0.0.2".
// TODO(saad-ali): Move this (and any other functions that are independent of ipAllocator) to some
// place more generic.
func GetIndexedIP(subnet *net.IPNet, index int) (net.IP, error) {
	ip := ipAdd(subnet.IP, index /* offset */)
	if !subnet.Contains(ip) {
		return nil, fmt.Errorf("can't generate IP with index %d from subnet. subnet too small. subnet: %q", index, subnet)
	}
	return ip, nil
}

func (ipa *ipAllocator) createRandomIp() net.IP {
	ip := ipa.subnet.IP
	mask := ipa.subnet.Mask
	n := len(ip)

	randomIp := make(net.IP, n, n)

	for i := 0; i < n; i++ {
		if mask[i] == 0xff {
			randomIp[i] = ipa.subnet.IP[i]
		} else {
			b := byte(ipa.random.Intn(256))
			randomIp[i] = (ipa.subnet.IP[i] & mask[i]) | (b &^ mask[i])
		}
	}

	return randomIp
}

// Add an offset to an IP address - used for joining network addr and host addr parts.
func ipAdd(ip net.IP, offset int) net.IP {
	out := copyIP(simplifyIP(ip))
	// Loop from least-significant to most.
	for i := len(out) - 1; i >= 0 && offset > 0; i-- {
		add := offset % 256
		result := int(out[i]) + add
		out[i] = byte(result % 256)
		offset >>= 8
		offset += result / 256 // carry
	}
	return out
}

// Get the optimal slice for an IP. IPv4 addresses will come back in a 4 byte slice. IPv6
// addresses will come back in a 16 byte slice. Non-IP arguments will produce nil.
func simplifyIP(in net.IP) net.IP {
	if ip4 := in.To4(); ip4 != nil {
		return ip4
	}
	return in.To16()
}

// Make a copy of a net.IP.  It appears to be a value type, but it is actually defined as a
// slice, so value assignment is shallow.  Why does a poor dumb user like me need to know
// this sort of implementation detail?
func copyIP(in net.IP) net.IP {
	out := make(net.IP, len(in))
	copy(out, in)
	return out
}

// Release de-allocates an IP.
func (ipa *ipAllocator) Release(ip net.IP) error {
	ipa.lock.Lock()
	defer ipa.lock.Unlock()

	if !ipa.subnet.Contains(ip) {
		return fmt.Errorf("IP %s does not fall within subnet %s", ip, ipa.subnet)
	}
	ipa.used.Remove(ip)
	return nil
}
