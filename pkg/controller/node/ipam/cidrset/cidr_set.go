/*
Copyright 2016 The Kubernetes Authors.

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

package cidrset

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math/big"
	"net"
	"sync"
)

// CidrSet manages a set of CIDR ranges from which blocks of IPs can
// be allocated from.
type CidrSet struct {
	sync.Mutex
	clusterCIDR     *net.IPNet
	clusterIP       net.IP
	clusterMaskSize int
	maxCIDRs        int
	nextCandidate   int
	used            big.Int
	subNetMaskSize  int
}

const (
	// The subnet mask size cannot be greater than 16 more than the cluster mask size
	// TODO: https://github.com/kubernetes/kubernetes/issues/44918
	// clusterSubnetMaxDiff limited to 16 due to the uncompressed bitmap
	clusterSubnetMaxDiff = 16
	// maximum 64 bits of prefix
	maxPrefixLength = 64
)

var (
	// ErrCIDRRangeNoCIDRsRemaining occurs when there are no more space
	// to allocate CIDR ranges.
	ErrCIDRRangeNoCIDRsRemaining = errors.New(
		"CIDR allocation failed; there are no remaining CIDRs left to allocate in the accepted range")
)

// NewCIDRSet creates a new CidrSet.
func NewCIDRSet(clusterCIDR *net.IPNet, subNetMaskSize int) *CidrSet {
	clusterMask := clusterCIDR.Mask
	clusterMaskSize, _ := clusterMask.Size()

	var maxCIDRs int
	if ((clusterCIDR.IP.To4() == nil) && (subNetMaskSize-clusterMaskSize > clusterSubnetMaxDiff)) || (subNetMaskSize > maxPrefixLength) {
		maxCIDRs = 0
	} else {
		maxCIDRs = 1 << uint32(subNetMaskSize-clusterMaskSize)
	}
	return &CidrSet{
		clusterCIDR:     clusterCIDR,
		clusterIP:       clusterCIDR.IP,
		clusterMaskSize: clusterMaskSize,
		maxCIDRs:        maxCIDRs,
		subNetMaskSize:  subNetMaskSize,
	}
}

func (s *CidrSet) indexToCIDRBlock(index int) *net.IPNet {
	var ip []byte
	var mask int
	switch /*v4 or v6*/ {
	case s.clusterIP.To4() != nil:
		{
			j := uint32(index) << uint32(32-s.subNetMaskSize)
			ipInt := (binary.BigEndian.Uint32(s.clusterIP)) | j
			ip = make([]byte, 4)
			binary.BigEndian.PutUint32(ip, ipInt)
			mask = 32

		}
	case s.clusterIP.To16() != nil:
		{
			j := uint64(index) << uint64(64-s.subNetMaskSize)
			ipInt := (binary.BigEndian.Uint64(s.clusterIP)) | j
			ip = make([]byte, 16)
			binary.BigEndian.PutUint64(ip, ipInt)
			mask = 128
		}
	}
	return &net.IPNet{
		IP:   ip,
		Mask: net.CIDRMask(s.subNetMaskSize, mask),
	}
}

// AllocateNext allocates the next free CIDR range. This will set the range
// as occupied and return the allocated range.
func (s *CidrSet) AllocateNext() (*net.IPNet, error) {
	s.Lock()
	defer s.Unlock()

	nextUnused := -1
	for i := 0; i < s.maxCIDRs; i++ {
		candidate := (i + s.nextCandidate) % s.maxCIDRs
		if s.used.Bit(candidate) == 0 {
			nextUnused = candidate
			break
		}
	}
	if nextUnused == -1 {
		return nil, ErrCIDRRangeNoCIDRsRemaining
	}
	s.nextCandidate = (nextUnused + 1) % s.maxCIDRs

	s.used.SetBit(&s.used, nextUnused, 1)

	return s.indexToCIDRBlock(nextUnused), nil
}

func (s *CidrSet) getBeginingAndEndIndices(cidr *net.IPNet) (begin, end int, err error) {
	begin, end = 0, s.maxCIDRs-1
	cidrMask := cidr.Mask
	maskSize, _ := cidrMask.Size()
	var ipSize int

	if cidr == nil {
		return -1, -1, fmt.Errorf("Error getting indices for cluster cidr %v, cidr is nil", s.clusterCIDR)
	}

	if !s.clusterCIDR.Contains(cidr.IP.Mask(s.clusterCIDR.Mask)) && !cidr.Contains(s.clusterCIDR.IP.Mask(cidr.Mask)) {
		return -1, -1, fmt.Errorf("cidr %v is out the range of cluster cidr %v", cidr, s.clusterCIDR)
	}

	if s.clusterMaskSize < maskSize {

		ipSize = net.IPv4len
		if cidr.IP.To4() == nil {
			ipSize = net.IPv6len
		}
		subNetMask := net.CIDRMask(s.subNetMaskSize, ipSize*8)
		begin, err = s.getIndexForCIDR(&net.IPNet{
			IP:   cidr.IP.Mask(subNetMask),
			Mask: subNetMask,
		})
		if err != nil {
			return -1, -1, err
		}
		ip := make([]byte, ipSize)
		if cidr.IP.To4() != nil {
			ipInt := binary.BigEndian.Uint32(cidr.IP) | (^binary.BigEndian.Uint32(cidr.Mask))
			binary.BigEndian.PutUint32(ip, ipInt)
		} else {
			ipInt := binary.BigEndian.Uint64(cidr.IP) | (^binary.BigEndian.Uint64(cidr.Mask))
			binary.BigEndian.PutUint64(ip, ipInt)
		}
		end, err = s.getIndexForCIDR(&net.IPNet{
			IP:   net.IP(ip).Mask(subNetMask),
			Mask: subNetMask,
		})
		if err != nil {
			return -1, -1, err
		}
	}
	return begin, end, nil
}

// Release releases the given CIDR range.
func (s *CidrSet) Release(cidr *net.IPNet) error {
	begin, end, err := s.getBeginingAndEndIndices(cidr)
	if err != nil {
		return err
	}
	s.Lock()
	defer s.Unlock()
	for i := begin; i <= end; i++ {
		s.used.SetBit(&s.used, i, 0)
	}
	return nil
}

// Occupy marks the given CIDR range as used. Occupy does not check if the CIDR
// range was previously used.
func (s *CidrSet) Occupy(cidr *net.IPNet) (err error) {
	begin, end, err := s.getBeginingAndEndIndices(cidr)
	if err != nil {
		return err
	}

	s.Lock()
	defer s.Unlock()
	for i := begin; i <= end; i++ {
		s.used.SetBit(&s.used, i, 1)
	}

	return nil
}

func (s *CidrSet) getIndexForCIDR(cidr *net.IPNet) (int, error) {
	return s.getIndexForIP(cidr.IP)
}

func (s *CidrSet) getIndexForIP(ip net.IP) (int, error) {
	if ip.To4() != nil {
		cidrIndex := (binary.BigEndian.Uint32(s.clusterIP) ^ binary.BigEndian.Uint32(ip.To4())) >> uint32(32-s.subNetMaskSize)
		if cidrIndex >= uint32(s.maxCIDRs) {
			return 0, fmt.Errorf("CIDR: %v/%v is out of the range of CIDR allocator", ip, s.subNetMaskSize)
		}
		return int(cidrIndex), nil
	}
	if ip.To16() != nil {
		cidrIndex := (binary.BigEndian.Uint64(s.clusterIP) ^ binary.BigEndian.Uint64(ip.To16())) >> uint64(64-s.subNetMaskSize)
		if cidrIndex >= uint64(s.maxCIDRs) {
			return 0, fmt.Errorf("CIDR: %v/%v is out of the range of CIDR allocator", ip, s.subNetMaskSize)
		}
		return int(cidrIndex), nil
	}

	return 0, fmt.Errorf("invalid IP: %v", ip)
}
