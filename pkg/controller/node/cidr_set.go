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

package node

import (
	"encoding/binary"
	"fmt"
	"math/big"
	"net"
	"sync"
)

type cidrSet struct {
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

func newCIDRSet(clusterCIDR *net.IPNet, subNetMaskSize int) *cidrSet {
	clusterMask := clusterCIDR.Mask
	clusterMaskSize, _ := clusterMask.Size()

	var maxCIDRs int
	if ((clusterCIDR.IP.To4() == nil) && (subNetMaskSize-clusterMaskSize > clusterSubnetMaxDiff)) || (subNetMaskSize > maxPrefixLength) {
		maxCIDRs = 0
	} else {
		maxCIDRs = 1 << uint32(subNetMaskSize-clusterMaskSize)
	}
	return &cidrSet{
		clusterCIDR:     clusterCIDR,
		clusterIP:       clusterCIDR.IP,
		clusterMaskSize: clusterMaskSize,
		maxCIDRs:        maxCIDRs,
		subNetMaskSize:  subNetMaskSize,
	}
}

func (s *cidrSet) allocateNext() (*net.IPNet, error) {
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
		return nil, errCIDRRangeNoCIDRsRemaining
	}
	s.nextCandidate = (nextUnused + 1) % s.maxCIDRs

	s.used.SetBit(&s.used, nextUnused, 1)

	var ip []byte
	var mask int

	switch /*v4 or v6*/ {
	case s.clusterIP.To4() != nil:
		{
			j := uint32(nextUnused) << uint32(32-s.subNetMaskSize)
			ipInt := (binary.BigEndian.Uint32(s.clusterIP)) | j
			ip = make([]byte, 4)
			binary.BigEndian.PutUint32(ip, ipInt)
			mask = 32

		}
	case s.clusterIP.To16() != nil:
		{
			j := uint64(nextUnused) << uint64(64-s.subNetMaskSize)
			ipInt := (binary.BigEndian.Uint64(s.clusterIP)) | j
			ip = make([]byte, 16)
			binary.BigEndian.PutUint64(ip, ipInt)
			mask = 128
		}
	}
	return &net.IPNet{
		IP:   ip,
		Mask: net.CIDRMask(s.subNetMaskSize, mask),
	}, nil
}

func (s *cidrSet) getBeginingAndEndIndices(cidr *net.IPNet) (begin, end int, err error) {
	begin, end = 0, s.maxCIDRs-1
	cidrMask := cidr.Mask
	maskSize, _ := cidrMask.Size()

	if !s.clusterCIDR.Contains(cidr.IP.Mask(s.clusterCIDR.Mask)) && !cidr.Contains(s.clusterCIDR.IP.Mask(cidr.Mask)) {
		return -1, -1, fmt.Errorf("cidr %v is out the range of cluster cidr %v", cidr, s.clusterCIDR)
	}

	if s.clusterMaskSize < maskSize {
		if cidr.IP.To4() != nil {
			subNetMask := net.CIDRMask(s.subNetMaskSize, 32)
			begin, err = s.getIndexForCIDR(&net.IPNet{
				IP:   cidr.IP.Mask(subNetMask),
				Mask: subNetMask,
			})
			if err != nil {
				return -1, -1, err
			}

			ip := make([]byte, 4)
			ipInt := binary.BigEndian.Uint32(cidr.IP) | (^binary.BigEndian.Uint32(cidr.Mask))
			binary.BigEndian.PutUint32(ip, ipInt)
			end, err = s.getIndexForCIDR(&net.IPNet{
				IP:   net.IP(ip).Mask(subNetMask),
				Mask: subNetMask,
			})
			if err != nil {
				return -1, -1, err
			}
		} else if cidr.IP.To16() != nil {
			subNetMask := net.CIDRMask(s.subNetMaskSize, 128)
			begin, err = s.getIndexForCIDR(&net.IPNet{
				IP:   cidr.IP.Mask(subNetMask),
				Mask: subNetMask,
			})
			if err != nil {
				return -1, -1, err
			}
			ip := make([]byte, 16)
			ipInt := binary.BigEndian.Uint64(cidr.IP) | (^binary.BigEndian.Uint64(cidr.Mask))
			binary.BigEndian.PutUint64(ip, ipInt)
			end, err = s.getIndexForCIDR(&net.IPNet{
				IP:   net.IP(ip).Mask(subNetMask),
				Mask: subNetMask,
			})
			if err != nil {
				return -1, -1, err
			}
		}
	}
	return begin, end, nil
}

func (s *cidrSet) release(cidr *net.IPNet) error {
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

func (s *cidrSet) occupy(cidr *net.IPNet) (err error) {
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

func (s *cidrSet) getIndexForCIDR(cidr *net.IPNet) (int, error) {
	var cidrIndex uint32
	if cidr.IP.To4() != nil {
		cidrIndex = (binary.BigEndian.Uint32(s.clusterIP) ^ binary.BigEndian.Uint32(cidr.IP.To4())) >> uint32(32-s.subNetMaskSize)
		if cidrIndex >= uint32(s.maxCIDRs) {
			return 0, fmt.Errorf("CIDR: %v is out of the range of CIDR allocator", cidr)
		}
	} else if cidr.IP.To16() != nil {
		cidrIndex64 := (binary.BigEndian.Uint64(s.clusterIP) ^ binary.BigEndian.Uint64(cidr.IP.To16())) >> uint64(64-s.subNetMaskSize)

		if cidrIndex64 >= uint64(s.maxCIDRs) {
			return 0, fmt.Errorf("CIDR: %v is out of the range of CIDR allocator", cidr)
		}
		cidrIndex = uint32(cidrIndex64)

	}
	return int(cidrIndex), nil
}
