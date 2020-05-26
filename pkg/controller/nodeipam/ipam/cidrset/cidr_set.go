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
	"go.etcd.io/etcd/pkg/adt"
	"math/big"
	"math/bits"
	"net"
	"strconv"
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
	used            adt.IntervalTree
	subNetMaskSize  int
}

const (
	// The subnet mask size cannot be greater than 16 more than the cluster mask size
	// TODO: https://github.com/kubernetes/kubernetes/issues/44918
	// clusterSubnetMaxDiff limited to 16 due to the uncompressed bitmap
	// Due to this limitation the subnet mask for IPv6 cluster cidr needs to be >= 48
	// as default mask size for IPv6 is 64.
	clusterSubnetMaxDiff = 16
	// halfIPv6Len is the half of the IPv6 length
	halfIPv6Len = net.IPv6len / 2
)

var (
	// ErrCIDRRangeNoCIDRsRemaining occurs when there is no more space
	// to allocate CIDR ranges.
	ErrCIDRRangeNoCIDRsRemaining = errors.New(
		"CIDR allocation failed; there are no remaining CIDRs left to allocate in the accepted range")
	// ErrCIDRSetSubNetTooBig occurs when the subnet mask size is too
	// big compared to the CIDR mask size.
	ErrCIDRSetSubNetTooBig = errors.New(
		"New CIDR set failed; the node CIDR size is too big")
	// ErrCIDROverlap occurs when Occupy CIDR but range was previously used
	ErrCIDROverlap = errors.New(
		"Occupy CIDR failed, range was previously used")
)

// NewCIDRSet creates a new CidrSet.
func NewCIDRSet(clusterCIDR *net.IPNet, subNetMaskSize int) (*CidrSet, error) {
	clusterMask := clusterCIDR.Mask
	clusterMaskSize, _ := clusterMask.Size()

	var maxCIDRs int
	if (clusterCIDR.IP.To4() == nil) && (subNetMaskSize-clusterMaskSize > clusterSubnetMaxDiff) {
		return nil, ErrCIDRSetSubNetTooBig
	}
	maxCIDRs = 1 << uint32(subNetMaskSize-clusterMaskSize)
	return &CidrSet{
		clusterCIDR:     clusterCIDR,
		clusterIP:       clusterCIDR.IP,
		clusterMaskSize: clusterMaskSize,
		maxCIDRs:        maxCIDRs,
		subNetMaskSize:  subNetMaskSize,
		used:            adt.NewIntervalTree(),
	}, nil
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
			// leftClusterIP      |     rightClusterIP
			// 2001:0DB8:1234:0000:0000:0000:0000:0000
			const v6NBits = 128
			const halfV6NBits = v6NBits / 2
			leftClusterIP := binary.BigEndian.Uint64(s.clusterIP[:halfIPv6Len])
			rightClusterIP := binary.BigEndian.Uint64(s.clusterIP[halfIPv6Len:])

			leftIP, rightIP := make([]byte, halfIPv6Len), make([]byte, halfIPv6Len)

			if s.subNetMaskSize <= halfV6NBits {
				// We only care about left side IP
				leftClusterIP |= uint64(index) << uint(halfV6NBits-s.subNetMaskSize)
			} else {
				if s.clusterMaskSize < halfV6NBits {
					// see how many bits are needed to reach the left side
					btl := uint(s.subNetMaskSize - halfV6NBits)
					indexMaxBit := uint(64 - bits.LeadingZeros64(uint64(index)))
					if indexMaxBit > btl {
						leftClusterIP |= uint64(index) >> btl
					}
				}
				// the right side will be calculated the same way either the
				// subNetMaskSize affects both left and right sides
				rightClusterIP |= uint64(index) << uint(v6NBits-s.subNetMaskSize)
			}
			binary.BigEndian.PutUint64(leftIP, leftClusterIP)
			binary.BigEndian.PutUint64(rightIP, rightClusterIP)

			ip = append(leftIP, rightIP...)
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
	var cidr *net.IPNet
	for i := 0; i < s.maxCIDRs; i++ {
		candidate := (i + s.nextCandidate) % s.maxCIDRs
		cidr = s.indexToCIDRBlock(candidate)
		interval, err := s.cidrToInterval(cidr)
		if err != nil {
			return nil, err
		}
		if !s.used.Intersects(interval) {
			s.used.Insert(interval, struct{}{})
			s.nextCandidate = (candidate + 1) % s.maxCIDRs
			return cidr, nil
		}
	}
	return nil, ErrCIDRRangeNoCIDRsRemaining
}

func (s *CidrSet) cidrToInterval(cidr *net.IPNet) (adt.Interval, error) {
	if cidr == nil {
		return adt.Interval{}, errors.New("cidr is nil")
	}
	if !s.clusterCIDR.Contains(cidr.IP.Mask(s.clusterCIDR.Mask)) {
		return adt.Interval{}, fmt.Errorf("cidr %v is out the range of cluster cidr %v", cidr, s.clusterCIDR)
	}
	begin := big.NewInt(0)
	end := big.NewInt(1)
	maskSize, length := cidr.Mask.Size()
	if s.clusterMaskSize < maskSize {
		if cidr.IP.To4() != nil {
			begin = begin.SetBytes(cidr.IP.To4())
		} else {
			begin = begin.SetBytes(cidr.IP.To16())
		}
		ones, bits := cidr.Mask.Size()
		end.Lsh(end, uint(bits-ones)).Add(begin, end)
	} else {
		if s.clusterCIDR.IP.To4() != nil {
			begin = begin.SetBytes(s.clusterCIDR.IP.To4())
		} else {
			begin = begin.SetBytes(s.clusterCIDR.IP.To16())
		}
		ones, bits := s.clusterCIDR.Mask.Size()
		end.Lsh(end, uint(bits-ones)).Add(begin, end)
	}
	return adt.NewStringInterval(
		fmt.Sprintf("%0"+strconv.Itoa(length)+"s", begin),
		fmt.Sprintf("%0"+strconv.Itoa(length)+"s", end),
	), nil
}

// Release releases the given CIDR range.
func (s *CidrSet) Release(cidr *net.IPNet) error {
	interval, err := s.cidrToInterval(cidr)
	if err != nil {
		return err
	}
	s.Lock()
	defer s.Unlock()
	s.used.Delete(interval)
	return nil
}

// Occupy marks the given CIDR range as used.
func (s *CidrSet) Occupy(cidr *net.IPNet) (err error) {
	interval, err := s.cidrToInterval(cidr)
	if err != nil {
		return err
	}

	s.Lock()
	defer s.Unlock()
	if s.used.Intersects(interval) {
		return ErrCIDROverlap
	} else {
		s.used.Insert(interval, struct{}{})
	}

	return nil
}
