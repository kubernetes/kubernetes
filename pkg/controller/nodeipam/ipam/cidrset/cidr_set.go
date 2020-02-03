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
	"math/bits"
	"net"
	"sync"

	"github.com/RoaringBitmap/roaring"
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
	// ErrCIDRAlreadyOccupied occurs when we try to allocate a subnet that is
	// already in use
	ErrCIDRAlreadyOccupied = errors.New(
		"The CIDR can't not be occupied because is already in use")
)

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

// CidrSet implements Interface
var _ Interface = &CidrSet{}

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
		return -1, -1, fmt.Errorf("error getting indices for cluster cidr %v, cidr is nil", s.clusterCIDR)
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
			// ipIntLeft          |         ipIntRight
			// 2001:0DB8:1234:0000:0000:0000:0000:0000
			ipIntLeft := binary.BigEndian.Uint64(cidr.IP[:net.IPv6len/2]) | (^binary.BigEndian.Uint64(cidr.Mask[:net.IPv6len/2]))
			ipIntRight := binary.BigEndian.Uint64(cidr.IP[net.IPv6len/2:]) | (^binary.BigEndian.Uint64(cidr.Mask[net.IPv6len/2:]))
			binary.BigEndian.PutUint64(ip[:net.IPv6len/2], ipIntLeft)
			binary.BigEndian.PutUint64(ip[net.IPv6len/2:], ipIntRight)
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
		bigIP := big.NewInt(0).SetBytes(s.clusterIP)
		bigIP = bigIP.Xor(bigIP, big.NewInt(0).SetBytes(ip))
		cidrIndexBig := bigIP.Rsh(bigIP, uint(net.IPv6len*8-s.subNetMaskSize))
		cidrIndex := cidrIndexBig.Uint64()
		if cidrIndex >= uint64(s.maxCIDRs) {
			return 0, fmt.Errorf("CIDR: %v/%v is out of the range of CIDR allocator", ip, s.subNetMaskSize)
		}
		return int(cidrIndex), nil
	}

	return 0, fmt.Errorf("invalid IP: %v", ip)
}

// A cluster has assigned a ClusterCIDR with mask X that is split into
// different CIDRs assigned to the nodes with mask Y
// The number of nodeCIDRs is equal to 2^(X-Y), i.e.:
// a clusterCIDR 192.168.0.0/16 with nodes subnets of /24 gives 2^8 = 256 node subnets

// cluster mask --->|
// 11000000.10101000.00000000.00000000
//                           |<-- subnetMask
//        bitmap --->|       |<-- offset
//
// base = 192.168.0.0
// offset = 0.0.0.255
// subnet1 = 192.168.0.0/24
// subnet2 = 192.168.1.0/24
// ...
// subnet256 = 192.168.255.0/24

// Using a roaring bitmap to store the subnets allocated give us a maximum of 2^32 subnets
// because it uses an uint32 index (2^32 = 4294967296 subnets.)

// That's perfectly fine for IPv4, but for IPv6 it imposes a restriction:
// The ClusterCIDR mask should be <= 32 than the Nodes masks.
// The IPv6 default mask for a node is /64.

const (
	clusterSubnetRoaringMaxDiff = 32
)

// RoaringCidrSet manages a set of CIDR ranges from which blocks of IPs can
// be allocated from.
// It allocates the IP subnet addresses in a bitmap as an offset of the ClusterCIDR IP subnet
// The masks of the ClusterCIDR and NodeCIDRs are known in advance
type RoaringCidrSet struct {
	sync.Mutex
	clusterCIDR   *net.IPNet
	subnetMask    net.IPMask
	base          *big.Int
	offset        *big.Int
	maxCIDRs      int
	nextCandidate int
	used          *roaring.Bitmap
}

// RoaringCidrSet implements Interface
var _ Interface = &RoaringCidrSet{}

// NewCIDRSetRoaring creates a new CidrSet.
func NewCIDRSetRoaring(clusterCIDR *net.IPNet, subNetMaskSize int) (*RoaringCidrSet, error) {
	clusterMaskSize, maskLen := clusterCIDR.Mask.Size()
	if subNetMaskSize-clusterMaskSize > clusterSubnetRoaringMaxDiff {
		return nil, ErrCIDRSetSubNetTooBig
	}
	// To avoid issues we mask the IP address to ensure we get the subnet address
	clusterCIDR.IP.Mask(clusterCIDR.Mask)
	// Obtain the subnets mask
	subnetMask := net.CIDRMask(subNetMaskSize, maskLen)
	// Calculate the base address of the subnets based on the Cluster IP subnet address
	base := bigForIP(clusterCIDR.IP.Mask(subnetMask))
	// The offset is the size of the subnet, that's the same that 2^(inverse mask size)
	offset := subnetSize(&net.IPNet{IP: nil, Mask: subnetMask})
	// Obtain the max numbers of subnets in the cluster
	maxCIDRs := 1 << uint32(subNetMaskSize-clusterMaskSize)
	return &RoaringCidrSet{
		clusterCIDR: clusterCIDR,
		base:        base,
		offset:      offset,
		maxCIDRs:    maxCIDRs,
		subnetMask:  subnetMask,
		used:        roaring.NewBitmap(),
	}, nil
}

// AllocateNext allocates the next free CIDR range and returns it.
func (s *RoaringCidrSet) AllocateNext() (*net.IPNet, error) {
	s.Lock()
	defer s.Unlock()

	// iterate over all the bitmap to find an unasiggned subnet
	// we start checking from the nextCandidate variable
	for i := 0; i < s.maxCIDRs; i++ {
		candidate := (s.nextCandidate + i) % s.maxCIDRs
		if s.used.CheckedAdd(uint32(candidate)) {
			// set the CIDR range to assign and return the CIDR subnet
			ip := addCIDROffset(s.base, s.offset, candidate)
			s.nextCandidate = (candidate + 1) % s.maxCIDRs
			return &net.IPNet{IP: ip.Mask(s.subnetMask), Mask: s.subnetMask}, nil
		}
	}
	return nil, ErrCIDRRangeNoCIDRsRemaining

}

// Release releases the given CIDR range. It never errors
func (s *RoaringCidrSet) Release(cidr *net.IPNet) error {
	s.Lock()
	defer s.Unlock()
	if err := s.validate(cidr); err != nil {
		return err
	}
	for _, subnet := range overlappingSubnets(cidr, s.subnetMask) {
		// mask the IP address to obtain the corresponding subnet address
		at := calculateCIDROffset(s.base, s.offset, subnet.IP.Mask(s.subnetMask))
		s.used.Remove(uint32(at))
	}
	return nil
}

// Occupy marks the given CIDR range as used. Occupy does check if the CIDR
// range was previously used.
func (s *RoaringCidrSet) Occupy(cidr *net.IPNet) (err error) {
	s.Lock()
	defer s.Unlock()
	if err := s.validate(cidr); err != nil {
		return err
	}
	var positions []int
	for _, subnet := range overlappingSubnets(cidr, s.subnetMask) {
		// do we need to validate again?
		// TODO: check corner cases
		if err := s.validate(subnet); err != nil {
			return err
		}
		// mask the IP address to obtain the corresponding subnet address
		at := calculateCIDROffset(s.base, s.offset, subnet.IP)
		if s.used.ContainsInt(at) {
			return ErrCIDRAlreadyOccupied
		}
		positions = append(positions, at)
	}
	// Do the allocation
	for _, pos := range positions {
		s.used.AddInt(pos)
	}
	return err
}

// validate returns an error if the IP cidr is not valid
// IP not in range or invalid Mask
func (s *RoaringCidrSet) validate(cidr *net.IPNet) error {
	// error if cidr is nil
	if cidr == nil {
		return fmt.Errorf("error getting indices for cluster cidr %v, cidr is nil", s.clusterCIDR)
	}
	// error if cidr does not belong to the cluster range
	if !s.clusterCIDR.Contains(cidr.IP.Mask(s.clusterCIDR.Mask)) && !cidr.Contains(s.clusterCIDR.IP.Mask(cidr.Mask)) {
		return fmt.Errorf("cidr %v is out the range of cluster cidr %v", cidr, s.clusterCIDR)
	}
	return nil
}

// allocated returns the number of CIDR allocated (only for testing)
func (s *RoaringCidrSet) allocated() int {
	return int(s.used.GetCardinality())
}

// bigForIP creates a big.Int based on the provided net.IP
func bigForIP(ip net.IP) *big.Int {
	b := ip.To4()
	if b == nil {
		b = ip.To16()
	}
	return big.NewInt(0).SetBytes(b)
}

// addCIDROffset adds the provided offset multiplied by the integer factor
// to a base big.Int representing a net.IP
func addCIDROffset(base *big.Int, offset *big.Int, factor int) net.IP {
	offset = big.NewInt(1).Mul(offset, big.NewInt(int64(factor)))
	return net.IP(big.NewInt(0).Add(base, offset).Bytes())
}

// calculateCIDROffset calculates the integer offset of ip from base such that
// base + offset = ip. It requires ip >= base.
func calculateCIDROffset(base *big.Int, offset *big.Int, ip net.IP) int {
	cidrOffset := big.NewInt(0).Sub(bigForIP(ip), base)
	return int(big.NewInt(1).Div(cidrOffset, offset).Int64())
}

// nextSubnet returns the next subnet address
func nextSubnet(cidr *net.IPNet) *net.IPNet {
	// Obtain the IP big number
	bigIP := bigForIP(cidr.IP)
	// Obtain the subnet size
	bigMask := subnetSize(cidr)
	// Obtain an IP that belongs to the next subnet
	ip := net.IP(big.NewInt(0).Add(bigIP, bigMask).Bytes())
	return &net.IPNet{IP: ip.Mask(cidr.Mask), Mask: cidr.Mask}
}

// subnetSize returns the size of a given subnet
func subnetSize(cidr *net.IPNet) *big.Int {
	ones, bits := cidr.Mask.Size()
	var i, e = big.NewInt(2), big.NewInt(int64(bits - ones))
	return big.NewInt(1).Exp(i, e, nil)
}

// overlappingSubnets returns a list with the subnets that overlaps
// with the mask given.
// If the mask is shorter it returns the super subnet
// If the mask is larger it returns all the inner subnets
func overlappingSubnets(cidr *net.IPNet, mask net.IPMask) []*net.IPNet {
	var cidrs []*net.IPNet
	cidrMaskSize, cidrMaskLen := cidr.Mask.Size()
	maskSize, maskLen := mask.Size()
	// If the mask length is different we are mixing ip families
	if maskLen != cidrMaskLen {
		return cidrs
	}
	// allocate the first subnet, corresponding to the cidr IP subnet address
	// if the cidr mask is smaller than the mask we return only this subnet
	cidrs = append(cidrs, &net.IPNet{IP: cidr.IP.Mask(cidr.Mask), Mask: mask})
	// allocate the corresponding inner subnets if the mask given is smaller
	if cidrMaskSize < maskSize {
		// use big numbers to handle cases where masks > 32
		one := big.NewInt(1)
		start := big.NewInt(1)
		// obtain the number of overlapping subnets
		var x, e = big.NewInt(2), big.NewInt(int64(maskSize - cidrMaskSize))
		end := big.NewInt(1).Exp(x, e, nil)
		// append each subsequent subnet
		for i := new(big.Int).Set(start); i.Cmp(end) < 0; i.Add(i, one) {
			cidrs = append(cidrs, nextSubnet(cidrs[len(cidrs)-1]))
		}
	}
	return cidrs
}
