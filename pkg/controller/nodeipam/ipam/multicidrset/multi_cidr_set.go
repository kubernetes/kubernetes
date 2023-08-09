/*
Copyright 2022 The Kubernetes Authors.

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

package multicidrset

import (
	"encoding/binary"
	"fmt"
	"math/big"
	"math/bits"
	"net"
	"sync"

	netutils "k8s.io/utils/net"
)

// MultiCIDRSet manages a set of CIDR ranges from which blocks of IPs can
// be allocated from.
type MultiCIDRSet struct {
	sync.Mutex
	// ClusterCIDR is the CIDR assigned to the cluster.
	ClusterCIDR *net.IPNet
	// NodeMaskSize is the mask size, in bits,assigned to the nodes
	// caches the mask size to avoid the penalty of calling nodeMask.Size().
	NodeMaskSize int
	// MaxCIDRs is the maximum number of CIDRs that can be allocated.
	MaxCIDRs int
	// Label stores the CIDR in a string, it is used to identify the metrics such
	// as Number of allocations, Total number of CIDR releases, Percentage of
	// allocated CIDRs, Tries required for allocating a CIDR for a particular CIDRSet.
	Label string
	// AllocatedCIDRMap stores all the allocated CIDRs from the current CIDRSet.
	// Stores a mapping of the next candidate CIDR for allocation to it's
	// allocation status. Next candidate is used only if allocation status is false.
	AllocatedCIDRMap map[string]bool

	// clusterMaskSize is the mask size, in bits, assigned to the cluster.
	// caches the mask size to avoid the penalty of calling clusterCIDR.Mask.Size().
	clusterMaskSize int
	// nodeMask is the network mask assigned to the nodes.
	nodeMask net.IPMask
	// allocatedCIDRs counts the number of CIDRs allocated.
	allocatedCIDRs int
	// nextCandidate points to the next CIDR that should be free.
	nextCandidate int
}

// ClusterCIDR is an internal representation of the ClusterCIDR API object.
type ClusterCIDR struct {
	// Name of the associated ClusterCIDR API object.
	Name string
	// IPv4CIDRSet is the MultiCIDRSet representation of ClusterCIDR.spec.ipv4
	// of the associated ClusterCIDR API object.
	IPv4CIDRSet *MultiCIDRSet
	// IPv6CIDRSet is the MultiCIDRSet representation of ClusterCIDR.spec.ipv6
	// of the associated ClusterCIDR API object.
	IPv6CIDRSet *MultiCIDRSet
	// AssociatedNodes is used to identify which nodes have CIDRs allocated from this ClusterCIDR.
	// Stores a mapping of node name to association status.
	AssociatedNodes map[string]bool
	// Terminating is used to identify whether ClusterCIDR has been marked for termination.
	Terminating bool
}

const (
	// The subnet mask size cannot be greater than 16 more than the cluster mask size
	// TODO: https://github.com/kubernetes/kubernetes/issues/44918
	// clusterSubnetMaxDiff limited to 16 due to the uncompressed bitmap.
	// Due to this limitation the subnet mask for IPv6 cluster cidr needs to be >= 48
	// as default mask size for IPv6 is 64.
	clusterSubnetMaxDiff = 16
	// halfIPv6Len is the half of the IPv6 length.
	halfIPv6Len = net.IPv6len / 2
)

// CIDRRangeNoCIDRsRemainingErr is an error type used to denote there is no more
// space to allocate CIDR ranges from the given CIDR.
type CIDRRangeNoCIDRsRemainingErr struct {
	// CIDR represents the CIDR which is exhausted.
	CIDR string
}

func (err *CIDRRangeNoCIDRsRemainingErr) Error() string {
	return fmt.Sprintf("CIDR allocation failed; there are no remaining CIDRs left to allocate in the range %s", err.CIDR)
}

// CIDRSetSubNetTooBigErr is an error type to denote that subnet mask size is too
// big compared to the CIDR mask size.
type CIDRSetSubNetTooBigErr struct {
	cidr            string
	subnetMaskSize  int
	clusterMaskSize int
}

func (err *CIDRSetSubNetTooBigErr) Error() string {
	return fmt.Sprintf("Creation of New CIDR Set failed for %s. "+
		"PerNodeMaskSize %d is too big for CIDR Mask %d, Maximum difference allowed "+
		"is %d", err.cidr, err.subnetMaskSize, err.clusterMaskSize, clusterSubnetMaxDiff)
}

// NewMultiCIDRSet creates a new MultiCIDRSet.
func NewMultiCIDRSet(cidrConfig *net.IPNet, perNodeHostBits int) (*MultiCIDRSet, error) {
	clusterMask := cidrConfig.Mask
	clusterMaskSize, bits := clusterMask.Size()

	var subNetMaskSize int
	switch /*v4 or v6*/ {
	case netutils.IsIPv4(cidrConfig.IP):
		subNetMaskSize = 32 - perNodeHostBits
	case netutils.IsIPv6(cidrConfig.IP):
		subNetMaskSize = 128 - perNodeHostBits
	}

	if netutils.IsIPv6(cidrConfig.IP) && (subNetMaskSize-clusterMaskSize > clusterSubnetMaxDiff) {
		return nil, &CIDRSetSubNetTooBigErr{
			cidr:            cidrConfig.String(),
			subnetMaskSize:  subNetMaskSize,
			clusterMaskSize: clusterMaskSize,
		}
	}

	// Register MultiCIDRSet metrics.
	registerCidrsetMetrics()

	maxCIDRs := getMaxCIDRs(subNetMaskSize, clusterMaskSize)
	multiCIDRSet := &MultiCIDRSet{
		ClusterCIDR:      cidrConfig,
		nodeMask:         net.CIDRMask(subNetMaskSize, bits),
		clusterMaskSize:  clusterMaskSize,
		MaxCIDRs:         maxCIDRs,
		NodeMaskSize:     subNetMaskSize,
		Label:            cidrConfig.String(),
		AllocatedCIDRMap: make(map[string]bool, 0),
	}
	cidrSetMaxCidrs.WithLabelValues(multiCIDRSet.Label).Set(float64(maxCIDRs))

	return multiCIDRSet, nil
}

func (s *MultiCIDRSet) indexToCIDRBlock(index int) (*net.IPNet, error) {
	var ip []byte
	switch /*v4 or v6*/ {
	case netutils.IsIPv4(s.ClusterCIDR.IP):
		j := uint32(index) << uint32(32-s.NodeMaskSize)
		ipInt := (binary.BigEndian.Uint32(s.ClusterCIDR.IP)) | j
		ip = make([]byte, net.IPv4len)
		binary.BigEndian.PutUint32(ip, ipInt)
	case netutils.IsIPv6(s.ClusterCIDR.IP):
		// leftClusterIP      |     rightClusterIP
		// 2001:0DB8:1234:0000:0000:0000:0000:0000
		const v6NBits = 128
		const halfV6NBits = v6NBits / 2
		leftClusterIP := binary.BigEndian.Uint64(s.ClusterCIDR.IP[:halfIPv6Len])
		rightClusterIP := binary.BigEndian.Uint64(s.ClusterCIDR.IP[halfIPv6Len:])

		ip = make([]byte, net.IPv6len)

		if s.NodeMaskSize <= halfV6NBits {
			// We only care about left side IP.
			leftClusterIP |= uint64(index) << uint(halfV6NBits-s.NodeMaskSize)
		} else {
			if s.clusterMaskSize < halfV6NBits {
				// see how many bits are needed to reach the left side.
				btl := uint(s.NodeMaskSize - halfV6NBits)
				indexMaxBit := uint(64 - bits.LeadingZeros64(uint64(index)))
				if indexMaxBit > btl {
					leftClusterIP |= uint64(index) >> btl
				}
			}
			// the right side will be calculated the same way either the
			// subNetMaskSize affects both left and right sides.
			rightClusterIP |= uint64(index) << uint(v6NBits-s.NodeMaskSize)
		}
		binary.BigEndian.PutUint64(ip[:halfIPv6Len], leftClusterIP)
		binary.BigEndian.PutUint64(ip[halfIPv6Len:], rightClusterIP)
	default:
		return nil, fmt.Errorf("invalid IP: %s", s.ClusterCIDR.IP)
	}
	return &net.IPNet{
		IP:   ip,
		Mask: s.nodeMask,
	}, nil
}

// NextCandidate returns the next candidate and the last evaluated index
// for the current cidrSet. Returns nil if the candidate is already allocated.
func (s *MultiCIDRSet) NextCandidate() (*net.IPNet, int, error) {
	s.Lock()
	defer s.Unlock()

	if s.allocatedCIDRs == s.MaxCIDRs {
		return nil, 0, &CIDRRangeNoCIDRsRemainingErr{
			CIDR: s.Label,
		}
	}

	candidate := s.nextCandidate
	for i := 0; i < s.MaxCIDRs; i++ {
		nextCandidateCIDR, err := s.indexToCIDRBlock(candidate)
		if err != nil {
			return nil, i, err
		}
		// Check if the nextCandidate is not already allocated.
		if _, ok := s.AllocatedCIDRMap[nextCandidateCIDR.String()]; !ok {
			s.nextCandidate = (candidate + 1) % s.MaxCIDRs
			return nextCandidateCIDR, i, nil
		}
		candidate = (candidate + 1) % s.MaxCIDRs
	}

	return nil, s.MaxCIDRs, &CIDRRangeNoCIDRsRemainingErr{
		CIDR: s.Label,
	}
}

// getBeginningAndEndIndices returns the indices for the given CIDR, returned
// values are inclusive indices [beginning, end].
func (s *MultiCIDRSet) getBeginningAndEndIndices(cidr *net.IPNet) (int, int, error) {
	if cidr == nil {
		return -1, -1, fmt.Errorf("error getting indices for cluster cidr %v, cidr is nil", s.ClusterCIDR)
	}
	begin, end := 0, s.MaxCIDRs-1
	cidrMask := cidr.Mask
	maskSize, _ := cidrMask.Size()
	var ipSize int

	if !s.ClusterCIDR.Contains(cidr.IP.Mask(s.ClusterCIDR.Mask)) && !cidr.Contains(s.ClusterCIDR.IP.Mask(cidr.Mask)) {
		return -1, -1, fmt.Errorf("cidr %v is out the range of cluster cidr %v", cidr, s.ClusterCIDR)
	}

	if s.clusterMaskSize < maskSize {
		var err error
		ipSize = net.IPv4len
		if netutils.IsIPv6(cidr.IP) {
			ipSize = net.IPv6len
		}
		begin, err = s.getIndexForCIDR(&net.IPNet{
			IP:   cidr.IP.Mask(s.nodeMask),
			Mask: s.nodeMask,
		})
		if err != nil {
			return -1, -1, err
		}
		ip := make([]byte, ipSize)
		if netutils.IsIPv4(cidr.IP) {
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
			IP:   net.IP(ip).Mask(s.nodeMask),
			Mask: s.nodeMask,
		})
		if err != nil {
			return -1, -1, err
		}
	}
	return begin, end, nil
}

// Release releases the given CIDR range.
func (s *MultiCIDRSet) Release(cidr *net.IPNet) error {
	begin, end, err := s.getBeginningAndEndIndices(cidr)
	if err != nil {
		return err
	}
	s.Lock()
	defer s.Unlock()

	for i := begin; i <= end; i++ {
		// Remove from the allocated CIDR Map and decrement the counter only if currently
		// marked allocated. Avoids double counting.
		currCIDR, err := s.indexToCIDRBlock(i)
		if err != nil {
			return err
		}
		if _, ok := s.AllocatedCIDRMap[currCIDR.String()]; ok {
			delete(s.AllocatedCIDRMap, currCIDR.String())
			s.allocatedCIDRs--
			cidrSetReleases.WithLabelValues(s.Label).Inc()
		}
	}

	cidrSetUsage.WithLabelValues(s.Label).Set(float64(s.allocatedCIDRs) / float64(s.MaxCIDRs))

	return nil
}

// Occupy marks the given CIDR range as used. Occupy succeeds even if the CIDR
// range was previously used.
func (s *MultiCIDRSet) Occupy(cidr *net.IPNet) (err error) {
	begin, end, err := s.getBeginningAndEndIndices(cidr)
	if err != nil {
		return err
	}
	s.Lock()
	defer s.Unlock()

	for i := begin; i <= end; i++ {
		// Add to the allocated CIDR Map and increment the counter only if not already
		// marked allocated. Prevents double counting.
		currCIDR, err := s.indexToCIDRBlock(i)
		if err != nil {
			return err
		}
		if _, ok := s.AllocatedCIDRMap[currCIDR.String()]; !ok {
			s.AllocatedCIDRMap[currCIDR.String()] = true
			cidrSetAllocations.WithLabelValues(s.Label).Inc()
			s.allocatedCIDRs++
		}
	}
	cidrSetUsage.WithLabelValues(s.Label).Set(float64(s.allocatedCIDRs) / float64(s.MaxCIDRs))

	return nil
}

func (s *MultiCIDRSet) getIndexForCIDR(cidr *net.IPNet) (int, error) {
	return s.getIndexForIP(cidr.IP)
}

func (s *MultiCIDRSet) getIndexForIP(ip net.IP) (int, error) {
	if ip.To4() != nil {
		cidrIndex := (binary.BigEndian.Uint32(s.ClusterCIDR.IP) ^ binary.BigEndian.Uint32(ip.To4())) >> uint32(32-s.NodeMaskSize)
		if cidrIndex >= uint32(s.MaxCIDRs) {
			return 0, fmt.Errorf("CIDR: %v/%v is out of the range of CIDR allocator", ip, s.NodeMaskSize)
		}
		return int(cidrIndex), nil
	}
	if netutils.IsIPv6(ip) {
		bigIP := big.NewInt(0).SetBytes(s.ClusterCIDR.IP)
		bigIP = bigIP.Xor(bigIP, big.NewInt(0).SetBytes(ip))
		cidrIndexBig := bigIP.Rsh(bigIP, uint(net.IPv6len*8-s.NodeMaskSize))
		cidrIndex := cidrIndexBig.Uint64()
		if cidrIndex >= uint64(s.MaxCIDRs) {
			return 0, fmt.Errorf("CIDR: %v/%v is out of the range of CIDR allocator", ip, s.NodeMaskSize)
		}
		return int(cidrIndex), nil
	}

	return 0, fmt.Errorf("invalid IP: %v", ip)
}

// UpdateEvaluatedCount increments the evaluated count.
func (s *MultiCIDRSet) UpdateEvaluatedCount(evaluated int) {
	cidrSetAllocationTriesPerRequest.WithLabelValues(s.Label).Observe(float64(evaluated))
}

// getMaxCIDRs returns the max number of CIDRs that can be obtained by subdividing a mask of size `clusterMaskSize`
// into subnets with mask of size `subNetMaskSize`.
func getMaxCIDRs(subNetMaskSize, clusterMaskSize int) int {
	return 1 << uint32(subNetMaskSize-clusterMaskSize)
}
