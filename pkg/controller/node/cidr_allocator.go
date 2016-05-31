/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"errors"
	"fmt"
	"math/big"
	"net"
	"sync"
)

var errCIDRRangeNoCIDRsRemaining = errors.New("CIDR allocation failed; there are no remaining CIDRs left to allocate in the accepted range")

// CIDRAllocator is an interface implemented by things that know how to allocate/occupy/recycle CIDR for nodes.
type CIDRAllocator interface {
	AllocateNext() (*net.IPNet, error)
	Occupy(*net.IPNet) error
	Release(*net.IPNet) error
}

type rangeAllocator struct {
	clusterCIDR     *net.IPNet
	clusterIP       net.IP
	clusterMaskSize int
	subNetMaskSize  int
	maxCIDRs        int
	used            big.Int
	lock            sync.Mutex
	nextCandidate   int
}

// NewCIDRRangeAllocator returns a CIDRAllocator to allocate CIDR for node
// Caller must ensure subNetMaskSize is not less than cluster CIDR mask size.
func NewCIDRRangeAllocator(clusterCIDR *net.IPNet, subNetMaskSize int) CIDRAllocator {
	clusterMask := clusterCIDR.Mask
	clusterMaskSize, _ := clusterMask.Size()

	ra := &rangeAllocator{
		clusterCIDR:     clusterCIDR,
		clusterIP:       clusterCIDR.IP.To4(),
		clusterMaskSize: clusterMaskSize,
		subNetMaskSize:  subNetMaskSize,
		maxCIDRs:        1 << uint32(subNetMaskSize-clusterMaskSize),
		nextCandidate:   0,
	}
	return ra
}

func (r *rangeAllocator) AllocateNext() (*net.IPNet, error) {
	r.lock.Lock()
	defer r.lock.Unlock()

	nextUnused := -1
	for i := 0; i < r.maxCIDRs; i++ {
		candidate := (i + r.nextCandidate) % r.maxCIDRs
		if r.used.Bit(candidate) == 0 {
			nextUnused = candidate
			break
		}
	}
	if nextUnused == -1 {
		return nil, errCIDRRangeNoCIDRsRemaining
	}
	r.nextCandidate = (nextUnused + 1) % r.maxCIDRs

	r.used.SetBit(&r.used, nextUnused, 1)

	j := uint32(nextUnused) << uint32(32-r.subNetMaskSize)
	ipInt := (binary.BigEndian.Uint32(r.clusterIP)) | j
	ip := make([]byte, 4)
	binary.BigEndian.PutUint32(ip, ipInt)

	return &net.IPNet{
		IP:   ip,
		Mask: net.CIDRMask(r.subNetMaskSize, 32),
	}, nil
}

func (r *rangeAllocator) Release(cidr *net.IPNet) error {
	used, err := r.getIndexForCIDR(cidr)
	if err != nil {
		return err
	}

	r.lock.Lock()
	defer r.lock.Unlock()
	r.used.SetBit(&r.used, used, 0)

	return nil
}

func (r *rangeAllocator) MaxCIDRs() int {
	return r.maxCIDRs
}

func (r *rangeAllocator) Occupy(cidr *net.IPNet) (err error) {
	begin, end := 0, r.maxCIDRs
	cidrMask := cidr.Mask
	maskSize, _ := cidrMask.Size()

	if !r.clusterCIDR.Contains(cidr.IP.Mask(r.clusterCIDR.Mask)) && !cidr.Contains(r.clusterCIDR.IP.Mask(cidr.Mask)) {
		return fmt.Errorf("cidr %v is out the range of cluster cidr %v", cidr, r.clusterCIDR)
	}

	if r.clusterMaskSize < maskSize {
		subNetMask := net.CIDRMask(r.subNetMaskSize, 32)
		begin, err = r.getIndexForCIDR(&net.IPNet{
			IP:   cidr.IP.To4().Mask(subNetMask),
			Mask: subNetMask,
		})
		if err != nil {
			return err
		}

		ip := make([]byte, 4)
		ipInt := binary.BigEndian.Uint32(cidr.IP) | (^binary.BigEndian.Uint32(cidr.Mask))
		binary.BigEndian.PutUint32(ip, ipInt)
		end, err = r.getIndexForCIDR(&net.IPNet{
			IP:   net.IP(ip).To4().Mask(subNetMask),
			Mask: subNetMask,
		})
		if err != nil {
			return err
		}
	}

	r.lock.Lock()
	defer r.lock.Unlock()
	for i := begin; i <= end; i++ {
		r.used.SetBit(&r.used, i, 1)
	}

	return nil
}

func (r *rangeAllocator) getIndexForCIDR(cidr *net.IPNet) (int, error) {
	cidrIndex := (binary.BigEndian.Uint32(r.clusterIP) ^ binary.BigEndian.Uint32(cidr.IP.To4())) >> uint32(32-r.subNetMaskSize)

	if cidrIndex >= uint32(r.maxCIDRs) {
		return 0, fmt.Errorf("CIDR: %v is out of the range of CIDR allocator", cidr)
	}

	return int(cidrIndex), nil
}
