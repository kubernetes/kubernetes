/*
Copyright 2019 The Kubernetes Authors.

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

package socketmask

import (
	"fmt"
	"math/bits"
)

// SocketMask interface allows hint providers to create SocketMasks for TopologyHints
type SocketMask interface {
	Add(sockets ...int) error
	Remove(sockets ...int) error
	And(masks ...SocketMask)
	Or(masks ...SocketMask)
	Clear()
	Fill()
	IsEqual(mask SocketMask) bool
	IsEmpty() bool
	IsSet(socket int) bool
	IsNarrowerThan(mask SocketMask) bool
	String() string
	Count() int
	GetSockets() []int
}

type socketMask uint64

// NewEmptySocketMask creates a new, empty SocketMask
func NewEmptySocketMask() SocketMask {
	s := socketMask(0)
	return &s
}

// NewSocketMask creates a new SocketMask
func NewSocketMask(sockets ...int) (SocketMask, error) {
	s := socketMask(0)
	err := (&s).Add(sockets...)
	if err != nil {
		return nil, err
	}
	return &s, nil
}

// Add adds the sockets with topology affinity to the SocketMask
func (s *socketMask) Add(sockets ...int) error {
	mask := *s
	for _, i := range sockets {
		if i < 0 || i >= 64 {
			return fmt.Errorf("socket number must be in range 0-63")
		}
		mask |= 1 << uint64(i)
	}
	*s = mask
	return nil
}

// Remove removes specified sockets from SocketMask
func (s *socketMask) Remove(sockets ...int) error {
	mask := *s
	for _, i := range sockets {
		if i < 0 || i >= 64 {
			return fmt.Errorf("socket number must be in range 0-63")
		}
		mask &^= 1 << uint64(i)
	}
	*s = mask
	return nil
}

// And performs and operation on all bits in masks
func (s *socketMask) And(masks ...SocketMask) {
	for _, m := range masks {
		*s &= *m.(*socketMask)
	}
}

// Or performs or operation on all bits in masks
func (s *socketMask) Or(masks ...SocketMask) {
	for _, m := range masks {
		*s |= *m.(*socketMask)
	}
}

// Clear resets all bits in mask to zero
func (s *socketMask) Clear() {
	*s = 0
}

// Fill sets all bits in mask to one
func (s *socketMask) Fill() {
	*s = socketMask(^uint64(0))
}

// IsEmpty checks mask to see if all bits are zero
func (s *socketMask) IsEmpty() bool {
	return *s == 0
}

// IsSet checks socket in mask to see if bit is set to one
func (s *socketMask) IsSet(socket int) bool {
	if socket < 0 || socket >= 64 {
		return false
	}
	return (*s & (1 << uint64(socket))) > 0
}

// IsEqual checks if masks are equal
func (s *socketMask) IsEqual(mask SocketMask) bool {
	return *s == *mask.(*socketMask)
}

// IsNarrowerThan checks if one mask is narrower than another.
//
// A mask is said to be "narrower" than another if it has lets bits set. If the
// same number of bits are set in both masks, then the mask with more
// lower-numbered bits set wins out.
func (s *socketMask) IsNarrowerThan(mask SocketMask) bool {
	if s.Count() == mask.Count() {
		if *s < *mask.(*socketMask) {
			return true
		}
	}
	return s.Count() < mask.Count()
}

// String converts mask to string
func (s *socketMask) String() string {
	return fmt.Sprintf("%064b", *s)
}

// Count counts number of bits in mask set to one
func (s *socketMask) Count() int {
	return bits.OnesCount64(uint64(*s))
}

// GetSockets returns each socket number with bits set to one
func (s *socketMask) GetSockets() []int {
	var sockets []int
	for i := uint64(0); i < 64; i++ {
		if (*s & (1 << i)) > 0 {
			sockets = append(sockets, int(i))
		}
	}
	return sockets
}

// And is a package level implementation of 'and' between first and masks
func And(first SocketMask, masks ...SocketMask) SocketMask {
	s := *first.(*socketMask)
	s.And(masks...)
	return &s
}

// Or is a package level implementation of 'or' between first and masks
func Or(first SocketMask, masks ...SocketMask) SocketMask {
	s := *first.(*socketMask)
	s.Or(masks...)
	return &s
}

// IterateSocketMasks iterates all possible masks from a list of sockets,
// issuing a callback on each mask.
func IterateSocketMasks(sockets []int, callback func(SocketMask)) {
	var iterate func(sockets, accum []int, size int)
	iterate = func(sockets, accum []int, size int) {
		if len(accum) == size {
			mask, _ := NewSocketMask(accum...)
			callback(mask)
			return
		}
		for i := range sockets {
			iterate(sockets[i+1:], append(accum, sockets[i]), size)
		}
	}

	for i := 1; i <= len(sockets); i++ {
		iterate(sockets, []int{}, i)
	}
}
