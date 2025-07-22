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

package ipallocator

import (
	"errors"
	"fmt"
	"net"

	api "k8s.io/kubernetes/pkg/apis/core"
)

// Interface manages the allocation of IP addresses out of a range. Interface
// should be threadsafe.
type Interface interface {
	Allocate(net.IP) error
	AllocateNext() (net.IP, error)
	Release(net.IP) error
	ForEach(func(net.IP))
	CIDR() net.IPNet
	IPFamily() api.IPFamily
	Has(ip net.IP) bool
	Destroy()
	EnableMetrics()

	// DryRun offers a way to try operations without persisting them.
	DryRun() Interface
}

var (
	ErrFull              = errors.New("range is full")
	ErrAllocated         = errors.New("provided IP is already allocated")
	ErrMismatchedNetwork = errors.New("the provided network does not match the current range")
	ErrNotReady          = errors.New("allocator not ready")
)

type ErrNotInRange struct {
	IP         net.IP
	ValidRange string
}

func (e *ErrNotInRange) Error() string {
	return fmt.Sprintf("the provided IP (%v) is not in the valid range. The range of valid IPs is %s", e.IP, e.ValidRange)
}
