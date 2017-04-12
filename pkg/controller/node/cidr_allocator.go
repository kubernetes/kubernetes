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
	"errors"
	"net"

	v1 "k8s.io/kubernetes/pkg/api/v1"
)

var errCIDRRangeNoCIDRsRemaining = errors.New(
	"CIDR allocation failed; there are no remaining CIDRs left to allocate in the accepted range")

type nodeAndCIDR struct {
	cidr     *net.IPNet
	nodeName string
}

// CIDRAllocatorType is the type of the allocator to use.
type CIDRAllocatorType string

const (
	RangeAllocatorType CIDRAllocatorType = "RangeAllocator"
	CloudAllocatorType CIDRAllocatorType = "CloudAllocator"
)

// CIDRAllocator is an interface implemented by things that know how to
// allocate/occupy/recycle CIDR for nodes.
type CIDRAllocator interface {
	// AllocateOrOccupyCIDR looks at the given node, assigns it a valid
	// CIDR if it doesn't currently have one or mark the CIDR as used if
	// the node already have one.
	AllocateOrOccupyCIDR(node *v1.Node) error
	// ReleaseCIDR releases the CIDR of the removed node
	ReleaseCIDR(node *v1.Node) error
}
