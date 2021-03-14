/*
Copyright 2021 The Kubernetes Authors.

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

package allocation

import (
	"context"
	"fmt"
	"math/big"
	"math/rand"
	"net"
	"sync"
	"time"

	allocation "k8s.io/api/allocation/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	allocationclient "k8s.io/client-go/kubernetes/typed/allocation/v1alpha1"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	utilnet "k8s.io/utils/net"
)

type IPRegistry struct {
	Pool   IPRegistryPool
	Shard  IPRegistryShard
	client allocationclient.AllocationV1alpha1Interface
}

// A shard is defined by its size and the index
// Give a pool of 10 IPs and 2 Shards
// The shard is 5 and one will have index 0 and the other index 1
// If a new node is added, the shard size will be 3, with indexes 0,1,2
type IPRegistryPool struct {
	sync.Mutex
	// IPPool contains all the range allocated IPs
	// however, only current shard IPs are guaranteed to be up to date
	// the rest of the IPs are updated by the controller
	IPPool sets.String
	// cidr contains the configured subnet to allocate IPs from
	cidr *net.IPNet
}

// A shard is defined by its size and the index
// Give a pool of 10 IPs and 2 Shards
// The shard is 5 and one will have index 0 and the other index 1
// If a new node is added, the shard size will be 3, with indexes 0,1,2
type IPRegistryShard struct {
	sync.Mutex
	// Size of the shard
	Size int64
	// Index of the shard
	Index int
}

var _ ipallocator.Interface = &IPRegistry{}

// NewAllocatorIPRegistry creates a IPRegistry using a net.IPNet
func NewAllocatorIPRegistry(cidr *net.IPNet, client allocationclient.AllocationV1alpha1Interface) (*IPRegistry, error) {
	return &IPRegistry{
		client: client,
		Pool: IPRegistryPool{
			IPPool: sets.String{},
			cidr:   cidr,
		},
		Shard: IPRegistryShard{},
	}, nil
}

func (r *IPRegistry) ip2ShardIndex(ip net.IP) (int, error) {
	r.Shard.Lock()
	defer r.Shard.Unlock()
	// check that the shard is ready
	if r.Shard.Size == 0 {
		return 0, fmt.Errorf("Allocator not ready, not shard available")
	}
	cidr := r.CIDR()
	cidrBig := utilnet.BigForIP(cidr.IP)
	ipBig := utilnet.BigForIP(ip)
	// calculate offset from the base
	offset := big.NewInt(0).Sub(ipBig, cidrBig).Int64()
	// current number of shards is the total_size / shard_size
	numShards := utilnet.RangeSize(&cidr) / int64(r.Shard.Size)
	// the shard for this ip is the offset / numer of shards
	if offset > 0 && offset <= numShards {
		return int(offset / numShards), nil
	}
	return 0, fmt.Errorf("IP %s out of range", ip.String())
}

// Allocate the corresponding IP Object
func (r *IPRegistry) Allocate(ip net.IP) error {
	// check if the belongs to our pool
	index, err := r.ip2ShardIndex(ip)
	if err != nil {
		return err
	}
	// not rebalancing allowed meanwhile we are trying to allocate
	r.Shard.Lock()
	defer r.Shard.Unlock()

	if index == r.Shard.Index {
		r.Pool.Lock()
		defer r.Pool.Unlock()
		if r.Pool.IPPool.Has(ip.String()) {
			return fmt.Errorf("IP already allocated")
		}
		r.Pool.IPPool.Insert(ip.String())
		return nil
	}

	// Request the IP to the shard owner
	ctx := context.Background()
	req := &allocation.IPRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: ip.String(),
		},
	}
	_, err = r.client.IPRequests().Create(ctx, req, metav1.CreateOptions{})
	if err != nil {
		return err
	}
	// Check if another apiserver has allocated the IP for us
	// TODO review timeout
	err = wait.PollImmediate(250*time.Millisecond, 3*time.Second, func() (bool, error) {
		req, err := r.client.IPRequests().Get(ctx, ip.String(), metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if req.Status.Allocated != nil && *req.Status.Allocated {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		return err
	}
	return nil
}

// AllocateNext return an IP address that wasn't allocated yet
func (r *IPRegistry) AllocateNext() (net.IP, error) {
	// not rebalancing allowed meanwhile we are trying to allocate
	r.Shard.Lock()
	defer r.Shard.Unlock()
	// try to find an empty IP in our shard first
	cidr := r.CIDR()
	numShards := utilnet.RangeSize(&cidr) / int64(r.Shard.Size)
	max := r.Shard.Size
	shards := make([]int, numShards)
	for i := range shards {
		shards[i] = i + 1
	}
	// check our shard first
	shards[0], shards[r.Shard.Index] = r.Shard.Index, 0
	for _, shard := range shards {
		base := int64(shard) * r.Shard.Size
		offset := rand.Int63n(max)
		var i int64
		for i = 0; i < max; i++ {
			at := (offset + i) % max
			ip, err := utilnet.GetIndexedIP(&cidr, int(base+at))
			if err != nil {
				return net.IP{}, ipallocator.ErrAllocated
			}
			if r.Has(ip) {
				continue
			}
			err = r.Allocate(ip)
			// TODO it can happen we fail to allocate
			// because it was already allocated by
			// other apiserver
			// if err is already allocated continue
			// otherwise return the error
			if err != nil {
				continue
			}
			return ip, nil
		}
	}

	return net.IP{}, ipallocator.ErrFull
}

// Release allocated IP
func (r *IPRegistry) Release(ip net.IP) error {
	r.Pool.Lock()
	r.Pool.IPPool.Delete(ip.String())
	r.Pool.Unlock()
	// clean as much as we can
	ctx := context.Background()
	r.client.IPRequests().Delete(ctx, ip.String(), metav1.DeleteOptions{})
	return nil

}

// ForEach executes the function in each allocated IP
func (r *IPRegistry) ForEach(func(net.IP)) {
	// it may not be needed with this approach
}

// CIDR return the current range CIDR
func (r *IPRegistry) CIDR() net.IPNet {
	return *r.Pool.cidr
}

// Has is only used for testing
// It checks the pool that only guarantee the current shard status
// the other shard status may lag behind
func (r *IPRegistry) Has(ip net.IP) bool {
	r.Pool.Lock()
	defer r.Pool.Unlock()
	return r.Pool.IPPool.Has(ip.String())

}
