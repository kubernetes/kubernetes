/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package portallocator

import (
	"fmt"
	mathrand "math/rand"
	"strconv"

	"github.com/golang/glog"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pool"
	etcdpool "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pool/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// The smallest size of port range we accept.
const minPortPoolSize = 8

// The base key in etc
const baseKey = "/pools/serviceports/"

type PortAllocator struct {
	portRange util.PortRange
	pool      pool.PoolAllocator
}

// newPortAllocator creates and initializes a new portAllocator object.
func NewPortAllocator(portRange *util.PortRange, etcd *tools.EtcdHelper) *PortAllocator {
	rangeSize := portRange.Size
	if rangeSize < minPortPoolSize {
		glog.Errorf("PortAllocator requires at least %d ports", minPortPoolSize)
		return nil
	}

	a := &PortAllocator{
		portRange: *portRange,
	}

	if etcd == nil {
		pa := &pool.MemoryPoolAllocator{}
		pa.Init(&portPoolDriver{portRange: *portRange})
		a.pool = pa
	} else {
		pa := &etcdpool.EtcdPoolAllocator{}
		pa.Init(&portPoolDriver{portRange: *portRange}, etcd, baseKey)
		a.pool = pa
	}
	return a
}

// Allocate portAllocator a specific port.  This is useful when recovering saved state.
func (a *PortAllocator) Allocate(port int, owner string) error {
	if !a.portRange.Contains(port) {
		return fmt.Errorf("Port %v does not fall within port-range %v", port, a.portRange)
	}

	locked, err := a.pool.Allocate(portToString(port), owner)
	if err != nil {
		return err
	}
	if !locked {
		return fmt.Errorf("Port %d is already allocated", port)
	}
	return nil
}

// Allocate allocates and returns a port.
func (a *PortAllocator) AllocateNext(owner string) (int, error) {
	s, err := a.pool.AllocateNext(owner)
	if err != nil {
		return 0, err
	}
	if s == "" {
		return 0, fmt.Errorf("can't find a free port in %s", a.portRange)
	}
	return stringToPort(s), nil
}

// Release de-allocates a port.
func (a *PortAllocator) Release(port int) error {
	if !a.portRange.Contains(port) {
		return fmt.Errorf("Port %v does not fall within port-range %v", port, a.portRange)
	}

	released, err := a.pool.Release(portToString(port))
	if err != nil {
		return err
	}
	if !released {
		glog.Warning("Released port that was not assigned: ", port)
	}
	return nil
}

// For tests
func (a *PortAllocator) DisableRandomAllocation() {
	a.pool.DisableRandomAllocation()
}

// Implements PoolDriver over the ports in a PortRange
type portPoolDriver struct {
	portRange util.PortRange
}

func (d *portPoolDriver) PickRandom(random *mathrand.Rand) string {
	n := random.Int()
	port := d.portRange.Base + (n % d.portRange.Size)
	return portToString(port)
}

func (d *portPoolDriver) Iterator() pool.PoolIterator {
	return &portPoolIterator{portRange: d.portRange}
}

// Implements StringIterator over the ports in a PortRange
type portPoolIterator struct {
	portRange util.PortRange
	count     int
}

// Implements StringIterator::Next
func (it *portPoolIterator) Next() (string, bool) {
	it.count++
	if it.count > it.portRange.Size {
		return "", false
	}

	port := it.count + it.portRange.Base - 1
	return portToString(port), true
}

func portToString(p int) string {
	return strconv.Itoa(p)
}

func stringToPort(s string) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		panic("Invalid port-string: " + s)
	}
	return n
}
