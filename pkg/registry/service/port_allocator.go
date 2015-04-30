/*
Copyright 2015 Google Inc. All rights reserved.

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

package service

import (
	"fmt"
	math_rand "math/rand"
	"strconv"

	"github.com/golang/glog"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// The smallest size of port range we accept.
const minPortPoolSize = 8

type portAllocator struct {
	portRange util.PortRange
	pool      PoolAllocator
}

// newPortAllocator creates and initializes a new portAllocator object.
func newPortAllocator(portRange *util.PortRange) *portAllocator {
	rangeSize := portRange.Size
	if rangeSize < minPortPoolSize {
		glog.Errorf("PortAllocator requires at least %d ports", minPortPoolSize)
		return nil
	}

	a := &portAllocator{
		portRange: *portRange,
	}
	a.pool.Init(&portPoolDriver{portRange: *portRange})

	return a
}

// Allocate portAllocator a specific port.  This is useful when recovering saved state.
func (a *portAllocator) Allocate(port int) error {
	if !a.portRange.Contains(port) {
		return fmt.Errorf("Port %v does not fall within port-range %v", port, a.portRange)
	}

	return a.pool.Allocate(portToString(port))
}

// Allocate allocates and returns a port.
func (a *portAllocator) AllocateNext() (int, error) {
	s := a.pool.AllocateNext()
	if s == "" {
		return 0, fmt.Errorf("can't find a free port in %s", a.portRange)
	}
	return stringToPort(s), nil
}

// Release de-allocates a port.
func (a *portAllocator) Release(port int) error {
	if !a.portRange.Contains(port) {
		return fmt.Errorf("Port %v does not fall within port-range %v", port, a.portRange)
	}

	if !a.pool.Release(portToString(port)) {
		glog.Warning("Released port that was not assigned: ", port)
	}
	return nil
}

// Implements PoolDriver over the ports in a PortRange
type portPoolDriver struct {
	portRange util.PortRange
}

func (d *portPoolDriver) PickRandom(random *math_rand.Rand) string {
	n := random.Int()
	port := d.portRange.Base + (n % d.portRange.Size)
	return portToString(port)
}

func (d *portPoolDriver) Iterate() StringIterator {
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
