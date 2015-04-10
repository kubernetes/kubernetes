/*
Copyright 2014 Google Inc. All rights reserved.

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
func newPortAllocator(portRange util.PortRange) *portAllocator {
	rangeSize := portRange.Size()
	if rangeSize < minPortPoolSize {
		glog.Errorf("PortAllocator requires at least %d ports", minPortPoolSize)
		return nil
	}

	a := &portAllocator{
		portRange: portRange,
	}
	a.pool.Init(&portPoolDriver{portRange: portRange})

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

type portPoolDriver struct {
	portRange util.PortRange
}

func (d *portPoolDriver) PickRandom(seed int) string {
	n := d.portRange.Size()
	port := d.portRange.MinInclusive + (seed % n)
	return portToString(port)
}

func (d *portPoolDriver) IterateStart() string {
	port := d.portRange.MinInclusive
	return portToString(port)
}

func (d *portPoolDriver) IterateNext(p string) string {
	port := stringToPort(p)
	port++
	if port >= d.portRange.MaxExclusive {
		return ""
	}
	return portToString(port)
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
