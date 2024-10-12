//go:build linux
// +build linux

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

package conntrack

import (
	"fmt"

	"github.com/vishvananda/netlink"

	"k8s.io/klog/v2"
)

// Interface for dealing with conntrack
type Interface interface {
	// ClearEntries deletes conntrack entries for connections of the given IP family,
	// filtered by the given filters.
	ClearEntries(ipFamily uint8, filters ...netlink.CustomConntrackFilter) error
}

// netlinkHandler allows consuming real and mockable implementation for testing.
type netlinkHandler interface {
	ConntrackDeleteFilters(netlink.ConntrackTableType, netlink.InetFamily, ...netlink.CustomConntrackFilter) (uint, error)
}

// conntracker implements Interface by using netlink APIs.
type conntracker struct {
	handler netlinkHandler
}

var _ Interface = &conntracker{}

func New() Interface {
	return newConntracker(&netlink.Handle{})
}

func newConntracker(handler netlinkHandler) Interface {
	return &conntracker{handler: handler}
}

// ClearEntries deletes conntrack entries for connections of the given IP family,
// filtered by the given filters.
func (ct *conntracker) ClearEntries(ipFamily uint8, filters ...netlink.CustomConntrackFilter) error {
	if len(filters) == 0 {
		klog.V(7).InfoS("no conntrack filters provided")
		return nil
	}

	n, err := ct.handler.ConntrackDeleteFilters(netlink.ConntrackTable, netlink.InetFamily(ipFamily), filters...)
	if err != nil {
		return fmt.Errorf("error deleting conntrack entries, error: %w", err)
	}
	klog.V(4).InfoS("Cleared conntrack entries", "count", n)
	return nil
}
