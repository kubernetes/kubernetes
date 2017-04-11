// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl plan9 solaris

package ipv4

import "net"

// MulticastTTL returns the time-to-live field value for outgoing
// multicast packets.
func (c *dgramOpt) MulticastTTL() (int, error) {
	return 0, errOpNoSupport
}

// SetMulticastTTL sets the time-to-live field value for future
// outgoing multicast packets.
func (c *dgramOpt) SetMulticastTTL(ttl int) error {
	return errOpNoSupport
}

// MulticastInterface returns the default interface for multicast
// packet transmissions.
func (c *dgramOpt) MulticastInterface() (*net.Interface, error) {
	return nil, errOpNoSupport
}

// SetMulticastInterface sets the default interface for future
// multicast packet transmissions.
func (c *dgramOpt) SetMulticastInterface(ifi *net.Interface) error {
	return errOpNoSupport
}

// MulticastLoopback reports whether transmitted multicast packets
// should be copied and send back to the originator.
func (c *dgramOpt) MulticastLoopback() (bool, error) {
	return false, errOpNoSupport
}

// SetMulticastLoopback sets whether transmitted multicast packets
// should be copied and send back to the originator.
func (c *dgramOpt) SetMulticastLoopback(on bool) error {
	return errOpNoSupport
}

// JoinGroup joins the group address group on the interface ifi.
// By default all sources that can cast data to group are accepted.
// It's possible to mute and unmute data transmission from a specific
// source by using ExcludeSourceSpecificGroup and
// IncludeSourceSpecificGroup.
// JoinGroup uses the system assigned multicast interface when ifi is
// nil, although this is not recommended because the assignment
// depends on platforms and sometimes it might require routing
// configuration.
func (c *dgramOpt) JoinGroup(ifi *net.Interface, group net.Addr) error {
	return errOpNoSupport
}

// LeaveGroup leaves the group address group on the interface ifi
// regardless of whether the group is any-source group or
// source-specific group.
func (c *dgramOpt) LeaveGroup(ifi *net.Interface, group net.Addr) error {
	return errOpNoSupport
}

// JoinSourceSpecificGroup joins the source-specific group comprising
// group and source on the interface ifi.
// JoinSourceSpecificGroup uses the system assigned multicast
// interface when ifi is nil, although this is not recommended because
// the assignment depends on platforms and sometimes it might require
// routing configuration.
func (c *dgramOpt) JoinSourceSpecificGroup(ifi *net.Interface, group, source net.Addr) error {
	return errOpNoSupport
}

// LeaveSourceSpecificGroup leaves the source-specific group on the
// interface ifi.
func (c *dgramOpt) LeaveSourceSpecificGroup(ifi *net.Interface, group, source net.Addr) error {
	return errOpNoSupport
}

// ExcludeSourceSpecificGroup excludes the source-specific group from
// the already joined any-source groups by JoinGroup on the interface
// ifi.
func (c *dgramOpt) ExcludeSourceSpecificGroup(ifi *net.Interface, group, source net.Addr) error {
	return errOpNoSupport
}

// IncludeSourceSpecificGroup includes the excluded source-specific
// group by ExcludeSourceSpecificGroup again on the interface ifi.
func (c *dgramOpt) IncludeSourceSpecificGroup(ifi *net.Interface, group, source net.Addr) error {
	return errOpNoSupport
}

// ICMPFilter returns an ICMP filter.
// Currently only Linux supports this.
func (c *dgramOpt) ICMPFilter() (*ICMPFilter, error) {
	return nil, errOpNoSupport
}

// SetICMPFilter deploys the ICMP filter.
// Currently only Linux supports this.
func (c *dgramOpt) SetICMPFilter(f *ICMPFilter) error {
	return errOpNoSupport
}
