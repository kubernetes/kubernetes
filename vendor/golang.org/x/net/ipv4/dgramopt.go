// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"net"

	"golang.org/x/net/bpf"
)

// MulticastTTL returns the time-to-live field value for outgoing
// multicast packets.
func (c *dgramOpt) MulticastTTL() (int, error) {
	if !c.ok() {
		return 0, errInvalidConn
	}
	so, ok := sockOpts[ssoMulticastTTL]
	if !ok {
		return 0, errNotImplemented
	}
	return so.GetInt(c.Conn)
}

// SetMulticastTTL sets the time-to-live field value for future
// outgoing multicast packets.
func (c *dgramOpt) SetMulticastTTL(ttl int) error {
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoMulticastTTL]
	if !ok {
		return errNotImplemented
	}
	return so.SetInt(c.Conn, ttl)
}

// MulticastInterface returns the default interface for multicast
// packet transmissions.
func (c *dgramOpt) MulticastInterface() (*net.Interface, error) {
	if !c.ok() {
		return nil, errInvalidConn
	}
	so, ok := sockOpts[ssoMulticastInterface]
	if !ok {
		return nil, errNotImplemented
	}
	return so.getMulticastInterface(c.Conn)
}

// SetMulticastInterface sets the default interface for future
// multicast packet transmissions.
func (c *dgramOpt) SetMulticastInterface(ifi *net.Interface) error {
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoMulticastInterface]
	if !ok {
		return errNotImplemented
	}
	return so.setMulticastInterface(c.Conn, ifi)
}

// MulticastLoopback reports whether transmitted multicast packets
// should be copied and send back to the originator.
func (c *dgramOpt) MulticastLoopback() (bool, error) {
	if !c.ok() {
		return false, errInvalidConn
	}
	so, ok := sockOpts[ssoMulticastLoopback]
	if !ok {
		return false, errNotImplemented
	}
	on, err := so.GetInt(c.Conn)
	if err != nil {
		return false, err
	}
	return on == 1, nil
}

// SetMulticastLoopback sets whether transmitted multicast packets
// should be copied and send back to the originator.
func (c *dgramOpt) SetMulticastLoopback(on bool) error {
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoMulticastLoopback]
	if !ok {
		return errNotImplemented
	}
	return so.SetInt(c.Conn, boolint(on))
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
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoJoinGroup]
	if !ok {
		return errNotImplemented
	}
	grp := netAddrToIP4(group)
	if grp == nil {
		return errMissingAddress
	}
	return so.setGroup(c.Conn, ifi, grp)
}

// LeaveGroup leaves the group address group on the interface ifi
// regardless of whether the group is any-source group or
// source-specific group.
func (c *dgramOpt) LeaveGroup(ifi *net.Interface, group net.Addr) error {
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoLeaveGroup]
	if !ok {
		return errNotImplemented
	}
	grp := netAddrToIP4(group)
	if grp == nil {
		return errMissingAddress
	}
	return so.setGroup(c.Conn, ifi, grp)
}

// JoinSourceSpecificGroup joins the source-specific group comprising
// group and source on the interface ifi.
// JoinSourceSpecificGroup uses the system assigned multicast
// interface when ifi is nil, although this is not recommended because
// the assignment depends on platforms and sometimes it might require
// routing configuration.
func (c *dgramOpt) JoinSourceSpecificGroup(ifi *net.Interface, group, source net.Addr) error {
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoJoinSourceGroup]
	if !ok {
		return errNotImplemented
	}
	grp := netAddrToIP4(group)
	if grp == nil {
		return errMissingAddress
	}
	src := netAddrToIP4(source)
	if src == nil {
		return errMissingAddress
	}
	return so.setSourceGroup(c.Conn, ifi, grp, src)
}

// LeaveSourceSpecificGroup leaves the source-specific group on the
// interface ifi.
func (c *dgramOpt) LeaveSourceSpecificGroup(ifi *net.Interface, group, source net.Addr) error {
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoLeaveSourceGroup]
	if !ok {
		return errNotImplemented
	}
	grp := netAddrToIP4(group)
	if grp == nil {
		return errMissingAddress
	}
	src := netAddrToIP4(source)
	if src == nil {
		return errMissingAddress
	}
	return so.setSourceGroup(c.Conn, ifi, grp, src)
}

// ExcludeSourceSpecificGroup excludes the source-specific group from
// the already joined any-source groups by JoinGroup on the interface
// ifi.
func (c *dgramOpt) ExcludeSourceSpecificGroup(ifi *net.Interface, group, source net.Addr) error {
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoBlockSourceGroup]
	if !ok {
		return errNotImplemented
	}
	grp := netAddrToIP4(group)
	if grp == nil {
		return errMissingAddress
	}
	src := netAddrToIP4(source)
	if src == nil {
		return errMissingAddress
	}
	return so.setSourceGroup(c.Conn, ifi, grp, src)
}

// IncludeSourceSpecificGroup includes the excluded source-specific
// group by ExcludeSourceSpecificGroup again on the interface ifi.
func (c *dgramOpt) IncludeSourceSpecificGroup(ifi *net.Interface, group, source net.Addr) error {
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoUnblockSourceGroup]
	if !ok {
		return errNotImplemented
	}
	grp := netAddrToIP4(group)
	if grp == nil {
		return errMissingAddress
	}
	src := netAddrToIP4(source)
	if src == nil {
		return errMissingAddress
	}
	return so.setSourceGroup(c.Conn, ifi, grp, src)
}

// ICMPFilter returns an ICMP filter.
// Currently only Linux supports this.
func (c *dgramOpt) ICMPFilter() (*ICMPFilter, error) {
	if !c.ok() {
		return nil, errInvalidConn
	}
	so, ok := sockOpts[ssoICMPFilter]
	if !ok {
		return nil, errNotImplemented
	}
	return so.getICMPFilter(c.Conn)
}

// SetICMPFilter deploys the ICMP filter.
// Currently only Linux supports this.
func (c *dgramOpt) SetICMPFilter(f *ICMPFilter) error {
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoICMPFilter]
	if !ok {
		return errNotImplemented
	}
	return so.setICMPFilter(c.Conn, f)
}

// SetBPF attaches a BPF program to the connection.
//
// Only supported on Linux.
func (c *dgramOpt) SetBPF(filter []bpf.RawInstruction) error {
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoAttachFilter]
	if !ok {
		return errNotImplemented
	}
	return so.setBPF(c.Conn, filter)
}
