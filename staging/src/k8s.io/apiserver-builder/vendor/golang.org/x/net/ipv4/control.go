// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

import (
	"fmt"
	"net"
	"sync"
)

type rawOpt struct {
	sync.RWMutex
	cflags ControlFlags
}

func (c *rawOpt) set(f ControlFlags)        { c.cflags |= f }
func (c *rawOpt) clear(f ControlFlags)      { c.cflags &^= f }
func (c *rawOpt) isset(f ControlFlags) bool { return c.cflags&f != 0 }

type ControlFlags uint

const (
	FlagTTL       ControlFlags = 1 << iota // pass the TTL on the received packet
	FlagSrc                                // pass the source address on the received packet
	FlagDst                                // pass the destination address on the received packet
	FlagInterface                          // pass the interface index on the received packet
)

// A ControlMessage represents per packet basis IP-level socket options.
type ControlMessage struct {
	// Receiving socket options: SetControlMessage allows to
	// receive the options from the protocol stack using ReadFrom
	// method of PacketConn or RawConn.
	//
	// Specifying socket options: ControlMessage for WriteTo
	// method of PacketConn or RawConn allows to send the options
	// to the protocol stack.
	//
	TTL     int    // time-to-live, receiving only
	Src     net.IP // source address, specifying only
	Dst     net.IP // destination address, receiving only
	IfIndex int    // interface index, must be 1 <= value when specifying
}

func (cm *ControlMessage) String() string {
	if cm == nil {
		return "<nil>"
	}
	return fmt.Sprintf("ttl=%d src=%v dst=%v ifindex=%d", cm.TTL, cm.Src, cm.Dst, cm.IfIndex)
}

// Ancillary data socket options
const (
	ctlTTL        = iota // header field
	ctlSrc               // header field
	ctlDst               // header field
	ctlInterface         // inbound or outbound interface
	ctlPacketInfo        // inbound or outbound packet path
	ctlMax
)

// A ctlOpt represents a binding for ancillary data socket option.
type ctlOpt struct {
	name    int // option name, must be equal or greater than 1
	length  int // option length
	marshal func([]byte, *ControlMessage) []byte
	parse   func(*ControlMessage, []byte)
}
