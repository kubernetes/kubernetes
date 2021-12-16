// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Hand edited based on zerrors_zos_s390x.go
// TODO(Bill O'Farrell): auto-generate.

package ipv6

const (
	sizeofSockaddrStorage = 128
	sizeofICMPv6Filter    = 32
	sizeofInet6Pktinfo    = 20
	sizeofIPv6Mtuinfo     = 32
	sizeofSockaddrInet6   = 28
	sizeofGroupReq        = 136
	sizeofGroupSourceReq  = 264
)

type sockaddrStorage struct {
	Len      uint8
	Family   byte
	ss_pad1  [6]byte
	ss_align int64
	ss_pad2  [112]byte
}

type sockaddrInet6 struct {
	Len      uint8
	Family   uint8
	Port     uint16
	Flowinfo uint32
	Addr     [16]byte
	Scope_id uint32
}

type inet6Pktinfo struct {
	Addr    [16]byte
	Ifindex uint32
}

type ipv6Mtuinfo struct {
	Addr sockaddrInet6
	Mtu  uint32
}

type groupReq struct {
	Interface uint32
	reserved  uint32
	Group     sockaddrStorage
}

type groupSourceReq struct {
	Interface uint32
	reserved  uint32
	Group     sockaddrStorage
	Source    sockaddrStorage
}

type icmpv6Filter struct {
	Filt [8]uint32
}
