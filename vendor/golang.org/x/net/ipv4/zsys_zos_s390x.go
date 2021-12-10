// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Hand edited based on zerrors_zos_s390x.go
// TODO(Bill O'Farrell): auto-generate.

package ipv4

const (
	sizeofIPMreq          = 8
	sizeofSockaddrInet4   = 16
	sizeofSockaddrStorage = 128
	sizeofGroupReq        = 136
	sizeofGroupSourceReq  = 264
	sizeofInetPktinfo     = 8
)

type sockaddrInet4 struct {
	Len    uint8
	Family uint8
	Port   uint16
	Addr   [4]byte
	Zero   [8]uint8
}

type inetPktinfo struct {
	Addr    [4]byte
	Ifindex uint32
}

type sockaddrStorage struct {
	Len      uint8
	Family   byte
	ss_pad1  [6]byte
	ss_align int64
	ss_pad2  [112]byte
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

type ipMreq struct {
	Multiaddr [4]byte /* in_addr */
	Interface [4]byte /* in_addr */
}
