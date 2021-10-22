// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package unix

import (
	"reflect"
	"strings"
	"testing"
	"unsafe"
)

// as per socket(2)
type SocketSpec struct {
	domain   int
	typ      int
	protocol int
}

func Test_anyToSockaddr(t *testing.T) {
	tests := []struct {
		name string
		rsa  *RawSockaddrAny
		sa   Sockaddr
		err  error
		skt  SocketSpec
	}{
		{
			name: "AF_TIPC bad addrtype",
			rsa: &RawSockaddrAny{
				Addr: RawSockaddr{
					Family: AF_TIPC,
				},
			},
			err: EINVAL,
		},
		{
			name: "AF_TIPC NameSeq",
			rsa: sockaddrTIPCToAny(RawSockaddrTIPC{
				Family:   AF_TIPC,
				Addrtype: TIPC_SERVICE_RANGE,
				Scope:    1,
				Addr: (&TIPCServiceRange{
					Type:  1,
					Lower: 2,
					Upper: 3,
				}).tipcAddr(),
			}),
			sa: &SockaddrTIPC{
				Scope: 1,
				Addr: &TIPCServiceRange{
					Type:  1,
					Lower: 2,
					Upper: 3,
				},
			},
		},
		{
			name: "AF_TIPC Name",
			rsa: sockaddrTIPCToAny(RawSockaddrTIPC{
				Family:   AF_TIPC,
				Addrtype: TIPC_SERVICE_ADDR,
				Scope:    2,
				Addr: (&TIPCServiceName{
					Type:     1,
					Instance: 2,
					Domain:   3,
				}).tipcAddr(),
			}),
			sa: &SockaddrTIPC{
				Scope: 2,
				Addr: &TIPCServiceName{
					Type:     1,
					Instance: 2,
					Domain:   3,
				},
			},
		},
		{
			name: "AF_TIPC ID",
			rsa: sockaddrTIPCToAny(RawSockaddrTIPC{
				Family:   AF_TIPC,
				Addrtype: TIPC_SOCKET_ADDR,
				Scope:    3,
				Addr: (&TIPCSocketAddr{
					Ref:  1,
					Node: 2,
				}).tipcAddr(),
			}),
			sa: &SockaddrTIPC{
				Scope: 3,
				Addr: &TIPCSocketAddr{
					Ref:  1,
					Node: 2,
				},
			},
		},
		{
			name: "AF_INET IPPROTO_L2TP",
			rsa: sockaddrL2TPIPToAny(RawSockaddrL2TPIP{
				Family:  AF_INET,
				Addr:    [4]byte{0xef, 0x10, 0x5b, 0xa2},
				Conn_id: 0x1234abcd,
			}),
			sa: &SockaddrL2TPIP{
				Addr:   [4]byte{0xef, 0x10, 0x5b, 0xa2},
				ConnId: 0x1234abcd,
			},
			skt: SocketSpec{domain: AF_INET, typ: SOCK_DGRAM, protocol: IPPROTO_L2TP},
		},
		{
			name: "AF_INET6 IPPROTO_L2TP",
			rsa: sockaddrL2TPIP6ToAny(RawSockaddrL2TPIP6{
				Family:   AF_INET6,
				Flowinfo: 42,
				Addr: [16]byte{
					0x20, 0x01, 0x0d, 0xb8,
					0x85, 0xa3, 0x00, 0x00,
					0x00, 0x00, 0x8a, 0x2e,
					0x03, 0x70, 0x73, 0x34,
				},
				Scope_id: 90210,
				Conn_id:  0x1234abcd,
			}),
			sa: &SockaddrL2TPIP6{
				Addr: [16]byte{
					0x20, 0x01, 0x0d, 0xb8,
					0x85, 0xa3, 0x00, 0x00,
					0x00, 0x00, 0x8a, 0x2e,
					0x03, 0x70, 0x73, 0x34,
				},
				ZoneId: 90210,
				ConnId: 0x1234abcd,
			},
			skt: SocketSpec{domain: AF_INET6, typ: SOCK_DGRAM, protocol: IPPROTO_L2TP},
		},
		{
			name: "AF_UNIX unnamed/abstract",
			rsa: sockaddrUnixToAny(RawSockaddrUnix{
				Family: AF_UNIX,
			}),
			sa: &SockaddrUnix{
				Name: "@",
			},
		},
		{
			name: "AF_UNIX named",
			rsa: sockaddrUnixToAny(RawSockaddrUnix{
				Family: AF_UNIX,
				Path:   [108]int8{'g', 'o', 'p', 'h', 'e', 'r'},
			}),
			sa: &SockaddrUnix{
				Name: "gopher",
			},
		},
		{
			name: "AF_IUCV",
			rsa: sockaddrIUCVToAny(RawSockaddrIUCV{
				Family:  AF_IUCV,
				User_id: [8]int8{'*', 'M', 'S', 'G', ' ', ' ', ' ', ' '},
				Name:    [8]int8{' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '},
			}),
			sa: &SockaddrIUCV{
				UserID: "*MSG    ",
				Name:   "        ",
			},
		},
		{
			name: "AF_CAN",
			rsa: sockaddrCANToAny(RawSockaddrCAN{
				Family:  AF_CAN,
				Ifindex: 12345678,
				Addr: [16]byte{
					0xAA, 0xAA, 0xAA, 0xAA,
					0xBB, 0xBB, 0xBB, 0xBB,
					0x0, 0x0, 0x0, 0x0,
					0x0, 0x0, 0x0, 0x0,
				},
			}),
			sa: &SockaddrCAN{
				Ifindex: 12345678,
				RxID:    0xAAAAAAAA,
				TxID:    0xBBBBBBBB,
			},
		},
		{
			name: "AF_MAX EAFNOSUPPORT",
			rsa: &RawSockaddrAny{
				Addr: RawSockaddr{
					Family: AF_MAX,
				},
			},
			err: EAFNOSUPPORT,
		},
		// TODO: expand to support other families.
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fd := int(0)
			var err error
			if tt.skt.domain != 0 {
				fd, err = Socket(tt.skt.domain, tt.skt.typ, tt.skt.protocol)
				// Some sockaddr types need specific kernel modules running: if these
				// are not present we'll get EPROTONOSUPPORT back when trying to create
				// the socket.  Skip the test in this situation.
				if err == EPROTONOSUPPORT {
					t.Skip("socket family/protocol not supported by kernel")
				} else if err != nil {
					t.Fatalf("socket(%v): %v", tt.skt, err)
				}
				defer Close(fd)
			}
			sa, err := anyToSockaddr(fd, tt.rsa)
			if err != tt.err {
				t.Fatalf("unexpected error: %v, want: %v", err, tt.err)
			}

			if !reflect.DeepEqual(sa, tt.sa) {
				t.Fatalf("unexpected Sockaddr:\n got: %#v\nwant: %#v", sa, tt.sa)
			}
		})
	}
}

func TestSockaddrTIPC_sockaddr(t *testing.T) {
	tests := []struct {
		name string
		sa   *SockaddrTIPC
		raw  *RawSockaddrTIPC
		err  error
	}{
		{
			name: "no fields set",
			sa:   &SockaddrTIPC{},
			err:  EINVAL,
		},
		{
			name: "ID",
			sa: &SockaddrTIPC{
				Scope: 1,
				Addr: &TIPCSocketAddr{
					Ref:  1,
					Node: 2,
				},
			},
			raw: &RawSockaddrTIPC{
				Family:   AF_TIPC,
				Addrtype: TIPC_SOCKET_ADDR,
				Scope:    1,
				Addr: (&TIPCSocketAddr{
					Ref:  1,
					Node: 2,
				}).tipcAddr(),
			},
		},
		{
			name: "NameSeq",
			sa: &SockaddrTIPC{
				Scope: 2,
				Addr: &TIPCServiceRange{
					Type:  1,
					Lower: 2,
					Upper: 3,
				},
			},
			raw: &RawSockaddrTIPC{
				Family:   AF_TIPC,
				Addrtype: TIPC_SERVICE_RANGE,
				Scope:    2,
				Addr: (&TIPCServiceRange{
					Type:  1,
					Lower: 2,
					Upper: 3,
				}).tipcAddr(),
			},
		},
		{
			name: "Name",
			sa: &SockaddrTIPC{
				Scope: 3,
				Addr: &TIPCServiceName{
					Type:     1,
					Instance: 2,
					Domain:   3,
				},
			},
			raw: &RawSockaddrTIPC{
				Family:   AF_TIPC,
				Addrtype: TIPC_SERVICE_ADDR,
				Scope:    3,
				Addr: (&TIPCServiceName{
					Type:     1,
					Instance: 2,
					Domain:   3,
				}).tipcAddr(),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, l, err := tt.sa.sockaddr()
			if err != tt.err {
				t.Fatalf("unexpected error: %v, want: %v", err, tt.err)
			}

			// Must be 0 on error or a fixed size otherwise.
			if (tt.err != nil && l != 0) || (tt.raw != nil && l != SizeofSockaddrTIPC) {
				t.Fatalf("unexpected Socklen: %d", l)
			}
			if out == nil {
				// No pointer to cast, return early.
				return
			}

			raw := (*RawSockaddrTIPC)(out)
			if !reflect.DeepEqual(raw, tt.raw) {
				t.Fatalf("unexpected RawSockaddrTIPC:\n got: %#v\nwant: %#v", raw, tt.raw)
			}
		})
	}
}

func TestSockaddrL2TPIP_sockaddr(t *testing.T) {
	tests := []struct {
		name string
		sa   *SockaddrL2TPIP
		raw  *RawSockaddrL2TPIP
		err  error
	}{
		{
			name: "L2TPIP",
			sa: &SockaddrL2TPIP{
				Addr:   [4]byte{0xef, 0x10, 0x5b, 0xa2},
				ConnId: 0x1234abcd,
			},
			raw: &RawSockaddrL2TPIP{
				Family:  AF_INET,
				Addr:    [4]byte{0xef, 0x10, 0x5b, 0xa2},
				Conn_id: 0x1234abcd,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, l, err := tt.sa.sockaddr()
			if err != tt.err {
				t.Fatalf("unexpected error: %v, want: %v", err, tt.err)
			}

			// Must be 0 on error or a fixed size otherwise.
			if (tt.err != nil && l != 0) || (tt.raw != nil && l != SizeofSockaddrL2TPIP) {
				t.Fatalf("unexpected Socklen: %d", l)
			}

			if out != nil {
				raw := (*RawSockaddrL2TPIP)(out)
				if !reflect.DeepEqual(raw, tt.raw) {
					t.Fatalf("unexpected RawSockaddrL2TPIP:\n got: %#v\nwant: %#v", raw, tt.raw)
				}
			}
		})
	}
}

func TestSockaddrL2TPIP6_sockaddr(t *testing.T) {
	tests := []struct {
		name string
		sa   *SockaddrL2TPIP6
		raw  *RawSockaddrL2TPIP6
		err  error
	}{
		{
			name: "L2TPIP6",
			sa: &SockaddrL2TPIP6{
				Addr: [16]byte{
					0x20, 0x01, 0x0d, 0xb8,
					0x85, 0xa3, 0x00, 0x00,
					0x00, 0x00, 0x8a, 0x2e,
					0x03, 0x70, 0x73, 0x34,
				},
				ZoneId: 90210,
				ConnId: 0x1234abcd,
			},
			raw: &RawSockaddrL2TPIP6{
				Family: AF_INET6,
				Addr: [16]byte{
					0x20, 0x01, 0x0d, 0xb8,
					0x85, 0xa3, 0x00, 0x00,
					0x00, 0x00, 0x8a, 0x2e,
					0x03, 0x70, 0x73, 0x34,
				},
				Scope_id: 90210,
				Conn_id:  0x1234abcd,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, l, err := tt.sa.sockaddr()
			if err != tt.err {
				t.Fatalf("unexpected error: %v, want: %v", err, tt.err)
			}

			// Must be 0 on error or a fixed size otherwise.
			if (tt.err != nil && l != 0) || (tt.raw != nil && l != SizeofSockaddrL2TPIP6) {
				t.Fatalf("unexpected Socklen: %d", l)
			}

			if out != nil {
				raw := (*RawSockaddrL2TPIP6)(out)
				if !reflect.DeepEqual(raw, tt.raw) {
					t.Fatalf("unexpected RawSockaddrL2TPIP6:\n got: %#v\nwant: %#v", raw, tt.raw)
				}
			}
		})
	}
}

func TestSockaddrUnix_sockaddr(t *testing.T) {
	tests := []struct {
		name string
		sa   *SockaddrUnix
		raw  *RawSockaddrUnix
		slen _Socklen
		err  error
	}{
		{
			name: "unnamed",
			sa:   &SockaddrUnix{},
			raw: &RawSockaddrUnix{
				Family: AF_UNIX,
			},
			slen: 2, // family (uint16)
		},
		{
			name: "abstract",
			sa: &SockaddrUnix{
				Name: "@",
			},
			raw: &RawSockaddrUnix{
				Family: AF_UNIX,
			},
			slen: 3, // family (uint16) + NULL
		},
		{
			name: "named",
			sa: &SockaddrUnix{
				Name: "gopher",
			},
			raw: &RawSockaddrUnix{
				Family: AF_UNIX,
				Path:   [108]int8{'g', 'o', 'p', 'h', 'e', 'r'},
			},
			slen: _Socklen(3 + len("gopher")), // family (uint16) + len(gopher)
		},
		{
			name: "named too long",
			sa: &SockaddrUnix{
				Name: strings.Repeat("A", 108),
			},
			err: EINVAL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, l, err := tt.sa.sockaddr()
			if err != tt.err {
				t.Fatalf("unexpected error: %v, want: %v", err, tt.err)
			}

			if l != tt.slen {
				t.Fatalf("unexpected Socklen: %d, want %d", l, tt.slen)
			}
			if out == nil {
				// No pointer to cast, return early.
				return
			}

			raw := (*RawSockaddrUnix)(out)
			if !reflect.DeepEqual(raw, tt.raw) {
				t.Fatalf("unexpected RawSockaddrUnix:\n got: %#v\nwant: %#v", raw, tt.raw)
			}
		})
	}
}

func TestSockaddrIUCV_sockaddr(t *testing.T) {
	tests := []struct {
		name string
		sa   *SockaddrIUCV
		raw  *RawSockaddrIUCV
		err  error
	}{
		{
			name: "no fields set",
			sa:   &SockaddrIUCV{},
			raw: &RawSockaddrIUCV{
				Family:  AF_IUCV,
				Nodeid:  [8]int8{' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '},
				User_id: [8]int8{' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '},
				Name:    [8]int8{' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '},
			},
		},
		{
			name: "both fields set",
			sa: &SockaddrIUCV{
				UserID: "USERID",
				Name:   "NAME",
			},
			raw: &RawSockaddrIUCV{
				Family:  AF_IUCV,
				Nodeid:  [8]int8{' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '},
				User_id: [8]int8{'U', 'S', 'E', 'R', 'I', 'D', ' ', ' '},
				Name:    [8]int8{'N', 'A', 'M', 'E', ' ', ' ', ' ', ' '},
			},
		},
		{
			name: "too long userid",
			sa: &SockaddrIUCV{
				UserID: "123456789",
			},
			err: EINVAL,
		},
		{
			name: "too long name",
			sa: &SockaddrIUCV{
				Name: "123456789",
			},
			err: EINVAL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, l, err := tt.sa.sockaddr()
			if err != tt.err {
				t.Fatalf("unexpected error: %v, want: %v", err, tt.err)
			}

			// Must be 0 on error or a fixed size otherwise.
			if (tt.err != nil && l != 0) || (tt.raw != nil && l != SizeofSockaddrIUCV) {
				t.Fatalf("unexpected Socklen: %d", l)
			}
			if out == nil {
				// No pointer to cast, return early.
				return
			}

			raw := (*RawSockaddrIUCV)(out)
			if !reflect.DeepEqual(raw, tt.raw) {
				t.Fatalf("unexpected RawSockaddrIUCV:\n got: %#v\nwant: %#v", raw, tt.raw)
			}
		})
	}
}

func TestSockaddrCAN_sockaddr(t *testing.T) {
	tests := []struct {
		name string
		sa   *SockaddrCAN
		raw  *RawSockaddrCAN
		err  error
	}{
		{
			name: "with ids",
			sa: &SockaddrCAN{
				Ifindex: 12345678,
				RxID:    0xAAAAAAAA,
				TxID:    0xBBBBBBBB,
			},
			raw: &RawSockaddrCAN{
				Family:  AF_CAN,
				Ifindex: 12345678,
				Addr: [16]byte{
					0xAA, 0xAA, 0xAA, 0xAA,
					0xBB, 0xBB, 0xBB, 0xBB,
					0x0, 0x0, 0x0, 0x0,
					0x0, 0x0, 0x0, 0x0,
				},
			},
		},
		{
			name: "negative ifindex",
			sa: &SockaddrCAN{
				Ifindex: -1,
			},
			err: EINVAL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, l, err := tt.sa.sockaddr()
			if err != tt.err {
				t.Fatalf("unexpected error: %v, want: %v", err, tt.err)
			}

			// Must be 0 on error or a fixed size otherwise.
			if (tt.err != nil && l != 0) || (tt.raw != nil && l != SizeofSockaddrCAN) {
				t.Fatalf("unexpected Socklen: %d", l)
			}

			if out != nil {
				raw := (*RawSockaddrCAN)(out)
				if !reflect.DeepEqual(raw, tt.raw) {
					t.Fatalf("unexpected RawSockaddrCAN:\n got: %#v\nwant: %#v", raw, tt.raw)
				}
			}
		})
	}
}

// These helpers explicitly copy the contents of in into out to produce
// the correct sockaddr structure, without relying on unsafe casting to
// a type of a larger size.
func sockaddrTIPCToAny(in RawSockaddrTIPC) *RawSockaddrAny {
	var out RawSockaddrAny
	copy(
		(*(*[SizeofSockaddrAny]byte)(unsafe.Pointer(&out)))[:],
		(*(*[SizeofSockaddrTIPC]byte)(unsafe.Pointer(&in)))[:],
	)
	return &out
}

func sockaddrL2TPIPToAny(in RawSockaddrL2TPIP) *RawSockaddrAny {
	var out RawSockaddrAny
	copy(
		(*(*[SizeofSockaddrAny]byte)(unsafe.Pointer(&out)))[:],
		(*(*[SizeofSockaddrL2TPIP]byte)(unsafe.Pointer(&in)))[:],
	)
	return &out
}

func sockaddrL2TPIP6ToAny(in RawSockaddrL2TPIP6) *RawSockaddrAny {
	var out RawSockaddrAny
	copy(
		(*(*[SizeofSockaddrAny]byte)(unsafe.Pointer(&out)))[:],
		(*(*[SizeofSockaddrL2TPIP6]byte)(unsafe.Pointer(&in)))[:],
	)
	return &out
}

func sockaddrUnixToAny(in RawSockaddrUnix) *RawSockaddrAny {
	var out RawSockaddrAny
	copy(
		(*(*[SizeofSockaddrAny]byte)(unsafe.Pointer(&out)))[:],
		(*(*[SizeofSockaddrUnix]byte)(unsafe.Pointer(&in)))[:],
	)
	return &out
}

func sockaddrIUCVToAny(in RawSockaddrIUCV) *RawSockaddrAny {
	var out RawSockaddrAny
	copy(
		(*(*[SizeofSockaddrAny]byte)(unsafe.Pointer(&out)))[:],
		(*(*[SizeofSockaddrUnix]byte)(unsafe.Pointer(&in)))[:],
	)
	return &out
}

func sockaddrCANToAny(in RawSockaddrCAN) *RawSockaddrAny {
	var out RawSockaddrAny
	copy(
		(*(*[SizeofSockaddrAny]byte)(unsafe.Pointer(&out)))[:],
		(*(*[SizeofSockaddrCAN]byte)(unsafe.Pointer(&in)))[:],
	)
	return &out
}
