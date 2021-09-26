// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

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
			name: "AF_UNIX zero length",
			rsa: sockaddrUnixToAny(RawSockaddrUnix{
				Family: AF_UNIX,
			}),
			err: EINVAL,
		},
		{
			name: "AF_UNIX unnamed",
			rsa: sockaddrUnixToAny(RawSockaddrUnix{
				Len:    2, // family (uint16)
				Family: AF_UNIX,
			}),
			sa: &SockaddrUnix{},
		},
		{
			name: "AF_UNIX named",
			rsa: sockaddrUnixToAny(RawSockaddrUnix{
				Len:    uint8(2 + len("gopher")), // family (uint16) + len(gopher)
				Family: AF_UNIX,
				Path:   [104]int8{'g', 'o', 'p', 'h', 'e', 'r'},
			}),
			sa: &SockaddrUnix{
				Name: "gopher",
			},
		},
		{
			name: "AF_UNIX named",
			rsa: sockaddrUnixToAny(RawSockaddrUnix{
				Len:    uint8(2 + len("go")),
				Family: AF_UNIX,
				Path:   [104]int8{'g', 'o', 'p', 'h', 'e', 'r'},
			}),
			sa: &SockaddrUnix{
				Name: "go",
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

func TestSockaddrUnix_sockaddr(t *testing.T) {
	tests := []struct {
		name string
		sa   *SockaddrUnix
		raw  *RawSockaddrUnix
		err  error
	}{
		{
			name: "unnamed",
			sa:   &SockaddrUnix{},
			raw: &RawSockaddrUnix{
				Family: AF_UNIX,
			},
			err: EINVAL,
		},
		{
			name: "named",
			sa: &SockaddrUnix{
				Name: "gopher",
			},
			raw: &RawSockaddrUnix{
				Len:    uint8(2 + len("gopher") + 1), // family (uint16) + len(gopher) + '\0'
				Family: AF_UNIX,
				Path:   [104]int8{'g', 'o', 'p', 'h', 'e', 'r'},
			},
		},
		{
			name: "named too long",
			sa: &SockaddrUnix{
				Name: strings.Repeat("A", 104),
			},
			err: EINVAL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, _, err := tt.sa.sockaddr()
			if err != tt.err {
				t.Fatalf("unexpected error: %v, want: %v", err, tt.err)
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

func sockaddrUnixToAny(in RawSockaddrUnix) *RawSockaddrAny {
	var out RawSockaddrAny

	// Explicitly copy the contents of in into out to produce the correct
	// sockaddr structure, without relying on unsafe casting to a type of a
	// larger size.
	copy(
		(*(*[SizeofSockaddrAny]byte)(unsafe.Pointer(&out)))[:],
		(*(*[SizeofSockaddrUnix]byte)(unsafe.Pointer(&in)))[:],
	)

	return &out
}
