// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"reflect"
	"testing"
	"unsafe"
)

func Test_anyToSockaddr_darwin(t *testing.T) {
	tests := []struct {
		name string
		rsa  *RawSockaddrAny
		sa   Sockaddr
		err  error
	}{
		{
			name: "AF_SYSTEM emtpy",
			rsa:  sockaddrCtlToAny(RawSockaddrCtl{}),
			err:  EAFNOSUPPORT,
		},
		{
			name: "AF_SYSTEM no sysaddr",
			rsa: sockaddrCtlToAny(RawSockaddrCtl{
				Sc_family: AF_SYSTEM,
			}),
			err: EAFNOSUPPORT,
		},
		{
			name: "AF_SYSTEM/AF_SYS_CONTROL empty ",
			rsa: sockaddrCtlToAny(RawSockaddrCtl{
				Sc_family:  AF_SYSTEM,
				Ss_sysaddr: AF_SYS_CONTROL,
			}),
			sa: &SockaddrCtl{},
		},
		{
			name: "AF_SYSTEM ID and unit",
			rsa: sockaddrCtlToAny(RawSockaddrCtl{
				Sc_family:  AF_SYSTEM,
				Ss_sysaddr: AF_SYS_CONTROL,
				Sc_id:      0x42,
				Sc_unit:    0xC71,
			}),
			sa: &SockaddrCtl{
				ID:   0x42,
				Unit: 0xC71,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fd := int(0)
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

func TestSockaddrCtl_sockaddr(t *testing.T) {
	tests := []struct {
		name string
		sa   *SockaddrCtl
		raw  *RawSockaddrCtl
		err  error
	}{
		{
			name: "empty",
			sa:   &SockaddrCtl{},
			raw: &RawSockaddrCtl{
				Sc_len:     SizeofSockaddrCtl,
				Sc_family:  AF_SYSTEM,
				Ss_sysaddr: AF_SYS_CONTROL,
			},
		},
		{
			name: "with ID and unit",
			sa: &SockaddrCtl{
				ID:   0x42,
				Unit: 0xff,
			},
			raw: &RawSockaddrCtl{
				Sc_len:     SizeofSockaddrCtl,
				Sc_family:  AF_SYSTEM,
				Ss_sysaddr: AF_SYS_CONTROL,
				Sc_id:      0x42,
				Sc_unit:    0xff,
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
			if (tt.err != nil && l != 0) || (tt.raw != nil && l != SizeofSockaddrCtl) {
				t.Fatalf("unexpected Socklen: %d", l)
			}

			if out != nil {
				raw := (*RawSockaddrCtl)(out)
				if !reflect.DeepEqual(raw, tt.raw) {
					t.Fatalf("unexpected RawSockaddrCtl:\n got: %#v\nwant: %#v", raw, tt.raw)
				}
			}
		})
	}
}

func sockaddrCtlToAny(in RawSockaddrCtl) *RawSockaddrAny {
	var out RawSockaddrAny
	copy(
		(*(*[SizeofSockaddrAny]byte)(unsafe.Pointer(&out)))[:],
		(*(*[SizeofSockaddrCtl]byte)(unsafe.Pointer(&in)))[:],
	)
	return &out
}
