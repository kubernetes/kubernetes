// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sockstest

import (
	"net"
	"reflect"
	"testing"

	"golang.org/x/net/internal/socks"
)

func TestParseAuthRequest(t *testing.T) {
	for i, tt := range []struct {
		wire []byte
		req  *AuthRequest
	}{
		{
			[]byte{0x05, 0x00},
			&AuthRequest{
				socks.Version5,
				nil,
			},
		},
		{
			[]byte{0x05, 0x01, 0xff},
			&AuthRequest{
				socks.Version5,
				[]socks.AuthMethod{
					socks.AuthMethodNoAcceptableMethods,
				},
			},
		},
		{
			[]byte{0x05, 0x02, 0x00, 0xff},
			&AuthRequest{
				socks.Version5,
				[]socks.AuthMethod{
					socks.AuthMethodNotRequired,
					socks.AuthMethodNoAcceptableMethods,
				},
			},
		},

		// corrupted requests
		{nil, nil},
		{[]byte{0x00, 0x01}, nil},
		{[]byte{0x06, 0x00}, nil},
		{[]byte{0x05, 0x02, 0x00}, nil},
	} {
		req, err := ParseAuthRequest(tt.wire)
		if !reflect.DeepEqual(req, tt.req) {
			t.Errorf("#%d: got %v, %v; want %v", i, req, err, tt.req)
			continue
		}
	}
}

func TestParseCmdRequest(t *testing.T) {
	for i, tt := range []struct {
		wire []byte
		req  *CmdRequest
	}{
		{
			[]byte{0x05, 0x01, 0x00, 0x01, 192, 0, 2, 1, 0x17, 0x4b},
			&CmdRequest{
				socks.Version5,
				socks.CmdConnect,
				socks.Addr{
					IP:   net.IP{192, 0, 2, 1},
					Port: 5963,
				},
			},
		},
		{
			[]byte{0x05, 0x01, 0x00, 0x03, 0x04, 'F', 'Q', 'D', 'N', 0x17, 0x4b},
			&CmdRequest{
				socks.Version5,
				socks.CmdConnect,
				socks.Addr{
					Name: "FQDN",
					Port: 5963,
				},
			},
		},

		// corrupted requests
		{nil, nil},
		{[]byte{0x05}, nil},
		{[]byte{0x06, 0x01, 0x00, 0x01, 192, 0, 2, 2, 0x17, 0x4b}, nil},
		{[]byte{0x05, 0x01, 0xff, 0x01, 192, 0, 2, 3}, nil},
		{[]byte{0x05, 0x01, 0x00, 0x01, 192, 0, 2, 4}, nil},
		{[]byte{0x05, 0x01, 0x00, 0x03, 0x04, 'F', 'Q', 'D', 'N'}, nil},
	} {
		req, err := ParseCmdRequest(tt.wire)
		if !reflect.DeepEqual(req, tt.req) {
			t.Errorf("#%d: got %v, %v; want %v", i, req, err, tt.req)
			continue
		}
	}
}
