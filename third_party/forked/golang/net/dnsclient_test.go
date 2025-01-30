// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package is copied from Go library net.
// https://golang.org/src/net/dnsclient.go
// The original private function reverseaddr
// is exported as public function.

package net

import (
	"net"
	"testing"
)

func TestReverseaddr(t *testing.T) {
	var revAddrTests = []struct {
		Addr      string
		Reverse   string
		ErrPrefix string
	}{
		{"1.2.3.4", "4.3.2.1.in-addr.arpa.", ""},
		{"245.110.36.114", "114.36.110.245.in-addr.arpa.", ""},
		{"::ffff:12.34.56.78", "78.56.34.12.in-addr.arpa.", ""},
		{"::1", "1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.ip6.arpa.", ""},
		{"1::", "0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.1.0.0.0.ip6.arpa.", ""},
		{"1234:567::89a:bcde", "e.d.c.b.a.9.8.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.7.6.5.0.4.3.2.1.ip6.arpa.", ""},
		{"1234:567:fefe:bcbc:adad:9e4a:89a:bcde", "e.d.c.b.a.9.8.0.a.4.e.9.d.a.d.a.c.b.c.b.e.f.e.f.7.6.5.0.4.3.2.1.ip6.arpa.", ""},
		{"1.2.3", "", "unrecognized address"},
		{"1.2.3.4.5", "", "unrecognized address"},
		{"1234:567:bcbca::89a:bcde", "", "unrecognized address"},
		{"1234:567::bcbc:adad::89a:bcde", "", "unrecognized address"},
	}
	for i, tt := range revAddrTests {
		a, err := Reverseaddr(tt.Addr)
		if len(tt.ErrPrefix) > 0 && err == nil {
			t.Errorf("#%d: expected %q, got <nil> (error)", i, tt.ErrPrefix)
			continue
		}
		if len(tt.ErrPrefix) == 0 && err != nil {
			t.Errorf("#%d: expected <nil>, got %q (error)", i, err)
		}
		if err != nil && err.(*net.DNSError).Err != tt.ErrPrefix {
			t.Errorf("#%d: expected %q, got %q (mismatched error)", i, tt.ErrPrefix, err.(*net.DNSError).Err)
		}
		if a != tt.Reverse {
			t.Errorf("#%d: expected %q, got %q (reverse address)", i, tt.Reverse, a)
		}
	}
}
