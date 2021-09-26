// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd

package test

import (
	"testing"
)

func TestBannerCallbackAgainstOpenSSH(t *testing.T) {
	server := newServer(t)
	defer server.Shutdown()

	clientConf := clientConfig()

	var receivedBanner string
	clientConf.BannerCallback = func(message string) error {
		receivedBanner = message
		return nil
	}

	conn := server.Dial(clientConf)
	defer conn.Close()

	expected := "Server Banner"
	if receivedBanner != expected {
		t.Fatalf("got %v; want %v", receivedBanner, expected)
	}
}
