// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl plan9 solaris

package ipv6

import "net"

func getMTUInfo(fd int, opt *sockOpt) (*net.Interface, int, error) {
	return nil, 0, errOpNoSupport
}
