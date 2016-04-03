// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl plan9 solaris

package ipv6

type sysSockoptLen int32

var (
	ctlOpts = [ctlMax]ctlOpt{}

	sockOpts = [ssoMax]sockOpt{}
)
