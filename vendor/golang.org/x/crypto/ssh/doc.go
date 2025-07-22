// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package ssh implements an SSH client and server.

SSH is a transport security protocol, an authentication protocol and a
family of application protocols. The most typical application level
protocol is a remote shell and this is specifically implemented.  However,
the multiplexed nature of SSH is exposed to users that wish to support
others.

References:

	[PROTOCOL]: https://cvsweb.openbsd.org/cgi-bin/cvsweb/src/usr.bin/ssh/PROTOCOL?rev=HEAD
	[PROTOCOL.certkeys]: http://cvsweb.openbsd.org/cgi-bin/cvsweb/src/usr.bin/ssh/PROTOCOL.certkeys?rev=HEAD
	[SSH-PARAMETERS]:    http://www.iana.org/assignments/ssh-parameters/ssh-parameters.xml#ssh-parameters-1

This package does not fall under the stability promise of the Go language itself,
so its API may be changed when pressing needs arise.
*/
package ssh
