// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.6

package http2

import (
	"crypto/tls"
	"net/http"
	"time"
)

func transportExpectContinueTimeout(t1 *http.Transport) time.Duration {
	return t1.ExpectContinueTimeout
}

// isBadCipher reports whether the cipher is blacklisted by the HTTP/2 spec.
func isBadCipher(cipher uint16) bool {
	switch cipher {
	case tls.TLS_RSA_WITH_RC4_128_SHA,
		tls.TLS_RSA_WITH_3DES_EDE_CBC_SHA,
		tls.TLS_RSA_WITH_AES_128_CBC_SHA,
		tls.TLS_RSA_WITH_AES_256_CBC_SHA,
		tls.TLS_RSA_WITH_AES_128_GCM_SHA256,
		tls.TLS_RSA_WITH_AES_256_GCM_SHA384,
		tls.TLS_ECDHE_ECDSA_WITH_RC4_128_SHA,
		tls.TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA,
		tls.TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA,
		tls.TLS_ECDHE_RSA_WITH_RC4_128_SHA,
		tls.TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA,
		tls.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA,
		tls.TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA:
		// Reject cipher suites from Appendix A.
		// "This list includes those cipher suites that do not
		// offer an ephemeral key exchange and those that are
		// based on the TLS null, stream or block cipher type"
		return true
	default:
		return false
	}
}
