// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package system

import (
	"crypto/x509"
	"encoding/pem"
	"errors"
)

func appendPEM(roots []*x509.Certificate, pemCerts []byte) ([]*x509.Certificate, bool) {
	var ok bool

	for len(pemCerts) > 0 {
		var block *pem.Block
		block, pemCerts = pem.Decode(pemCerts)
		if block == nil {
			break
		}
		if block.Type != "CERTIFICATE" || len(block.Headers) != 0 {
			continue
		}

		cert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			continue
		}

		roots = append(roots, cert)
		ok = true
	}

	return roots, ok
}

// New returns a new certificate pool loaded with the system
// roots. The provided argument is not used; it is included for
// compatibility with other functions.
func New(metadata map[string]string) ([]*x509.Certificate, error) {
	roots := initSystemRoots()
	if len(roots) == 0 {
		return nil, errors.New("transport: unable to find system roots")
	}
	return roots, nil
}
