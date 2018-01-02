// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run root_darwin_arm_gen.go -output root_darwin_armx.go

package system

import (
	"crypto/x509"
	"errors"
	"os/exec"
)

func execSecurityRoots() ([]*x509.Certificate, error) {
	cmd := exec.Command("/usr/bin/security", "find-certificate", "-a", "-p", "/System/Library/Keychains/SystemRootCertificates.keychain")
	data, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	roots, ok := appendPEM(nil, data)
	if !ok {
		return nil, errors.New("transport: no system roots found")
	}
	return roots, nil
}
