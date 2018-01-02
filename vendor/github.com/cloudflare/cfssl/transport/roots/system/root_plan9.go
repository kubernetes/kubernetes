// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build plan9

package system

import (
	"crypto/x509"
	"io/ioutil"
)

// Possible certificate files; stop after finding one.
var certFiles = []string{
	"/sys/lib/tls/ca.pem",
}

func initSystemRoots() (roots []*x509.Certificate) {
	for _, file := range certFiles {
		data, err := ioutil.ReadFile(file)
		if err == nil {
			roots, _ = appendPEM(roots, data)
			return
		}
	}

	// All of the files failed to load. systemRoots will be nil which will
	// trigger a specific error at verification time.
	return nil
}
