// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd linux nacl netbsd openbsd solaris

package system

import (
	"crypto/x509"
	"io/ioutil"
)

// Possible directories with certificate files; stop after successfully
// reading at least one file from a directory.
var certDirectories = []string{
	"/system/etc/security/cacerts", // Android
}

func initSystemRoots() []*x509.Certificate {
	var roots []*x509.Certificate
	for _, file := range certFiles {
		data, err := ioutil.ReadFile(file)
		if err == nil {
			roots, _ = appendPEM(roots, data)
			return roots
		}
	}

	for _, directory := range certDirectories {
		fis, err := ioutil.ReadDir(directory)
		if err != nil {
			continue
		}
		rootsAdded := false
		for _, fi := range fis {
			var ok bool
			data, err := ioutil.ReadFile(directory + "/" + fi.Name())
			if err == nil {
				if roots, ok = appendPEM(roots, data); ok {
					rootsAdded = true
				}
			}
		}
		if rootsAdded {
			return roots
		}
	}

	// All of the files failed to load. systemRoots will be nil which will
	// trigger a specific error at verification time.
	return nil
}
