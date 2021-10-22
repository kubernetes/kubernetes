// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.12
// +build go1.12

package acme

import "runtime/debug"

func init() {
	// Set packageVersion if the binary was built in modules mode and x/crypto
	// was not replaced with a different module.
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return
	}
	for _, m := range info.Deps {
		if m.Path != "golang.org/x/crypto" {
			continue
		}
		if m.Replace == nil {
			packageVersion = m.Version
		}
		break
	}
}
