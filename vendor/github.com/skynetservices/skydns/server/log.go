// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package server

import "log"

// printf calls log.Printf with the parameters given.
func logf(format string, a ...interface{}) {
	log.Printf("skydns: "+format, a...)
}

// fatalf calls log.Fatalf with the parameters given.
func fatalf(format string, a ...interface{}) {
	log.Fatalf("skydns: "+format, a...)
}
