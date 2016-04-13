// Copyright 2013 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build appengine

package cloudsql

import (
	"net"

	"appengine/cloudsql"
)

func connect(instance string) (net.Conn, error) {
	return cloudsql.Dial(instance)
}
