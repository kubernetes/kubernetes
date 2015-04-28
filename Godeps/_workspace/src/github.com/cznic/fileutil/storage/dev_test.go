// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

package storage

import (
	"testing"
)

func TestDevNothing(t *testing.T) {
	if !*devFlag {
		t.Log("not enabled")
		return
	}
}
