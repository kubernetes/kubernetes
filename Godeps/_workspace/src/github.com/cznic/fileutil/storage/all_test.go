// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

package storage

import (
	"flag"
	"runtime"
)

var (
	devFlag = flag.Bool("dev", false, "enable dev tests")
	goFlag  = flag.Int("go", 1, "GOMAXPROCS")
)

func init() {
	flag.Parse()
	runtime.GOMAXPROCS(*goFlag)
}
