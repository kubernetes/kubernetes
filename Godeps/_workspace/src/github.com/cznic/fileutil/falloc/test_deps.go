// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

package falloc

// Pull test dependencies too.
// Enables easy 'go test X' after 'go get X'
import (
	_ "github.com/cznic/fileutil"
	_ "github.com/cznic/fileutil/storage"
	_ "github.com/cznic/mathutil"
)
