// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testtext

import (
	"flag"
	"testing"

	"golang.org/x/text/internal/gen"
)

var long = flag.Bool("long", false,
	"run tests that require fetching data online")

// SkipIfNotLong returns whether long tests should be performed.
func SkipIfNotLong(t *testing.T) {
	if testing.Short() || !(gen.IsLocal() || *long) {
		t.Skip("skipping test to prevent downloading; to run use -long or use -local or UNICODE_DIR to specify a local source")
	}
}
