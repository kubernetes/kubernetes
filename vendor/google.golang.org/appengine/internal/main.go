// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build appengine

package internal

import (
	"appengine_internal"
)

func Main() {
	MainPath = ""
	appengine_internal.Main()
}
