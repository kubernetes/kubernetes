// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package internal

// These functions are implementations of the wrapper functions
// in ../appengine/identity.go. See that file for commentary.

func AppID(fqai string) string {
	return appID(fqai)
}
