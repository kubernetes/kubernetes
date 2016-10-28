// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package internal

import netcontext "golang.org/x/net/context"

// These functions are implementations of the wrapper functions
// in ../appengine/identity.go. See that file for commentary.

func AppID(c netcontext.Context) string {
	return appID(FullyQualifiedAppID(c))
}
