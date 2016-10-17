// Copyright 2013 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package appengine

import "golang.org/x/net/context"

// IsTimeoutError reports whether err is a timeout error.
func IsTimeoutError(err error) bool {
	if err == context.DeadlineExceeded {
		return true
	}
	if t, ok := err.(interface {
		IsTimeout() bool
	}); ok {
		return t.IsTimeout()
	}
	return false
}
