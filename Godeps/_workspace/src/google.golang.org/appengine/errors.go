// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// This file provides error functions for common API failure modes.

package appengine

import (
	"fmt"

	"google.golang.org/appengine/internal"
)

// IsOverQuota reports whether err represents an API call failure
// due to insufficient available quota.
func IsOverQuota(err error) bool {
	callErr, ok := err.(*internal.CallError)
	return ok && callErr.Code == 4
}

// MultiError is returned by batch operations when there are errors with
// particular elements. Errors will be in a one-to-one correspondence with
// the input elements; successful elements will have a nil entry.
type MultiError []error

func (m MultiError) Error() string {
	s, n := "", 0
	for _, e := range m {
		if e != nil {
			if n == 0 {
				s = e.Error()
			}
			n++
		}
	}
	switch n {
	case 0:
		return "(0 errors)"
	case 1:
		return s
	case 2:
		return s + " (and 1 other error)"
	}
	return fmt.Sprintf("%s (and %d other errors)", s, n-1)
}
