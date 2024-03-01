// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package internal

import (
	"context"
	"os"
)

var (
	// This is set to true in identity_classic.go, which is behind the appengine build tag.
	// The appengine build tag is set for the first generation runtimes (<= Go 1.9) but not
	// the second generation runtimes (>= Go 1.11), so this indicates whether we're on a
	// first-gen runtime. See IsStandard below for the second-gen check.
	appengineStandard bool

	// This is set to true in identity_flex.go, which is behind the appenginevm build tag.
	appengineFlex bool
)

// AppID is the implementation of the wrapper function of the same name in
// ../identity.go. See that file for commentary.
func AppID(c context.Context) string {
	return appID(FullyQualifiedAppID(c))
}

// IsStandard is the implementation of the wrapper function of the same name in
// ../appengine.go. See that file for commentary.
func IsStandard() bool {
	// appengineStandard will be true for first-gen runtimes (<= Go 1.9) but not
	// second-gen (>= Go 1.11).
	return appengineStandard || IsSecondGen()
}

// IsSecondGen is the implementation of the wrapper function of the same name in
// ../appengine.go. See that file for commentary.
func IsSecondGen() bool {
	// Second-gen runtimes set $GAE_ENV so we use that to check if we're on a second-gen runtime.
	return os.Getenv("GAE_ENV") == "standard"
}

// IsFlex is the implementation of the wrapper function of the same name in
// ../appengine.go. See that file for commentary.
func IsFlex() bool {
	return appengineFlex
}

// IsAppEngine is the implementation of the wrapper function of the same name in
// ../appengine.go. See that file for commentary.
func IsAppEngine() bool {
	return IsStandard() || IsFlex()
}
