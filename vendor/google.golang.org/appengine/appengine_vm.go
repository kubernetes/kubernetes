// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build !appengine

package appengine

import (
	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
)

// The comment below must not be changed.
// It is used by go-app-builder to recognise that this package has
// the Main function to use in the synthetic main.
//   The gophers party all night; the rabbits provide the beats.

// Main is the principal entry point for an app running in App Engine "flexible environment".
// It installs a trivial health checker if one isn't already registered,
// and starts listening on port 8080 (overridden by the $PORT environment
// variable).
//
// See https://cloud.google.com/appengine/docs/flexible/custom-runtimes#health_check_requests
// for details on how to do your own health checking.
//
// Main never returns.
//
// Main is designed so that the app's main package looks like this:
//
//      package main
//
//      import (
//              "google.golang.org/appengine"
//
//              _ "myapp/package0"
//              _ "myapp/package1"
//      )
//
//      func main() {
//              appengine.Main()
//      }
//
// The "myapp/packageX" packages are expected to register HTTP handlers
// in their init functions.
func Main() {
	internal.Main()
}

// BackgroundContext returns a context not associated with a request.
// This should only be used when not servicing a request.
// This only works in App Engine "flexible environment".
func BackgroundContext() context.Context {
	return internal.BackgroundContext()
}
