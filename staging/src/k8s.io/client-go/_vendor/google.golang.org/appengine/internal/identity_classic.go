// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build appengine

package internal

import (
	"appengine"

	netcontext "golang.org/x/net/context"
)

func DefaultVersionHostname(ctx netcontext.Context) string {
	return appengine.DefaultVersionHostname(fromContext(ctx))
}

func RequestID(ctx netcontext.Context) string  { return appengine.RequestID(fromContext(ctx)) }
func Datacenter(_ netcontext.Context) string   { return appengine.Datacenter() }
func ServerSoftware() string                   { return appengine.ServerSoftware() }
func ModuleName(ctx netcontext.Context) string { return appengine.ModuleName(fromContext(ctx)) }
func VersionID(ctx netcontext.Context) string  { return appengine.VersionID(fromContext(ctx)) }
func InstanceID() string                       { return appengine.InstanceID() }
func IsDevAppServer() bool                     { return appengine.IsDevAppServer() }

func fullyQualifiedAppID(ctx netcontext.Context) string { return fromContext(ctx).FullyQualifiedAppID() }
