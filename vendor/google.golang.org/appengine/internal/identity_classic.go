// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build appengine

package internal

import (
	"appengine"

	netcontext "golang.org/x/net/context"
)

func init() {
	appengineStandard = true
}

func DefaultVersionHostname(ctx netcontext.Context) string {
	c := fromContext(ctx)
	if c == nil {
		panic(errNotAppEngineContext)
	}
	return appengine.DefaultVersionHostname(c)
}

func Datacenter(_ netcontext.Context) string { return appengine.Datacenter() }
func ServerSoftware() string                 { return appengine.ServerSoftware() }
func InstanceID() string                     { return appengine.InstanceID() }
func IsDevAppServer() bool                   { return appengine.IsDevAppServer() }

func RequestID(ctx netcontext.Context) string {
	c := fromContext(ctx)
	if c == nil {
		panic(errNotAppEngineContext)
	}
	return appengine.RequestID(c)
}

func ModuleName(ctx netcontext.Context) string {
	c := fromContext(ctx)
	if c == nil {
		panic(errNotAppEngineContext)
	}
	return appengine.ModuleName(c)
}
func VersionID(ctx netcontext.Context) string {
	c := fromContext(ctx)
	if c == nil {
		panic(errNotAppEngineContext)
	}
	return appengine.VersionID(c)
}

func fullyQualifiedAppID(ctx netcontext.Context) string {
	c := fromContext(ctx)
	if c == nil {
		panic(errNotAppEngineContext)
	}
	return c.FullyQualifiedAppID()
}
