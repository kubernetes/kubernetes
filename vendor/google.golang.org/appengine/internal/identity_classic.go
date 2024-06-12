// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

//go:build appengine
// +build appengine

package internal

import (
	"context"

	"appengine"
)

func init() {
	appengineStandard = true
}

func DefaultVersionHostname(ctx context.Context) string {
	c := fromContext(ctx)
	if c == nil {
		panic(errNotAppEngineContext)
	}
	return appengine.DefaultVersionHostname(c)
}

func Datacenter(_ context.Context) string { return appengine.Datacenter() }
func ServerSoftware() string              { return appengine.ServerSoftware() }
func InstanceID() string                  { return appengine.InstanceID() }
func IsDevAppServer() bool                { return appengine.IsDevAppServer() }

func RequestID(ctx context.Context) string {
	c := fromContext(ctx)
	if c == nil {
		panic(errNotAppEngineContext)
	}
	return appengine.RequestID(c)
}

func ModuleName(ctx context.Context) string {
	c := fromContext(ctx)
	if c == nil {
		panic(errNotAppEngineContext)
	}
	return appengine.ModuleName(c)
}
func VersionID(ctx context.Context) string {
	c := fromContext(ctx)
	if c == nil {
		panic(errNotAppEngineContext)
	}
	return appengine.VersionID(c)
}

func fullyQualifiedAppID(ctx context.Context) string {
	c := fromContext(ctx)
	if c == nil {
		panic(errNotAppEngineContext)
	}
	return c.FullyQualifiedAppID()
}
