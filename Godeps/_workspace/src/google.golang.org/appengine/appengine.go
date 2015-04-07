// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Package appengine provides basic functionality for Google App Engine.
//
// For more information on how to write Go apps for Google App Engine, see:
// https://cloud.google.com/appengine/docs/go/
package appengine

import (
	"net/http"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine/internal"
)

// IsDevAppServer reports whether the App Engine app is running in the
// development App Server.
func IsDevAppServer() bool {
	// TODO(dsymonds): Detect this.
	return false
}

// Context represents the context of an in-flight HTTP request.
type Context interface {
	// Debugf formats its arguments according to the format, analogous to fmt.Printf,
	// and records the text as a log message at Debug level.
	Debugf(format string, args ...interface{})

	// Infof is like Debugf, but at Info level.
	Infof(format string, args ...interface{})

	// Warningf is like Debugf, but at Warning level.
	Warningf(format string, args ...interface{})

	// Errorf is like Debugf, but at Error level.
	Errorf(format string, args ...interface{})

	// Criticalf is like Debugf, but at Critical level.
	Criticalf(format string, args ...interface{})

	// The remaining methods are for internal use only.
	// Developer-facing APIs wrap these methods to provide a more friendly API.

	// Internal use only.
	Call(service, method string, in, out proto.Message, opts *internal.CallOptions) error
	// Internal use only. Use AppID instead.
	FullyQualifiedAppID() string
	// Internal use only.
	Request() interface{}
}

// NewContext returns a context for an in-flight HTTP request.
// Repeated calls will return the same value.
func NewContext(req *http.Request) Context {
	return internal.NewContext(req)
}

// TODO(dsymonds): Add BackgroundContext function?

// BlobKey is a key for a blobstore blob.
//
// Conceptually, this type belongs in the blobstore package, but it lives in
// the appengine package to avoid a circular dependency: blobstore depends on
// datastore, and datastore needs to refer to the BlobKey type.
type BlobKey string

// GeoPoint represents a location as latitude/longitude in degrees.
type GeoPoint struct {
	Lat, Lng float64
}

// Valid returns whether a GeoPoint is within [-90, 90] latitude and [-180, 180] longitude.
func (g GeoPoint) Valid() bool {
	return -90 <= g.Lat && g.Lat <= 90 && -180 <= g.Lng && g.Lng <= 180
}
