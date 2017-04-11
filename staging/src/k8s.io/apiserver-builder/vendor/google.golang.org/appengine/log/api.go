// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package log

// This file implements the logging API.

import (
	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
)

// Debugf formats its arguments according to the format, analogous to fmt.Printf,
// and records the text as a log message at Debug level. The message will be associated
// with the request linked with the provided context.
func Debugf(ctx context.Context, format string, args ...interface{}) {
	internal.Logf(ctx, 0, format, args...)
}

// Infof is like Debugf, but at Info level.
func Infof(ctx context.Context, format string, args ...interface{}) {
	internal.Logf(ctx, 1, format, args...)
}

// Warningf is like Debugf, but at Warning level.
func Warningf(ctx context.Context, format string, args ...interface{}) {
	internal.Logf(ctx, 2, format, args...)
}

// Errorf is like Debugf, but at Error level.
func Errorf(ctx context.Context, format string, args ...interface{}) {
	internal.Logf(ctx, 3, format, args...)
}

// Criticalf is like Debugf, but at Critical level.
func Criticalf(ctx context.Context, format string, args ...interface{}) {
	internal.Logf(ctx, 4, format, args...)
}
