// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package capability exposes information about outages and scheduled downtime
for specific API capabilities.

This package does not work in App Engine "flexible environment".

Example:
	if !capability.Enabled(c, "datastore_v3", "write") {
		// show user a different page
	}
*/
package capability // import "google.golang.org/appengine/capability"

import (
	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
	"google.golang.org/appengine/log"

	pb "google.golang.org/appengine/internal/capability"
)

// Enabled returns whether an API's capabilities are enabled.
// The wildcard "*" capability matches every capability of an API.
// If the underlying RPC fails (if the package is unknown, for example),
// false is returned and information is written to the application log.
func Enabled(ctx context.Context, api, capability string) bool {
	req := &pb.IsEnabledRequest{
		Package:    &api,
		Capability: []string{capability},
	}
	res := &pb.IsEnabledResponse{}
	if err := internal.Call(ctx, "capability_service", "IsEnabled", req, res); err != nil {
		log.Warningf(ctx, "capability.Enabled: RPC failed: %v", err)
		return false
	}
	switch *res.SummaryStatus {
	case pb.IsEnabledResponse_ENABLED,
		pb.IsEnabledResponse_SCHEDULED_FUTURE,
		pb.IsEnabledResponse_SCHEDULED_NOW:
		return true
	case pb.IsEnabledResponse_UNKNOWN:
		log.Errorf(ctx, "capability.Enabled: unknown API capability %s/%s", api, capability)
		return false
	default:
		return false
	}
}
