// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build !appengine

package internal

import (
	"log"
	"net/http"
	"os"
	"strings"

	netcontext "golang.org/x/net/context"
)

// These functions are implementations of the wrapper functions
// in ../appengine/identity.go. See that file for commentary.

const (
	hDefaultVersionHostname = "X-AppEngine-Default-Version-Hostname"
	hRequestLogId           = "X-AppEngine-Request-Log-Id"
	hDatacenter             = "X-AppEngine-Datacenter"
)

func ctxHeaders(ctx netcontext.Context) http.Header {
	c := fromContext(ctx)
	if c == nil {
		return nil
	}
	return c.Request().Header
}

func DefaultVersionHostname(ctx netcontext.Context) string {
	return ctxHeaders(ctx).Get(hDefaultVersionHostname)
}

func RequestID(ctx netcontext.Context) string {
	return ctxHeaders(ctx).Get(hRequestLogId)
}

func Datacenter(ctx netcontext.Context) string {
	if dc := ctxHeaders(ctx).Get(hDatacenter); dc != "" {
		return dc
	}
	// If the header isn't set, read zone from the metadata service.
	// It has the format projects/[NUMERIC_PROJECT_ID]/zones/[ZONE]
	zone, err := getMetadata("instance/zone")
	if err != nil {
		log.Printf("Datacenter: %v", err)
		return ""
	}
	parts := strings.Split(string(zone), "/")
	if len(parts) == 0 {
		return ""
	}
	return parts[len(parts)-1]
}

func ServerSoftware() string {
	// TODO(dsymonds): Remove fallback when we've verified this.
	if s := os.Getenv("SERVER_SOFTWARE"); s != "" {
		return s
	}
	if s := os.Getenv("GAE_ENV"); s != "" {
		return s
	}
	return "Google App Engine/1.x.x"
}

// TODO(dsymonds): Remove the metadata fetches.

func ModuleName(_ netcontext.Context) string {
	if s := os.Getenv("GAE_MODULE_NAME"); s != "" {
		return s
	}
	if s := os.Getenv("GAE_SERVICE"); s != "" {
		return s
	}
	return string(mustGetMetadata("instance/attributes/gae_backend_name"))
}

func VersionID(_ netcontext.Context) string {
	if s1, s2 := os.Getenv("GAE_MODULE_VERSION"), os.Getenv("GAE_MINOR_VERSION"); s1 != "" && s2 != "" {
		return s1 + "." + s2
	}
	if s1, s2 := os.Getenv("GAE_VERSION"), os.Getenv("GAE_DEPLOYMENT_ID"); s1 != "" && s2 != "" {
		return s1 + "." + s2
	}
	return string(mustGetMetadata("instance/attributes/gae_backend_version")) + "." + string(mustGetMetadata("instance/attributes/gae_backend_minor_version"))
}

func InstanceID() string {
	if s := os.Getenv("GAE_MODULE_INSTANCE"); s != "" {
		return s
	}
	if s := os.Getenv("GAE_INSTANCE"); s != "" {
		return s
	}
	return string(mustGetMetadata("instance/attributes/gae_backend_instance"))
}

func partitionlessAppID() string {
	// gae_project has everything except the partition prefix.
	if appID := os.Getenv("GAE_LONG_APP_ID"); appID != "" {
		return appID
	}
	if project := os.Getenv("GOOGLE_CLOUD_PROJECT"); project != "" {
		return project
	}
	return string(mustGetMetadata("instance/attributes/gae_project"))
}

func fullyQualifiedAppID(_ netcontext.Context) string {
	if s := os.Getenv("GAE_APPLICATION"); s != "" {
		return s
	}
	appID := partitionlessAppID()

	part := os.Getenv("GAE_PARTITION")
	if part == "" {
		part = string(mustGetMetadata("instance/attributes/gae_partition"))
	}

	if part != "" {
		appID = part + "~" + appID
	}
	return appID
}

func IsDevAppServer() bool {
	return os.Getenv("RUN_WITH_DEVAPPSERVER") != ""
}
