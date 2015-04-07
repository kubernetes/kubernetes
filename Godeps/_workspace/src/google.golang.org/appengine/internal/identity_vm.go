// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package internal

import (
	"net/http"
	"os"
)

// These functions are implementations of the wrapper functions
// in ../appengine/identity.go. See that file for commentary.

const (
	hDefaultVersionHostname = "X-AppEngine-Default-Version-Hostname"
	hRequestLogId           = "X-AppEngine-Request-Log-Id"
	hDatacenter             = "X-AppEngine-Datacenter"
)

func DefaultVersionHostname(req interface{}) string {
	return req.(*http.Request).Header.Get(hDefaultVersionHostname)
}

func RequestID(req interface{}) string {
	return req.(*http.Request).Header.Get(hRequestLogId)
}

func Datacenter(req interface{}) string {
	return req.(*http.Request).Header.Get(hDatacenter)
}

func ServerSoftware() string {
	// TODO(dsymonds): Remove fallback when we've verified this.
	if s := os.Getenv("SERVER_SOFTWARE"); s != "" {
		return s
	}
	return "Google App Engine/1.x.x"
}

// TODO(dsymonds): Remove the metadata fetches.

func ModuleName() string {
	if s := os.Getenv("GAE_MODULE_NAME"); s != "" {
		return s
	}
	return string(mustGetMetadata("instance/attributes/gae_backend_name"))
}

func VersionID() string {
	if s := os.Getenv("GAE_MODULE_VERSION"); s != "" {
		return s
	}
	return string(mustGetMetadata("instance/attributes/gae_backend_version"))
}

func InstanceID() string {
	if s := os.Getenv("GAE_MODULE_INSTANCE"); s != "" {
		return s
	}
	return string(mustGetMetadata("instance/attributes/gae_backend_instance"))
}

func partitionlessAppID() string {
	// gae_project has everything except the partition prefix.
	appID := os.Getenv("GAE_LONG_APP_ID")
	if appID == "" {
		appID = string(mustGetMetadata("instance/attributes/gae_project"))
	}
	return appID
}

func fullyQualifiedAppID() string {
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
