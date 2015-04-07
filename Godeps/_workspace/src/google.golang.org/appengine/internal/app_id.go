// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package internal

import (
	"strings"
)

func parseFullAppID(appid string) (partition, domain, displayID string) {
	if i := strings.Index(appid, "~"); i != -1 {
		partition, appid = appid[:i], appid[i+1:]
	}
	if i := strings.Index(appid, ":"); i != -1 {
		domain, appid = appid[:i], appid[i+1:]
	}
	return partition, domain, appid
}

// appID returns "appid" or "domain.com:appid".
func appID(fullAppID string) string {
	_, dom, dis := parseFullAppID(fullAppID)
	if dom != "" {
		return dom + ":" + dis
	}
	return dis
}
