// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package remote_api

import (
	"testing"
)

func TestAppIDRE(t *testing.T) {
	appID := "s~my-appid-539"
	tests := []string{
		"{rtok: 8306111115908860449, app_id: s~my-appid-539}\n",
		"{rtok: 8306111115908860449, app_id: 's~my-appid-539'}\n",
		`{rtok: 8306111115908860449, app_id: "s~my-appid-539"}`,
		`{rtok: 8306111115908860449, "app_id":"s~my-appid-539"}`,
	}
	for _, v := range tests {
		if g := appIDRE.FindStringSubmatch(v); g == nil || g[1] != appID {
			t.Errorf("appIDRE.FindStringSubmatch(%s) got %q, want %q", v, g, appID)
		}
	}
}
