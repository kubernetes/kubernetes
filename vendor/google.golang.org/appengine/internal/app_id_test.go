// Copyright 2011 Google Inc. All Rights Reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package internal

import (
	"testing"
)

func TestAppIDParsing(t *testing.T) {
	testCases := []struct {
		in                           string
		partition, domain, displayID string
	}{
		{"simple-app-id", "", "", "simple-app-id"},
		{"domain.com:domain-app-id", "", "domain.com", "domain-app-id"},
		{"part~partition-app-id", "part", "", "partition-app-id"},
		{"part~domain.com:display", "part", "domain.com", "display"},
	}

	for _, tc := range testCases {
		part, dom, dis := parseFullAppID(tc.in)
		if part != tc.partition {
			t.Errorf("partition of %q: got %q, want %q", tc.in, part, tc.partition)
		}
		if dom != tc.domain {
			t.Errorf("domain of %q: got %q, want %q", tc.in, dom, tc.domain)
		}
		if dis != tc.displayID {
			t.Errorf("displayID of %q: got %q, want %q", tc.in, dis, tc.displayID)
		}
	}
}
