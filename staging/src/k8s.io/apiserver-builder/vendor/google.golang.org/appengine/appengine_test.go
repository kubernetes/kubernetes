// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package appengine

import (
	"testing"
)

func TestValidGeoPoint(t *testing.T) {
	testCases := []struct {
		desc string
		pt   GeoPoint
		want bool
	}{
		{
			"valid",
			GeoPoint{67.21, 13.37},
			true,
		},
		{
			"high lat",
			GeoPoint{-90.01, 13.37},
			false,
		},
		{
			"low lat",
			GeoPoint{90.01, 13.37},
			false,
		},
		{
			"high lng",
			GeoPoint{67.21, 182},
			false,
		},
		{
			"low lng",
			GeoPoint{67.21, -181},
			false,
		},
	}

	for _, tc := range testCases {
		if got := tc.pt.Valid(); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.desc, got, tc.want)
		}
	}
}
