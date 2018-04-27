/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package meta

import (
	"testing"
)

func TestKeyType(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		key  *Key
		want KeyType
	}{
		{GlobalKey("abc"), Global},
		{ZonalKey("abc", "us-central1-b"), Zonal},
		{RegionalKey("abc", "us-central1"), Regional},
	} {
		if tc.key.Type() != tc.want {
			t.Errorf("key.Type() == %v, want %v", tc.key.Type(), tc.want)
		}
	}
}

func TestKeyString(t *testing.T) {
	t.Parallel()

	for _, k := range []*Key{
		GlobalKey("abc"),
		RegionalKey("abc", "us-central1"),
		ZonalKey("abc", "us-central1-b"),
	} {
		if k.String() == "" {
			t.Errorf(`k.String() = "", want non-empty`)
		}
	}
}

func TestKeyValid(t *testing.T) {
	t.Parallel()

	region := "us-central1"
	zone := "us-central1-b"

	for _, tc := range []struct {
		key  *Key
		want bool
	}{
		{GlobalKey("abc"), true},
		{RegionalKey("abc", region), true},
		{ZonalKey("abc", zone), true},
		{RegionalKey("abc", "/invalid/"), false},
		{ZonalKey("abc", "/invalid/"), false},
		{&Key{"abc", zone, region}, false},
	} {
		got := tc.key.Valid()
		if got != tc.want {
			t.Errorf("key %+v; key.Valid() = %v, want %v", tc.key, got, tc.want)
		}
	}
}
