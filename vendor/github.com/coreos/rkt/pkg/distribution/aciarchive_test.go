// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package distribution

import (
	"net/url"
	"testing"
)

func TestACIArchive(t *testing.T) {
	tests := []struct {
		transportURL string
		expectedCIMD string
	}{
		{
			"file:///full/path/to/aci/file.aci",
			"cimd:aci-archive:v=0:file%3A%2F%2F%2Ffull%2Fpath%2Fto%2Faci%2Ffile.aci",
		},
		{
			"https://test.com/fileabc.aci",
			"cimd:aci-archive:v=0:https%3A%2F%2Ftest.com%2Ffileabc.aci",
		},
		{
			"https://test.com/filedef.aci?z=value1&a=value2",
			"cimd:aci-archive:v=0:https%3A%2F%2Ftest.com%2Ffiledef.aci%3Fz%3Dvalue1%26a%3Dvalue2",
		},
	}

	for _, tt := range tests {
		u, err := url.Parse(tt.transportURL)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		d, err := NewACIArchiveFromTransportURL(u)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		u, err = url.Parse(tt.expectedCIMD)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		td, err := NewACIArchive(u)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !d.Equals(td) {
			t.Fatalf("expected identical distribution but got %q != %q", td.CIMD().String(), d.CIMD().String())
		}
	}
}

func TestTransportURL(t *testing.T) {
	tests := []struct {
		transportURL string
	}{
		{
			"file:///full/path/to/aci/fileabc.aci",
		},
		{
			"https://test.com/fileabc.aci",
		},
	}

	for _, tt := range tests {
		u, err := url.Parse(tt.transportURL)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		d, err := NewACIArchiveFromTransportURL(u)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		transportURL := d.(*ACIArchive).TransportURL()
		if tt.transportURL != transportURL.String() {
			t.Fatalf("expected transport url %q, but got %q", tt.transportURL, transportURL.String())
		}
	}
}
