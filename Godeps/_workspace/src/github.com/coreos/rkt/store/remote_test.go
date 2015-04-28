// Copyright 2014 CoreOS, Inc.
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

package store

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestNewRemote(t *testing.T) {
	const (
		u1   = "https://example.com"
		u2   = "https://foo.com"
		data = "asdf"
	)
	dir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	s, err := NewStore(dir)
	if err != nil {
		t.Fatal(err)
	}

	// Create our first Remote, and simulate Store() to create row in the table
	na := NewRemote(u1, "")
	na.BlobKey = data
	s.WriteRemote(na)

	// Get a new remote w the same parameters, reading from table should be fine
	nb, ok, err := s.GetRemote(u1)
	if err != nil {
		t.Fatalf("unexpected error reading index: %v", err)
	}
	if !ok {
		t.Fatalf("unexpected index not found")
	}
	if nb.BlobKey != data {
		t.Fatalf("bad data returned from store: got %v, want %v", nb.BlobKey, data)
	}

	// Get a remote with a different URI
	nc, ok, err := s.GetRemote(u2)
	// Should get an error, since the URI shouldn't be present in the table
	if ok {
		t.Fatalf("unexpected index found")
	}
	// Remote shouldn't be populated
	if nc.BlobKey != "" {
		t.Errorf("unexpected blob: got %v", nc.BlobKey)
	}
}
