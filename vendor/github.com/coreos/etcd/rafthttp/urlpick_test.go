// Copyright 2015 The etcd Authors
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

package rafthttp

import (
	"net/url"
	"testing"

	"github.com/coreos/etcd/pkg/testutil"
)

// TestURLPickerPickTwice tests that pick returns a possible url,
// and always returns the same one.
func TestURLPickerPickTwice(t *testing.T) {
	picker := mustNewURLPicker(t, []string{"http://127.0.0.1:2380", "http://127.0.0.1:7001"})

	u := picker.pick()
	urlmap := map[url.URL]bool{
		{Scheme: "http", Host: "127.0.0.1:2380"}: true,
		{Scheme: "http", Host: "127.0.0.1:7001"}: true,
	}
	if !urlmap[u] {
		t.Errorf("url picked = %+v, want a possible url in %+v", u, urlmap)
	}

	// pick out the same url when calling pick again
	uu := picker.pick()
	if u != uu {
		t.Errorf("url picked = %+v, want %+v", uu, u)
	}
}

func TestURLPickerUpdate(t *testing.T) {
	picker := mustNewURLPicker(t, []string{"http://127.0.0.1:2380", "http://127.0.0.1:7001"})
	picker.update(testutil.MustNewURLs(t, []string{"http://localhost:2380", "http://localhost:7001"}))

	u := picker.pick()
	urlmap := map[url.URL]bool{
		{Scheme: "http", Host: "localhost:2380"}: true,
		{Scheme: "http", Host: "localhost:7001"}: true,
	}
	if !urlmap[u] {
		t.Errorf("url picked = %+v, want a possible url in %+v", u, urlmap)
	}
}

func TestURLPickerUnreachable(t *testing.T) {
	picker := mustNewURLPicker(t, []string{"http://127.0.0.1:2380", "http://127.0.0.1:7001"})
	u := picker.pick()
	picker.unreachable(u)

	uu := picker.pick()
	if u == uu {
		t.Errorf("url picked = %+v, want other possible urls", uu)
	}
}

func mustNewURLPicker(t *testing.T, us []string) *urlPicker {
	urls := testutil.MustNewURLs(t, us)
	return newURLPicker(urls)
}
