// Copyright 2015 CoreOS, Inc.
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

package types

import (
	"reflect"
	"testing"

	"github.com/coreos/etcd/pkg/testutil"
)

func TestNewURLs(t *testing.T) {
	tests := []struct {
		strs  []string
		wurls URLs
	}{
		{
			[]string{"http://127.0.0.1:2379"},
			testutil.MustNewURLs(t, []string{"http://127.0.0.1:2379"}),
		},
		// it can trim space
		{
			[]string{"   http://127.0.0.1:2379    "},
			testutil.MustNewURLs(t, []string{"http://127.0.0.1:2379"}),
		},
		// it does sort
		{
			[]string{
				"http://127.0.0.2:2379",
				"http://127.0.0.1:2379",
			},
			testutil.MustNewURLs(t, []string{
				"http://127.0.0.1:2379",
				"http://127.0.0.2:2379",
			}),
		},
	}
	for i, tt := range tests {
		urls, _ := NewURLs(tt.strs)
		if !reflect.DeepEqual(urls, tt.wurls) {
			t.Errorf("#%d: urls = %+v, want %+v", i, urls, tt.wurls)
		}
	}
}

func TestURLsString(t *testing.T) {
	tests := []struct {
		us   URLs
		wstr string
	}{
		{
			URLs{},
			"",
		},
		{
			testutil.MustNewURLs(t, []string{"http://127.0.0.1:2379"}),
			"http://127.0.0.1:2379",
		},
		{
			testutil.MustNewURLs(t, []string{
				"http://127.0.0.1:2379",
				"http://127.0.0.2:2379",
			}),
			"http://127.0.0.1:2379,http://127.0.0.2:2379",
		},
		{
			testutil.MustNewURLs(t, []string{
				"http://127.0.0.2:2379",
				"http://127.0.0.1:2379",
			}),
			"http://127.0.0.2:2379,http://127.0.0.1:2379",
		},
	}
	for i, tt := range tests {
		g := tt.us.String()
		if g != tt.wstr {
			t.Errorf("#%d: string = %s, want %s", i, g, tt.wstr)
		}
	}
}

func TestURLsSort(t *testing.T) {
	g := testutil.MustNewURLs(t, []string{
		"http://127.0.0.4:2379",
		"http://127.0.0.2:2379",
		"http://127.0.0.1:2379",
		"http://127.0.0.3:2379",
	})
	w := testutil.MustNewURLs(t, []string{
		"http://127.0.0.1:2379",
		"http://127.0.0.2:2379",
		"http://127.0.0.3:2379",
		"http://127.0.0.4:2379",
	})
	gurls := URLs(g)
	gurls.Sort()
	if !reflect.DeepEqual(g, w) {
		t.Errorf("URLs after sort = %#v, want %#v", g, w)
	}
}

func TestURLsStringSlice(t *testing.T) {
	tests := []struct {
		us   URLs
		wstr []string
	}{
		{
			URLs{},
			[]string{},
		},
		{
			testutil.MustNewURLs(t, []string{"http://127.0.0.1:2379"}),
			[]string{"http://127.0.0.1:2379"},
		},
		{
			testutil.MustNewURLs(t, []string{
				"http://127.0.0.1:2379",
				"http://127.0.0.2:2379",
			}),
			[]string{"http://127.0.0.1:2379", "http://127.0.0.2:2379"},
		},
		{
			testutil.MustNewURLs(t, []string{
				"http://127.0.0.2:2379",
				"http://127.0.0.1:2379",
			}),
			[]string{"http://127.0.0.2:2379", "http://127.0.0.1:2379"},
		},
	}
	for i, tt := range tests {
		g := tt.us.StringSlice()
		if !reflect.DeepEqual(g, tt.wstr) {
			t.Errorf("#%d: string slice = %+v, want %+v", i, g, tt.wstr)
		}
	}
}

func TestNewURLsFail(t *testing.T) {
	tests := [][]string{
		// no urls given
		{},
		// missing protocol scheme
		{"://127.0.0.1:2379"},
		// unsupported scheme
		{"mailto://127.0.0.1:2379"},
		// not conform to host:port
		{"http://127.0.0.1"},
		// contain a path
		{"http://127.0.0.1:2379/path"},
	}
	for i, tt := range tests {
		_, err := NewURLs(tt)
		if err == nil {
			t.Errorf("#%d: err = nil, but error", i)
		}
	}
}
