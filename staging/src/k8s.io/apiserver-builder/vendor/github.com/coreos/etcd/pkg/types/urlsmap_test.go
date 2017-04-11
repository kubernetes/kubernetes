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

package types

import (
	"github.com/coreos/etcd/pkg/testutil"
	"reflect"
	"testing"
)

func TestParseInitialCluster(t *testing.T) {
	c, err := NewURLsMap("mem1=http://10.0.0.1:2379,mem1=http://128.193.4.20:2379,mem2=http://10.0.0.2:2379,default=http://127.0.0.1:2379")
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	wc := URLsMap(map[string]URLs{
		"mem1":    testutil.MustNewURLs(t, []string{"http://10.0.0.1:2379", "http://128.193.4.20:2379"}),
		"mem2":    testutil.MustNewURLs(t, []string{"http://10.0.0.2:2379"}),
		"default": testutil.MustNewURLs(t, []string{"http://127.0.0.1:2379"}),
	})
	if !reflect.DeepEqual(c, wc) {
		t.Errorf("cluster = %+v, want %+v", c, wc)
	}
}

func TestParseInitialClusterBad(t *testing.T) {
	tests := []string{
		// invalid URL
		"%^",
		// no URL defined for member
		"mem1=,mem2=http://128.193.4.20:2379,mem3=http://10.0.0.2:2379",
		"mem1,mem2=http://128.193.4.20:2379,mem3=http://10.0.0.2:2379",
		// bad URL for member
		"default=http://localhost/",
	}
	for i, tt := range tests {
		if _, err := NewURLsMap(tt); err == nil {
			t.Errorf("#%d: unexpected successful parse, want err", i)
		}
	}
}

func TestNameURLPairsString(t *testing.T) {
	cls := URLsMap(map[string]URLs{
		"abc": testutil.MustNewURLs(t, []string{"http://1.1.1.1:1111", "http://0.0.0.0:0000"}),
		"def": testutil.MustNewURLs(t, []string{"http://2.2.2.2:2222"}),
		"ghi": testutil.MustNewURLs(t, []string{"http://3.3.3.3:1234", "http://127.0.0.1:2380"}),
		// no PeerURLs = not included
		"four": testutil.MustNewURLs(t, []string{}),
		"five": testutil.MustNewURLs(t, nil),
	})
	w := "abc=http://0.0.0.0:0000,abc=http://1.1.1.1:1111,def=http://2.2.2.2:2222,ghi=http://127.0.0.1:2380,ghi=http://3.3.3.3:1234"
	if g := cls.String(); g != w {
		t.Fatalf("NameURLPairs.String():\ngot  %#v\nwant %#v", g, w)
	}
}

func TestParse(t *testing.T) {
	tests := []struct {
		s  string
		wm map[string][]string
	}{
		{
			"",
			map[string][]string{},
		},
		{
			"a=b",
			map[string][]string{"a": {"b"}},
		},
		{
			"a=b,a=c",
			map[string][]string{"a": {"b", "c"}},
		},
		{
			"a=b,a1=c",
			map[string][]string{"a": {"b"}, "a1": {"c"}},
		},
	}
	for i, tt := range tests {
		m := parse(tt.s)
		if !reflect.DeepEqual(m, tt.wm) {
			t.Errorf("#%d: m = %+v, want %+v", i, m, tt.wm)
		}
	}
}

// TestNewURLsMapIPV6 is only tested in Go1.5+ because Go1.4 doesn't support literal IPv6 address with zone in
// URI (https://github.com/golang/go/issues/6530).
func TestNewURLsMapIPV6(t *testing.T) {
	c, err := NewURLsMap("mem1=http://[2001:db8::1]:2380,mem1=http://[fe80::6e40:8ff:feb1:58e4%25en0]:2380,mem2=http://[fe80::92e2:baff:fe7c:3224%25ext0]:2380")
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	wc := URLsMap(map[string]URLs{
		"mem1": testutil.MustNewURLs(t, []string{"http://[2001:db8::1]:2380", "http://[fe80::6e40:8ff:feb1:58e4%25en0]:2380"}),
		"mem2": testutil.MustNewURLs(t, []string{"http://[fe80::92e2:baff:fe7c:3224%25ext0]:2380"}),
	})
	if !reflect.DeepEqual(c, wc) {
		t.Errorf("cluster = %#v, want %#v", c, wc)
	}
}

func TestNewURLsMapFromStringMapEmpty(t *testing.T) {
	mss := make(map[string]string)
	urlsMap, err := NewURLsMapFromStringMap(mss, ",")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	s := ""
	um, err := NewURLsMap(s)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if um.String() != urlsMap.String() {
		t.Errorf("Expected:\n%+v\ngot:\n%+v", um, urlsMap)
	}
}

func TestNewURLsMapFromStringMapNormal(t *testing.T) {
	mss := make(map[string]string)
	mss["host0"] = "http://127.0.0.1:2379,http://127.0.0.1:2380"
	mss["host1"] = "http://127.0.0.1:2381,http://127.0.0.1:2382"
	mss["host2"] = "http://127.0.0.1:2383,http://127.0.0.1:2384"
	mss["host3"] = "http://127.0.0.1:2385,http://127.0.0.1:2386"
	urlsMap, err := NewURLsMapFromStringMap(mss, ",")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	s := "host0=http://127.0.0.1:2379,host0=http://127.0.0.1:2380," +
		"host1=http://127.0.0.1:2381,host1=http://127.0.0.1:2382," +
		"host2=http://127.0.0.1:2383,host2=http://127.0.0.1:2384," +
		"host3=http://127.0.0.1:2385,host3=http://127.0.0.1:2386"
	um, err := NewURLsMap(s)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if um.String() != urlsMap.String() {
		t.Errorf("Expected:\n%+v\ngot:\n%+v", um, urlsMap)
	}
}
