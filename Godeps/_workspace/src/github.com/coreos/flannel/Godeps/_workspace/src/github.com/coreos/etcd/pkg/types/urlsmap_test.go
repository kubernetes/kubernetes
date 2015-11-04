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
