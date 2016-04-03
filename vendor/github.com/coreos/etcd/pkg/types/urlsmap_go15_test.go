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

// +build go1.5

package types

import (
	"reflect"
	"testing"

	"github.com/coreos/etcd/pkg/testutil"
)

// This is only tested in Go1.5+ because Go1.4 doesn't support literal IPv6 address with zone in
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
