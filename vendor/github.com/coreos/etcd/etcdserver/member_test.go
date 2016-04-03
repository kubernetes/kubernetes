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

package etcdserver

import (
	"net/url"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/etcd/pkg/types"
)

func timeParse(value string) *time.Time {
	t, err := time.Parse(time.RFC3339, value)
	if err != nil {
		panic(err)
	}
	return &t
}

func TestMemberTime(t *testing.T) {
	tests := []struct {
		mem *Member
		id  types.ID
	}{
		{NewMember("mem1", []url.URL{{Scheme: "http", Host: "10.0.0.8:2379"}}, "", nil), 14544069596553697298},
		// Same ID, different name (names shouldn't matter)
		{NewMember("memfoo", []url.URL{{Scheme: "http", Host: "10.0.0.8:2379"}}, "", nil), 14544069596553697298},
		// Same ID, different Time
		{NewMember("mem1", []url.URL{{Scheme: "http", Host: "10.0.0.8:2379"}}, "", timeParse("1984-12-23T15:04:05Z")), 2448790162483548276},
		// Different cluster name
		{NewMember("mcm1", []url.URL{{Scheme: "http", Host: "10.0.0.8:2379"}}, "etcd", timeParse("1984-12-23T15:04:05Z")), 6973882743191604649},
		{NewMember("mem1", []url.URL{{Scheme: "http", Host: "10.0.0.1:2379"}}, "", timeParse("1984-12-23T15:04:05Z")), 1466075294948436910},
		// Order shouldn't matter
		{NewMember("mem1", []url.URL{{Scheme: "http", Host: "10.0.0.1:2379"}, {Scheme: "http", Host: "10.0.0.2:2379"}}, "", nil), 16552244735972308939},
		{NewMember("mem1", []url.URL{{Scheme: "http", Host: "10.0.0.2:2379"}, {Scheme: "http", Host: "10.0.0.1:2379"}}, "", nil), 16552244735972308939},
	}
	for i, tt := range tests {
		if tt.mem.ID != tt.id {
			t.Errorf("#%d: mem.ID = %v, want %v", i, tt.mem.ID, tt.id)
		}
	}
}

func TestMemberPick(t *testing.T) {
	tests := []struct {
		memb *Member
		urls map[string]bool
	}{
		{
			newTestMember(1, []string{"abc", "def", "ghi", "jkl", "mno", "pqr", "stu"}, "", nil),
			map[string]bool{
				"abc": true,
				"def": true,
				"ghi": true,
				"jkl": true,
				"mno": true,
				"pqr": true,
				"stu": true,
			},
		},
		{
			newTestMember(2, []string{"xyz"}, "", nil),
			map[string]bool{"xyz": true},
		},
	}
	for i, tt := range tests {
		for j := 0; j < 1000; j++ {
			a := tt.memb.PickPeerURL()
			if !tt.urls[a] {
				t.Errorf("#%d: returned ID %q not in expected range!", i, a)
				break
			}
		}
	}
}

func TestMemberClone(t *testing.T) {
	tests := []*Member{
		newTestMember(1, nil, "abc", nil),
		newTestMember(1, []string{"http://a"}, "abc", nil),
		newTestMember(1, nil, "abc", []string{"http://b"}),
		newTestMember(1, []string{"http://a"}, "abc", []string{"http://b"}),
	}
	for i, tt := range tests {
		nm := tt.Clone()
		if nm == tt {
			t.Errorf("#%d: the pointers are the same, and clone doesn't happen", i)
		}
		if !reflect.DeepEqual(nm, tt) {
			t.Errorf("#%d: member = %+v, want %+v", i, nm, tt)
		}
	}
}

func newTestMember(id uint64, peerURLs []string, name string, clientURLs []string) *Member {
	return &Member{
		ID:             types.ID(id),
		RaftAttributes: RaftAttributes{PeerURLs: peerURLs},
		Attributes:     Attributes{Name: name, ClientURLs: clientURLs},
	}
}
