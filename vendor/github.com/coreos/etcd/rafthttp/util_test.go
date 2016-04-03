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

package rafthttp

import (
	"bytes"
	"net/http"
	"reflect"
	"testing"

	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/version"
	"github.com/coreos/go-semver/semver"
)

func TestEntry(t *testing.T) {
	tests := []raftpb.Entry{
		{},
		{Term: 1, Index: 1},
		{Term: 1, Index: 1, Data: []byte("some data")},
	}
	for i, tt := range tests {
		b := &bytes.Buffer{}
		if err := writeEntryTo(b, &tt); err != nil {
			t.Errorf("#%d: unexpected write ents error: %v", i, err)
			continue
		}
		var ent raftpb.Entry
		if err := readEntryFrom(b, &ent); err != nil {
			t.Errorf("#%d: unexpected read ents error: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(ent, tt) {
			t.Errorf("#%d: ent = %+v, want %+v", i, ent, tt)
		}
	}
}

func TestCompareMajorMinorVersion(t *testing.T) {
	tests := []struct {
		va, vb *semver.Version
		w      int
	}{
		// equal to
		{
			semver.Must(semver.NewVersion("2.1.0")),
			semver.Must(semver.NewVersion("2.1.0")),
			0,
		},
		// smaller than
		{
			semver.Must(semver.NewVersion("2.0.0")),
			semver.Must(semver.NewVersion("2.1.0")),
			-1,
		},
		// bigger than
		{
			semver.Must(semver.NewVersion("2.2.0")),
			semver.Must(semver.NewVersion("2.1.0")),
			1,
		},
		// ignore patch
		{
			semver.Must(semver.NewVersion("2.1.1")),
			semver.Must(semver.NewVersion("2.1.0")),
			0,
		},
		// ignore prerelease
		{
			semver.Must(semver.NewVersion("2.1.0-alpha.0")),
			semver.Must(semver.NewVersion("2.1.0")),
			0,
		},
	}
	for i, tt := range tests {
		if g := compareMajorMinorVersion(tt.va, tt.vb); g != tt.w {
			t.Errorf("#%d: compare = %d, want %d", i, g, tt.w)
		}
	}
}

func TestServerVersion(t *testing.T) {
	tests := []struct {
		h  http.Header
		wv *semver.Version
	}{
		// backward compatibility with etcd 2.0
		{
			http.Header{},
			semver.Must(semver.NewVersion("2.0.0")),
		},
		{
			http.Header{"X-Server-Version": []string{"2.1.0"}},
			semver.Must(semver.NewVersion("2.1.0")),
		},
		{
			http.Header{"X-Server-Version": []string{"2.1.0-alpha.0+git"}},
			semver.Must(semver.NewVersion("2.1.0-alpha.0+git")),
		},
	}
	for i, tt := range tests {
		v := serverVersion(tt.h)
		if v.String() != tt.wv.String() {
			t.Errorf("#%d: version = %s, want %s", i, v, tt.wv)
		}
	}
}

func TestMinClusterVersion(t *testing.T) {
	tests := []struct {
		h  http.Header
		wv *semver.Version
	}{
		// backward compatibility with etcd 2.0
		{
			http.Header{},
			semver.Must(semver.NewVersion("2.0.0")),
		},
		{
			http.Header{"X-Min-Cluster-Version": []string{"2.1.0"}},
			semver.Must(semver.NewVersion("2.1.0")),
		},
		{
			http.Header{"X-Min-Cluster-Version": []string{"2.1.0-alpha.0+git"}},
			semver.Must(semver.NewVersion("2.1.0-alpha.0+git")),
		},
	}
	for i, tt := range tests {
		v := minClusterVersion(tt.h)
		if v.String() != tt.wv.String() {
			t.Errorf("#%d: version = %s, want %s", i, v, tt.wv)
		}
	}
}

func TestCheckVersionCompatibility(t *testing.T) {
	ls := semver.Must(semver.NewVersion(version.Version))
	lmc := semver.Must(semver.NewVersion(version.MinClusterVersion))
	tests := []struct {
		server     *semver.Version
		minCluster *semver.Version
		wok        bool
	}{
		// the same version as local
		{
			ls,
			lmc,
			true,
		},
		// one version lower
		{
			lmc,
			&semver.Version{},
			true,
		},
		// one version higher
		{
			&semver.Version{Major: ls.Major + 1},
			ls,
			true,
		},
		// too low version
		{
			&semver.Version{Major: lmc.Major - 1},
			&semver.Version{},
			false,
		},
		// too high version
		{
			&semver.Version{Major: ls.Major + 1, Minor: 1},
			&semver.Version{Major: ls.Major + 1},
			false,
		},
	}
	for i, tt := range tests {
		err := checkVersionCompability("", tt.server, tt.minCluster)
		if ok := err == nil; ok != tt.wok {
			t.Errorf("#%d: ok = %v, want %v", i, ok, tt.wok)
		}
	}
}
