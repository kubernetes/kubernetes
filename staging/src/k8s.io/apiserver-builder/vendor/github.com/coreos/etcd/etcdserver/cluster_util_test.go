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

package etcdserver

import (
	"reflect"
	"testing"

	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/version"
	"github.com/coreos/go-semver/semver"
)

func TestDecideClusterVersion(t *testing.T) {
	tests := []struct {
		vers  map[string]*version.Versions
		wdver *semver.Version
	}{
		{
			map[string]*version.Versions{"a": {Server: "2.0.0"}},
			semver.Must(semver.NewVersion("2.0.0")),
		},
		// unknown
		{
			map[string]*version.Versions{"a": nil},
			nil,
		},
		{
			map[string]*version.Versions{"a": {Server: "2.0.0"}, "b": {Server: "2.1.0"}, "c": {Server: "2.1.0"}},
			semver.Must(semver.NewVersion("2.0.0")),
		},
		{
			map[string]*version.Versions{"a": {Server: "2.1.0"}, "b": {Server: "2.1.0"}, "c": {Server: "2.1.0"}},
			semver.Must(semver.NewVersion("2.1.0")),
		},
		{
			map[string]*version.Versions{"a": nil, "b": {Server: "2.1.0"}, "c": {Server: "2.1.0"}},
			nil,
		},
	}

	for i, tt := range tests {
		dver := decideClusterVersion(tt.vers)
		if !reflect.DeepEqual(dver, tt.wdver) {
			t.Errorf("#%d: ver = %+v, want %+v", i, dver, tt.wdver)
		}
	}
}

func TestIsCompatibleWithVers(t *testing.T) {
	tests := []struct {
		vers       map[string]*version.Versions
		local      types.ID
		minV, maxV *semver.Version
		wok        bool
	}{
		// too low
		{
			map[string]*version.Versions{
				"a": {Server: "2.0.0", Cluster: "not_decided"},
				"b": {Server: "2.1.0", Cluster: "2.1.0"},
				"c": {Server: "2.1.0", Cluster: "2.1.0"},
			},
			0xa,
			semver.Must(semver.NewVersion("2.0.0")), semver.Must(semver.NewVersion("2.0.0")),
			false,
		},
		{
			map[string]*version.Versions{
				"a": {Server: "2.1.0", Cluster: "not_decided"},
				"b": {Server: "2.1.0", Cluster: "2.1.0"},
				"c": {Server: "2.1.0", Cluster: "2.1.0"},
			},
			0xa,
			semver.Must(semver.NewVersion("2.0.0")), semver.Must(semver.NewVersion("2.1.0")),
			true,
		},
		// too high
		{
			map[string]*version.Versions{
				"a": {Server: "2.2.0", Cluster: "not_decided"},
				"b": {Server: "2.0.0", Cluster: "2.0.0"},
				"c": {Server: "2.0.0", Cluster: "2.0.0"},
			},
			0xa,
			semver.Must(semver.NewVersion("2.1.0")), semver.Must(semver.NewVersion("2.2.0")),
			false,
		},
		// cannot get b's version, expect ok
		{
			map[string]*version.Versions{
				"a": {Server: "2.1.0", Cluster: "not_decided"},
				"b": nil,
				"c": {Server: "2.1.0", Cluster: "2.1.0"},
			},
			0xa,
			semver.Must(semver.NewVersion("2.0.0")), semver.Must(semver.NewVersion("2.1.0")),
			true,
		},
		// cannot get b and c's version, expect not ok
		{
			map[string]*version.Versions{
				"a": {Server: "2.1.0", Cluster: "not_decided"},
				"b": nil,
				"c": nil,
			},
			0xa,
			semver.Must(semver.NewVersion("2.0.0")), semver.Must(semver.NewVersion("2.1.0")),
			false,
		},
	}

	for i, tt := range tests {
		ok := isCompatibleWithVers(tt.vers, tt.local, tt.minV, tt.maxV)
		if ok != tt.wok {
			t.Errorf("#%d: ok = %+v, want %+v", i, ok, tt.wok)
		}
	}
}
