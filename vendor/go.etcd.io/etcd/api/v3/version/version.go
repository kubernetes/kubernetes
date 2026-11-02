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

// Package version implements etcd version parsing and contains latest version
// information.
package version

import (
	"fmt"
	"strings"

	"github.com/Masterminds/semver/v3"
)

var (
	// MinClusterVersion is the min cluster version this etcd binary is compatible with.
	MinClusterVersion = "3.0.0"
	Version           = "3.8.0-alpha.0"
	APIVersion        = "unknown"

	// Git SHA Value will be set during build
	GitSHA = "Not provided (use ./build instead of go build)"
)

// Get all constant versions defined in a centralized place.
var (
	V3_0 = *semver.New(3, 0, 0, "", "")
	V3_1 = *semver.New(3, 1, 0, "", "")
	V3_2 = *semver.New(3, 2, 0, "", "")
	V3_3 = *semver.New(3, 3, 0, "", "")
	V3_4 = *semver.New(3, 4, 0, "", "")
	V3_5 = *semver.New(3, 5, 0, "", "")
	V3_6 = *semver.New(3, 6, 0, "", "")
	V3_7 = *semver.New(3, 7, 0, "", "")
	V3_8 = *semver.New(3, 8, 0, "", "")
	V3_9 = *semver.New(3, 9, 0, "", "")
	V4_0 = *semver.New(4, 0, 0, "", "")

	// AllVersions keeps all the versions in ascending order.
	AllVersions = []semver.Version{V3_0, V3_1, V3_2, V3_3, V3_4, V3_5, V3_6, V3_7, V3_8, V4_0}
)

func init() {
	ver, err := semver.NewVersion(Version)
	if err == nil {
		APIVersion = fmt.Sprintf("%d.%d", ver.Major(), ver.Minor())
	}
}

type Versions struct {
	Server  string `json:"etcdserver"`
	Cluster string `json:"etcdcluster"`
	Storage string `json:"storage"`
	// TODO: raft state machine version
}

// Cluster only keeps the major.minor.
func Cluster(v string) string {
	if ver, err := semver.NewVersion(v); err == nil {
		return fmt.Sprintf("%d.%d", ver.Major(), ver.Minor())
	}
	vs := strings.Split(v, ".")
	if len(vs) <= 2 {
		return v
	}
	return fmt.Sprintf("%s.%s", vs[0], vs[1])
}

func Compare(ver1, ver2 semver.Version) int {
	return ver1.Compare(&ver2)
}

func LessThan(ver1, ver2 semver.Version) bool {
	return ver1.LessThan(&ver2)
}

func Equal(ver1, ver2 semver.Version) bool {
	return ver1.Equal(&ver2)
}
