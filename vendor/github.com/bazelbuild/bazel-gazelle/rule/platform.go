/* Copyright 2017 The Bazel Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package rule

import (
	"sort"
)

// Platform represents a GOOS/GOARCH pair. When Platform is used to describe
// sources, dependencies, or flags, either OS or Arch may be empty.
//
// DEPRECATED: do not use outside language/go. This type is Go-specific
// and should be moved to the Go extension.
type Platform struct {
	OS, Arch string
}

// String returns OS, Arch, or "OS_Arch" if both are set. This must match
// the names of config_setting rules in @io_bazel_rules_go//go/platform.
func (p Platform) String() string {
	switch {
	case p.OS != "" && p.Arch != "":
		return p.OS + "_" + p.Arch
	case p.OS != "":
		return p.OS
	case p.Arch != "":
		return p.Arch
	default:
		return ""
	}
}

// KnownPlatforms is the set of target platforms that Go supports. Gazelle
// will generate multi-platform build files using these tags. rules_go and
// Bazel may not actually support all of these.
//
// DEPRECATED: do not use outside language/go.
var KnownPlatforms = []Platform{
	{"android", "386"},
	{"android", "amd64"},
	{"android", "arm"},
	{"android", "arm64"},
	{"darwin", "386"},
	{"darwin", "amd64"},
	{"darwin", "arm"},
	{"darwin", "arm64"},
	{"dragonfly", "amd64"},
	{"freebsd", "386"},
	{"freebsd", "amd64"},
	{"freebsd", "arm"},
	{"ios", "386"},
	{"ios", "amd64"},
	{"ios", "arm"},
	{"ios", "arm64"},
	{"linux", "386"},
	{"linux", "amd64"},
	{"linux", "arm"},
	{"linux", "arm64"},
	{"linux", "mips"},
	{"linux", "mips64"},
	{"linux", "mips64le"},
	{"linux", "mipsle"},
	{"linux", "ppc64"},
	{"linux", "ppc64le"},
	{"linux", "s390x"},
	{"nacl", "386"},
	{"nacl", "amd64p32"},
	{"nacl", "arm"},
	{"netbsd", "386"},
	{"netbsd", "amd64"},
	{"netbsd", "arm"},
	{"openbsd", "386"},
	{"openbsd", "amd64"},
	{"openbsd", "arm"},
	{"plan9", "386"},
	{"plan9", "amd64"},
	{"plan9", "arm"},
	{"solaris", "amd64"},
	{"windows", "386"},
	{"windows", "amd64"},
}

var OSAliases = map[string][]string{
	"android": []string{"linux"},
	"ios":     []string{"darwin"},
}

var (
	// KnownOSs is the sorted list of operating systems that Go supports.
	KnownOSs []string

	// KnownOSSet is the set of operating systems that Go supports.
	KnownOSSet map[string]bool

	// KnownArchs is the sorted list of architectures that Go supports.
	KnownArchs []string

	// KnownArchSet is the set of architectures that Go supports.
	KnownArchSet map[string]bool

	// KnownOSArchs is a map from OS to the archictures they run on.
	KnownOSArchs map[string][]string

	// KnownArchOSs is a map from architectures to that OSs that run on them.
	KnownArchOSs map[string][]string
)

func init() {
	KnownOSSet = make(map[string]bool)
	KnownArchSet = make(map[string]bool)
	KnownOSArchs = make(map[string][]string)
	KnownArchOSs = make(map[string][]string)
	for _, p := range KnownPlatforms {
		KnownOSSet[p.OS] = true
		KnownArchSet[p.Arch] = true
		KnownOSArchs[p.OS] = append(KnownOSArchs[p.OS], p.Arch)
		KnownArchOSs[p.Arch] = append(KnownArchOSs[p.Arch], p.OS)
	}
	KnownOSs = make([]string, 0, len(KnownOSSet))
	KnownArchs = make([]string, 0, len(KnownArchSet))
	for os := range KnownOSSet {
		KnownOSs = append(KnownOSs, os)
	}
	for arch := range KnownArchSet {
		KnownArchs = append(KnownArchs, arch)
	}
	sort.Strings(KnownOSs)
	sort.Strings(KnownArchs)
}
