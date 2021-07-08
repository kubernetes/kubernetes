// +build windows

/*
   Copyright The containerd Authors.

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

package platforms

import (
	"fmt"
	"runtime"
	"strconv"
	"strings"

	imagespec "github.com/opencontainers/image-spec/specs-go/v1"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"golang.org/x/sys/windows"
)

type matchComparer struct {
	defaults        Matcher
	osVersionPrefix string
}

// Match matches platform with the same windows major, minor
// and build version.
func (m matchComparer) Match(p imagespec.Platform) bool {
	if m.defaults.Match(p) {
		// TODO(windows): Figure out whether OSVersion is deprecated.
		return strings.HasPrefix(p.OSVersion, m.osVersionPrefix)
	}
	return false
}

// Less sorts matched platforms in front of other platforms.
// For matched platforms, it puts platforms with larger revision
// number in front.
func (m matchComparer) Less(p1, p2 imagespec.Platform) bool {
	m1, m2 := m.Match(p1), m.Match(p2)
	if m1 && m2 {
		r1, r2 := revision(p1.OSVersion), revision(p2.OSVersion)
		return r1 > r2
	}
	return m1 && !m2
}

func revision(v string) int {
	parts := strings.Split(v, ".")
	if len(parts) < 4 {
		return 0
	}
	r, err := strconv.Atoi(parts[3])
	if err != nil {
		return 0
	}
	return r
}

// Default returns the current platform's default platform specification.
func Default() MatchComparer {
	major, minor, build := windows.RtlGetNtVersionNumbers()
	return matchComparer{
		defaults: Ordered(DefaultSpec(), specs.Platform{
			OS:           "linux",
			Architecture: runtime.GOARCH,
		}),
		osVersionPrefix: fmt.Sprintf("%d.%d.%d", major, minor, build),
	}
}
