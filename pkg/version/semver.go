/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package version

import (
	"strings"
	"unicode"

	"github.com/blang/semver"
	"github.com/golang/glog"
)

func Parse(gitversion string) (semver.Version, error) {
	// optionally trim leading spaces then one v
	var seen bool
	gitversion = strings.TrimLeftFunc(gitversion, func(ch rune) bool {
		if seen {
			return false
		}
		if ch == 'v' {
			seen = true
			return true
		}
		return unicode.IsSpace(ch)
	})

	return semver.Make(gitversion)
}

func MustParse(gitversion string) semver.Version {
	v, err := Parse(gitversion)
	if err != nil {
		glog.Fatalf("failed to parse semver from gitversion %q: %v", gitversion, err)
	}
	return v
}
