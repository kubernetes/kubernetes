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

import specs "github.com/opencontainers/image-spec/specs-go/v1"

// MatchComparer is able to match and compare platforms to
// filter and sort platforms.
type MatchComparer interface {
	Matcher

	Less(specs.Platform, specs.Platform) bool
}

// Only returns a match comparer for a single platform
// using default resolution logic for the platform.
//
// For ARMv8, will also match ARMv7, ARMv6 and ARMv5 (for 32bit runtimes)
// For ARMv7, will also match ARMv6 and ARMv5
// For ARMv6, will also match ARMv5
func Only(platform specs.Platform) MatchComparer {
	platform = Normalize(platform)
	if platform.Architecture == "arm" {
		if platform.Variant == "v8" {
			return orderedPlatformComparer{
				matchers: []Matcher{
					&matcher{
						Platform: platform,
					},
					&matcher{
						Platform: specs.Platform{
							Architecture: platform.Architecture,
							OS:           platform.OS,
							OSVersion:    platform.OSVersion,
							OSFeatures:   platform.OSFeatures,
							Variant:      "v7",
						},
					},
					&matcher{
						Platform: specs.Platform{
							Architecture: platform.Architecture,
							OS:           platform.OS,
							OSVersion:    platform.OSVersion,
							OSFeatures:   platform.OSFeatures,
							Variant:      "v6",
						},
					},
					&matcher{
						Platform: specs.Platform{
							Architecture: platform.Architecture,
							OS:           platform.OS,
							OSVersion:    platform.OSVersion,
							OSFeatures:   platform.OSFeatures,
							Variant:      "v5",
						},
					},
				},
			}
		}
		if platform.Variant == "v7" {
			return orderedPlatformComparer{
				matchers: []Matcher{
					&matcher{
						Platform: platform,
					},
					&matcher{
						Platform: specs.Platform{
							Architecture: platform.Architecture,
							OS:           platform.OS,
							OSVersion:    platform.OSVersion,
							OSFeatures:   platform.OSFeatures,
							Variant:      "v6",
						},
					},
					&matcher{
						Platform: specs.Platform{
							Architecture: platform.Architecture,
							OS:           platform.OS,
							OSVersion:    platform.OSVersion,
							OSFeatures:   platform.OSFeatures,
							Variant:      "v5",
						},
					},
				},
			}
		}
		if platform.Variant == "v6" {
			return orderedPlatformComparer{
				matchers: []Matcher{
					&matcher{
						Platform: platform,
					},
					&matcher{
						Platform: specs.Platform{
							Architecture: platform.Architecture,
							OS:           platform.OS,
							OSVersion:    platform.OSVersion,
							OSFeatures:   platform.OSFeatures,
							Variant:      "v5",
						},
					},
				},
			}
		}
	}

	return singlePlatformComparer{
		Matcher: &matcher{
			Platform: platform,
		},
	}
}

// Ordered returns a platform MatchComparer which matches any of the platforms
// but orders them in order they are provided.
func Ordered(platforms ...specs.Platform) MatchComparer {
	matchers := make([]Matcher, len(platforms))
	for i := range platforms {
		matchers[i] = NewMatcher(platforms[i])
	}
	return orderedPlatformComparer{
		matchers: matchers,
	}
}

// Any returns a platform MatchComparer which matches any of the platforms
// with no preference for ordering.
func Any(platforms ...specs.Platform) MatchComparer {
	matchers := make([]Matcher, len(platforms))
	for i := range platforms {
		matchers[i] = NewMatcher(platforms[i])
	}
	return anyPlatformComparer{
		matchers: matchers,
	}
}

// All is a platform MatchComparer which matches all platforms
// with preference for ordering.
var All MatchComparer = allPlatformComparer{}

type singlePlatformComparer struct {
	Matcher
}

func (c singlePlatformComparer) Less(p1, p2 specs.Platform) bool {
	return c.Match(p1) && !c.Match(p2)
}

type orderedPlatformComparer struct {
	matchers []Matcher
}

func (c orderedPlatformComparer) Match(platform specs.Platform) bool {
	for _, m := range c.matchers {
		if m.Match(platform) {
			return true
		}
	}
	return false
}

func (c orderedPlatformComparer) Less(p1 specs.Platform, p2 specs.Platform) bool {
	for _, m := range c.matchers {
		p1m := m.Match(p1)
		p2m := m.Match(p2)
		if p1m && !p2m {
			return true
		}
		if p1m || p2m {
			return false
		}
	}
	return false
}

type anyPlatformComparer struct {
	matchers []Matcher
}

func (c anyPlatformComparer) Match(platform specs.Platform) bool {
	for _, m := range c.matchers {
		if m.Match(platform) {
			return true
		}
	}
	return false
}

func (c anyPlatformComparer) Less(p1, p2 specs.Platform) bool {
	var p1m, p2m bool
	for _, m := range c.matchers {
		if !p1m && m.Match(p1) {
			p1m = true
		}
		if !p2m && m.Match(p2) {
			p2m = true
		}
		if p1m && p2m {
			return false
		}
	}
	// If one matches, and the other does, sort match first
	return p1m && !p2m
}

type allPlatformComparer struct{}

func (allPlatformComparer) Match(specs.Platform) bool {
	return true
}

func (allPlatformComparer) Less(specs.Platform, specs.Platform) bool {
	return false
}
