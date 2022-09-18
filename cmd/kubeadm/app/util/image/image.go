/*
Copyright 2020 The Kubernetes Authors.

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

package image

import "regexp"

var (
	// tagMatcher is the regex used to match a tag.
	// Basically we presume an image can be made of `[domain][:port][path]<name>[:tag][@sha256:digest]`
	// We are obvously interested only in the tag, but for the purpose of properly matching it, we also match the digest
	// (if present). All the parts before the tag we match in a single match everything (but not greedy) group.
	// All matched sub-groups, except the tag one, get thrown away. Hence, in a result of FindStringSubmatch, if a tag
	// matches, it's going to be the second returned element (after the full match).
	tagMatcher = regexp.MustCompile(`^(?U:.*)(?::([[:word:]][[:word:].-]*))?(?:@sha256:[a-fA-F\d]{64})?$`)
)

// TagFromImage extracts a tag from image. An empty string is returned if no tag is discovered.
func TagFromImage(image string) string {
	matches := tagMatcher.FindStringSubmatch(image)
	if len(matches) >= 2 {
		return matches[1]
	}
	return ""
}
