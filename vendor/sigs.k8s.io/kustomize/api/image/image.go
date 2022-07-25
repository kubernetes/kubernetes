// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package image

import (
	"regexp"
	"strings"
)

// IsImageMatched returns true if the value of t is identical to the
// image name in the full image name and tag as given by s.
func IsImageMatched(s, t string) bool {
	// Tag values are limited to [a-zA-Z0-9_.{}-].
	// Some tools like Bazel rules_k8s allow tag patterns with {} characters.
	// More info: https://github.com/bazelbuild/rules_k8s/pull/423
	pattern, _ := regexp.Compile("^" + t + "(:[a-zA-Z0-9_.{}-]*)?(@sha256:[a-zA-Z0-9_.{}-]*)?$")
	return pattern.MatchString(s)
}

// Split separates and returns the name and tag parts
// from the image string using either colon `:` or at `@` separators.
// image reference pattern: [[host[:port]/]component/]component[:tag][@digest]
func Split(imageName string) (name string, tag string, digest string) {
	// check if image name contains a domain
	// if domain is present, ignore domain and check for `:`
	searchName := imageName
	slashIndex := strings.Index(imageName, "/")
	if slashIndex > 0 {
		searchName = imageName[slashIndex:]
	} else {
		slashIndex = 0
	}

	id := strings.Index(searchName, "@")
	ic := strings.Index(searchName, ":")

	// no tag or digest
	if ic < 0 && id < 0 {
		return imageName, "", ""
	}

	// digest only
	if id >= 0 && (id < ic || ic < 0) {
		id += slashIndex
		name = imageName[:id]
		digest = strings.TrimPrefix(imageName[id:], "@")
		return name, "", digest
	}

	// tag and digest
	if id >= 0 && ic >= 0 {
		id += slashIndex
		ic += slashIndex
		name = imageName[:ic]
		tag = strings.TrimPrefix(imageName[ic:id], ":")
		digest = strings.TrimPrefix(imageName[id:], "@")
		return name, tag, digest
	}

	// tag only
	ic += slashIndex
	name = imageName[:ic]
	tag = strings.TrimPrefix(imageName[ic:], ":")
	return name, tag, ""
}
