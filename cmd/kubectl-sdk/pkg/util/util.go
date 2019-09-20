/*
Copyright 2019 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"

	"k8s.io/apimachinery/pkg/version"
)

// FilterArgs returns a copy of "l" with elements from "toRemove" filtered out.
func FilterList(l []string, rl []string) []string {
	c := CopyStrSlice(l)
	for _, r := range rl {
		c = RemoveAllElements(c, r)
	}
	return c
}

// RemoveAllElements removes all elements from "s" which match the string "r".
func RemoveAllElements(s []string, r string) []string {
	for i, rlen := 0, len(s); i < rlen; i++ {
		j := i - (rlen - len(s))
		if s[j] == r {
			s = append(s[:j], s[j+1:]...)
		}
	}
	return s
}

// CopyStrSlice returns a copy of the slice of strings.
func CopyStrSlice(s []string) []string {
	c := make([]string, len(s))
	copy(c, s)
	return c
}

// versionMatch returns true if the Major and Minor versions match
// for the passed version infos v1 and v2. Examples:
//   1.11.7 == 1.11.9
//   1.11.7 != 1.10.7
func VersionMatch(v1 version.Info, v2 version.Info) bool {
	major1, err := GetMajorVersion(v1)
	if err != nil {
		return false
	}
	major2, err := GetMajorVersion(v2)
	if err != nil {
		return false
	}
	minor1, err := GetMinorVersion(v1)
	if err != nil {
		return false
	}
	minor2, err := GetMinorVersion(v2)
	if err != nil {
		return false
	}
	if major1 == major2 && minor1 == minor2 {
		return true
	}

	return false
}

func GetMajorVersion(serverVersion version.Info) (int, error) {
	majorStr, err := normalizeVersionStr(serverVersion.Major)
	if err != nil {
		return -1, err
	}
	major, err := strconv.Atoi(majorStr)
	if err != nil || major <= 0 { // NOTE: zero is also not allowed
		return -1, fmt.Errorf("Bad major version string: %v", err)
	}
	return major, nil
}

func GetMinorVersion(serverVersion version.Info) (int, error) {
	minorStr, err := normalizeVersionStr(serverVersion.Minor)
	if err != nil {
		return -1, err
	}
	minor, err := strconv.Atoi(minorStr)
	if err != nil || minor <= 0 { // NOTE: zero is also not allowed
		return -1, fmt.Errorf("Bad minor version string: %v", err)
	}
	return minor, nil
}

// Example:
//   9+ -> 9
//   9.3 -> 9
//   9.1-gke -> 9
func normalizeVersionStr(majorMinor string) (string, error) {
	trimmed := strings.TrimSpace(majorMinor)
	if trimmed == "" {
		return "", fmt.Errorf("Empty server version major/minor string")
	}
	versionStr := ""
	for _, c := range trimmed {
		if unicode.IsDigit(c) {
			versionStr += string(c)
		} else {
			break
		}
	}
	if versionStr == "" {
		return "", fmt.Errorf("Bad server version major/minor string (%s)", trimmed)
	}
	return versionStr, nil
}
