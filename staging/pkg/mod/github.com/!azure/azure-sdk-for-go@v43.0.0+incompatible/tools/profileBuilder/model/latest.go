// +build go1.9

// Copyright 2018 Microsoft Corporation
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

package model

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const previewSubdir = string(os.PathSeparator) + "preview" + string(os.PathSeparator)

var previewVer = regexp.MustCompile(`(?:v?\d{4}-\d{2}-\d{2}|v?\d+[\.\d+\.\d\-]*)(?:-preview|-beta)`)

// these predicates are used when walking the package directories.
// if a predicate returns true it means to include that package.

func acceptAllPredicate(name string) bool {
	return true
}

func includePreviewPredicate(name string) bool {
	// check if the path contains a /preview/ subdirectory
	if strings.Contains(name, previewSubdir) {
		return false
	}
	return !previewVer.MatchString(name)
}

type operationGroup struct {
	provider string
	arm      bool
	group    string
	api      string
}

type operInfo struct {
	version string
	rawpath string
	modver  string
}

type latestTracker map[operationGroup]operInfo

func (tracker latestTracker) Upsert(path string, pi PathInfo) (operInfo, operationGroup, int) {
	group := operationGroup{
		provider: pi.Provider,
		arm:      pi.IsArm,
		group:    pi.Group,
		api:      pi.APIPkg,
	}

	prev, ok := tracker[group]
	if !ok {
		tracker[group] = operInfo{pi.Version, path, pi.ModVer}
		return prev, group, 1
	}

	if le, _ := versionLE(prev.version, pi.Version); le {
		tracker[group] = operInfo{pi.Version, path, pi.ModVer}
		return prev, group, 0
	}
	return prev, group, -1
}

// skipFileCheck is for testing purposes
func (tracker latestTracker) ToListDefinition(skipFileCheck bool) (ListDefinition, error) {
	listDef := ListDefinition{
		Include: []string{},
	}
	for _, entry := range tracker {
		if skipFileCheck {
			listDef.Include = append(listDef.Include, entry.rawpath)
			continue
		}
		absolute, err := filepath.Abs(entry.rawpath)
		if err != nil {
			return listDef, err
		}
		// ensure the directory actually contains files
		entries, err := ioutil.ReadDir(absolute)
		if err != nil {
			return listDef, err
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				listDef.Include = append(listDef.Include, absolute)
				break
			}
		}
	}
	sort.Strings(listDef.Include)
	return listDef, nil
}

// GetLatestPackages returns the collection of latest packages in terms of API and module version.
func GetLatestPackages(rootDir string, includePreview bool, verboseLog *log.Logger) (ListDefinition, error) {
	predicate := includePreviewPredicate
	if includePreview {
		predicate = acceptAllPredicate
	}

	tracker := latestTracker{}

	filepath.Walk(rootDir, func(currentPath string, info os.FileInfo, openErr error) error {
		pi, err := DeconstructPath(currentPath)
		if err != nil || !info.IsDir() {
			return nil
		} else if !predicate(currentPath) {
			verboseLog.Printf("%q rejected by Predicate", currentPath)
			return nil
		}

		prev, group, result := tracker.Upsert(currentPath, pi)
		switch result {
		case 1:
			verboseLog.Printf("New group found %v using version %q modver %q", group, pi.Version, pi.ModVer)
		case 0:
			verboseLog.Printf("Updating group %v from version %q to %q modver %q", group, prev.version, pi.Version, pi.ModVer)
		case -1:
			verboseLog.Printf("Evaluated group %v version %q decided to stay with %q", group, pi.Version, prev.version)
		}

		return nil
	})

	return tracker.ToListDefinition(false)
}

// versionLE takes two version strings that share a format and returns true if the one on the
// left is less than or equal to the one on the right. If the two do not match in format, or
// are not in a well known format, this will return false and an error.
var versionLE = func() func(string, string) (bool, error) {
	type strategyTuple struct {
		match   *regexp.Regexp
		handler func([]string, []string) (bool, error)
	}

	// there are two strategies in the following order:
	// The first handles Azure API Versions which have a date optionally followed by some tag.
	// The second strategy compares two semvers.
	// the order is important as the semver strategy is less specific than the API version strategy due to
	// inconsistencies in the directory structure (e.g. v1, 6.2, v1.0 etc).  given this we must always check
	// the API version strategry first as the semver strategy can match an API version yielding incorrect results.
	wellKnownStrategies := []strategyTuple{
		{
			match: regexp.MustCompile(`^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})(?:[\.\-](?P<tag>.+))?$`),
			handler: func(leftMatch, rightMatch []string) (bool, error) {
				var err error
				for i := 1; i <= 3; i++ { // Start with index 1 because the element 0 is the entire match, not a group. End at 3 because there are three numeric groups.
					if leftMatch[i] == rightMatch[i] {
						continue
					}

					var leftNum, rightNum int
					leftNum, err = strconv.Atoi(leftMatch[i])
					if err != nil {
						return false, err
					}

					rightNum, err = strconv.Atoi(rightMatch[i])
					if err != nil {
						return false, err
					}

					if leftNum < rightNum {
						return true, nil
					}
					return false, nil
				}

				if leftTag, rightTag := leftMatch[4], rightMatch[4]; leftTag == "" && rightTag != "" { // match[4] is the tag portion of a date based API Version label
					return false, nil
				} else if leftTag != "" && rightTag != "" {
					return leftTag <= rightTag, nil
				}
				return true, nil
			},
		},
		{
			match: regexp.MustCompile(`(?P<major>\d+)(?:\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?-?(?P<tag>.*))?`),
			handler: func(leftMatch, rightMatch []string) (bool, error) {
				for i := 1; i <= 3; i++ {
					if len(leftMatch[i]) == 0 || len(rightMatch[i]) == 0 {
						return leftMatch[i] <= rightMatch[i], nil
					}
					numLeft, err := strconv.Atoi(leftMatch[i])
					if err != nil {
						return false, err
					}
					numRight, err := strconv.Atoi(rightMatch[i])
					if err != nil {
						return false, err
					}

					if numLeft < numRight {
						return true, nil
					}

					if numLeft > numRight {
						return false, nil
					}
				}

				return leftMatch[4] <= rightMatch[4], nil
			},
		},
	}

	// This function finds a strategy which recognizes the versions passed to it, then applies that strategy.
	return func(left, right string) (bool, error) {
		if left == right {
			return true, nil
		}

		for _, strategy := range wellKnownStrategies {
			if leftMatch, rightMatch := strategy.match.FindAllStringSubmatch(left, 1), strategy.match.FindAllStringSubmatch(right, 1); len(leftMatch) > 0 && len(rightMatch) > 0 {
				return strategy.handler(leftMatch[0], rightMatch[0])
			}
		}
		return false, fmt.Errorf("Unable to find versioning strategy that could compare %q and %q", left, right)
	}
}()
