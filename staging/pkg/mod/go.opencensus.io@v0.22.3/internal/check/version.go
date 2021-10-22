// Copyright 2018, OpenCensus Authors
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

// Command version checks that the version string matches the latest Git tag.
// This is expected to pass only on the master branch.
package main

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"

	opencensus "go.opencensus.io"
)

func main() {
	cmd := exec.Command("git", "tag")
	var buf bytes.Buffer
	cmd.Stdout = &buf
	err := cmd.Run()
	if err != nil {
		log.Fatal(err)
	}
	var versions []version
	for _, vStr := range strings.Split(buf.String(), "\n") {
		if len(vStr) == 0 {
			continue
		}
		// ignore pre-release versions
		if isPreRelease(vStr) {
			continue
		}
		versions = append(versions, parseVersion(vStr))
	}
	sort.Slice(versions, func(i, j int) bool {
		return versionLess(versions[i], versions[j])
	})
	latest := versions[len(versions)-1]
	codeVersion := parseVersion("v" + opencensus.Version())
	if !versionLess(latest, codeVersion) {
		fmt.Printf("exporter.Version is out of date with Git tags. Got %s; want something greater than %s\n", opencensus.Version(), latest)
		os.Exit(1)
	}
	fmt.Printf("exporter.Version is up-to-date: %s\n", opencensus.Version())
}

type version [3]int

func versionLess(v1, v2 version) bool {
	for c := 0; c < 3; c++ {
		if diff := v1[c] - v2[c]; diff != 0 {
			return diff < 0
		}
	}
	return false
}

func isPreRelease(vStr string) bool {
	split := strings.Split(vStr[1:], ".")
	return strings.Contains(split[2], "-")
}

func parseVersion(vStr string) version {
	split := strings.Split(vStr[1:], ".")
	var (
		v   version
		err error
	)
	for i := 0; i < 3; i++ {
		v[i], err = strconv.Atoi(split[i])
		if err != nil {
			fmt.Printf("Unrecognized version tag %q: %s\n", vStr, err)
			os.Exit(2)
		}
	}
	return v
}

func (v version) String() string {
	return fmt.Sprintf("%d.%d.%d", v[0], v[1], v[2])
}
