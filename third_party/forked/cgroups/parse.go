/*
Copyright 2024 The Kubernetes Authors.

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

package cgroups

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
)

// ParseCgroupFileUnified returns legacy subsystem paths as the first value,
// and returns the unified path as the second value.
func ParseCgroupFileUnified(path string) (map[string]string, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	return parseCgroupFromReaderUnified(f)
}

func parseCgroupFromReaderUnified(r io.Reader) (map[string]string, string, error) {
	var (
		cgroups = make(map[string]string)
		unified = ""
		s       = bufio.NewScanner(r)
	)
	for s.Scan() {
		var (
			text  = s.Text()
			parts = strings.SplitN(text, ":", 3)
		)
		if len(parts) < 3 {
			return nil, unified, fmt.Errorf("invalid cgroup entry: %q", text)
		}
		for _, subs := range strings.Split(parts[1], ",") {
			if subs == "" {
				unified = parts[2]
			} else {
				cgroups[subs] = parts[2]
			}
		}
	}
	if err := s.Err(); err != nil {
		return nil, unified, err
	}
	return cgroups, unified, nil
}
