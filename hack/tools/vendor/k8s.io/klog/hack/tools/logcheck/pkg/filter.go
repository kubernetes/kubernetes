/*
Copyright 2022 The Kubernetes Authors.

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

package pkg

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// RegexpFilter implements flag.Value by accepting a file name and parsing that
// file.
type RegexpFilter struct {
	filename    string
	validChecks map[string]bool
	lines       []filter
}

type filter struct {
	enabled map[string]bool
	match   *regexp.Regexp
}

var _ flag.Value = &RegexpFilter{}

func (f *RegexpFilter) String() string {
	return f.filename
}

func (f *RegexpFilter) Set(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Reset before parsing.
	f.filename = filename
	f.lines = nil

	// Read line-by-line.
	scanner := bufio.NewScanner(file)
	for lineNr := 0; scanner.Scan(); lineNr++ {
		text := scanner.Text()
		if strings.HasPrefix(text, "#") {
			continue
		}
		text = strings.TrimSpace(text)
		if text == "" {
			continue
		}

		line := filter{
			enabled: map[string]bool{},
		}
		parts := strings.SplitN(text, " ", 2)
		if len(parts) != 2 {
			return fmt.Errorf("%s:%d: not of the format <checks> <regexp>: %s", filename, lineNr, text)
		}
		for _, c := range strings.Split(parts[0], ",") {
			enabled := true
			if strings.HasPrefix(c, "+") {
				c = c[1:]
			} else if strings.HasPrefix(c, "-") {
				enabled = false
				c = c[1:]
			}
			if !f.validChecks[c] {
				return fmt.Errorf("%s:%d: %q is not a supported check: %s", filename, lineNr, c, text)
			}
			line.enabled[c] = enabled
		}

		// Must match entire string.
		re, err := regexp.Compile("^" + parts[1] + "$")
		if err != nil {
			return fmt.Errorf("%s:%d: %v", filename, lineNr, err)
		}
		line.match = re
		f.lines = append(f.lines, line)
	}

	if err := scanner.Err(); err != nil {
		return err
	}
	return nil
}

// Enabled checks whether a certain check is enabled for a file.
func (f *RegexpFilter) Enabled(check string, enabled bool, filename string) bool {
	for _, l := range f.lines {
		if l.match.MatchString(filename) {
			if e, ok := l.enabled[check]; ok {
				enabled = e
			}
		}
	}
	return enabled
}
