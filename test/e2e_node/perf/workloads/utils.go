/*
Copyright 2018 The Kubernetes Authors.

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

package workloads

import (
	"fmt"
	"os/exec"
	"regexp"
	"strings"
)

func runCmd(cmd string, args []string) error {
	err := exec.Command(cmd, args...).Run()
	return err
}

func getMatchingLineFromLog(log string, pattern string) (line string, err error) {
	regex, err := regexp.Compile(pattern)
	if err != nil {
		return line, fmt.Errorf("failed to compile regexp %v: %w", pattern, err)
	}

	logLines := strings.Split(log, "\n")
	for _, line := range logLines {
		if regex.MatchString(line) {
			return line, nil
		}
	}

	return line, fmt.Errorf("line with pattern %v not found in log", pattern)
}
