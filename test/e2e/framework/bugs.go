/*
Copyright 2023 The Kubernetes Authors.

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

package framework

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/onsi/ginkgo/v2/types"
)

var (
	bugs     []Bug
	bugMutex sync.Mutex
)

// RecordBug stores information about a bug in the E2E suite source code that
// cannot be reported through ginkgo.Fail because it was found outside of some
// test, for example during test registration.
//
// This can be used instead of raising a panic. Then all bugs can be reported
// together instead of failing after the first one.
func RecordBug(bug Bug) {
	bugMutex.Lock()
	defer bugMutex.Unlock()

	bugs = append(bugs, bug)
}

type Bug struct {
	FileName   string
	LineNumber int
	Message    string
}

// NewBug creates a new bug with a location that is obtained by skipping a certain number
// of stack frames. Passing zero will record the source code location of the direct caller
// of NewBug.
func NewBug(message string, skip int) Bug {
	location := types.NewCodeLocation(skip + 1)
	return Bug{FileName: location.FileName, LineNumber: location.LineNumber, Message: message}
}

// FormatBugs produces a report that includes all bugs recorded earlier via
// RecordBug. An error is returned with the report if there have been bugs.
func FormatBugs() error {
	bugMutex.Lock()
	defer bugMutex.Unlock()

	if len(bugs) == 0 {
		return nil
	}

	lines := make([]string, 0, len(bugs))
	wd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("get current directory: %v", err)
	}
	// Sort by file name, line number, message. For the sake of simplicity
	// this uses the full file name even though the output the may use a
	// relative path. Usually the result should be the same because full
	// paths will all have the same prefix.
	sort.Slice(bugs, func(i, j int) bool {
		switch strings.Compare(bugs[i].FileName, bugs[j].FileName) {
		case -1:
			return true
		case 1:
			return false
		}
		if bugs[i].LineNumber < bugs[j].LineNumber {
			return true
		}
		if bugs[i].LineNumber > bugs[j].LineNumber {
			return false
		}
		return bugs[i].Message < bugs[j].Message
	})
	for _, bug := range bugs {
		// Use relative paths, if possible.
		path := bug.FileName
		if wd != "" {
			if relpath, err := filepath.Rel(wd, bug.FileName); err == nil {
				path = relpath
			}
		}
		lines = append(lines, fmt.Sprintf("ERROR: %s:%d: %s\n", path, bug.LineNumber, strings.TrimSpace(bug.Message)))
	}
	return errors.New(strings.Join(lines, ""))
}
