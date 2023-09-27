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

package bugs

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/test/e2e/framework"
)

// The line number of the following code is checked in BugOutput below.
// Be careful when moving it around or changing the import statements above.
// Here are some intentionally blank lines that can be removed to compensate
// for future additional import statements.
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
// This must be line #50.

func helper() {
	framework.RecordBug(framework.NewBug("new bug", 0))
	framework.RecordBug(framework.NewBug("parent", 1))
}

func recordBugs() {
	helper()
	framework.RecordBug(framework.Bug{FileName: "buggy/buggy.go", LineNumber: 100, Message: "hello world"})
	framework.RecordBug(framework.Bug{FileName: "some/relative/path/buggy.go", LineNumber: 200, Message: "    with spaces    \n"})
}

const (
	numBugs   = 3
	bugOutput = `ERROR: bugs_test.go:53: new bug
ERROR: bugs_test.go:58: parent
ERROR: buggy/buggy.go:100: hello world
ERROR: some/relative/path/buggy.go:200: with spaces
`
)

func TestBugs(t *testing.T) {
	assert.NoError(t, framework.FormatBugs())
	recordBugs()
	err := framework.FormatBugs()
	if assert.Error(t, err) {
		assert.Equal(t, bugOutput, err.Error())
	}
}
