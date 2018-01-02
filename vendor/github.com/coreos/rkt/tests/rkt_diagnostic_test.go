// Copyright 2016 The rkt Authors
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

// +build host coreos src kvm

package main

import (
	"fmt"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

func TestDiagnostic(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	var img, runCmd, expectedRegex string

	img = getInspectImagePath()
	runCmd = fmt.Sprintf("%s --insecure-options=image run %s --exec=/foo-not-there", ctx.Cmd(), img)
	expectedRegex = ".*Error: Unable to open \"/foo-not-there\": No such file or directory"
	runRktAndCheckRegexOutput(t, runCmd, expectedRegex)

	runCmd = fmt.Sprintf("%s --insecure-options=image run %s --exec=/foo-not-there -- arg1 arg2", ctx.Cmd(), img)
	expectedRegex = ".*Error: Unable to open \"/foo-not-there\": No such file or directory"
	runRktAndCheckRegexOutput(t, runCmd, expectedRegex)

	runCmd = fmt.Sprintf("%s --insecure-options=image run %s --exec=/foo-not-there\\ X -- arg1 arg2", ctx.Cmd(), img)
	expectedRegex = ".*Error: Unable to open \"/foo-not-there X\": No such file or directory"
	runRktAndCheckRegexOutput(t, runCmd, expectedRegex)

	runCmd = fmt.Sprintf("%s --insecure-options=image run --volume=host-bin,kind=host,source=/bin/ --mount volume=host-bin,target=/var/host-bin %s --exec=/var/host-bin/ls", ctx.Cmd(), img)
	expectedRegex = ".*Error: Unable to open \".*lib.*\\.so\\.\\d\": No such file or directory"
	runRktAndCheckRegexOutput(t, runCmd, expectedRegex)
}
