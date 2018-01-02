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

var rootRequiringCommands = []string{
	"enter",
	"gc",
	"prepare",
	"rm",
	"run",
	"run-prepared",
}

// TestCommandsNeedRoot tests that an error is immediately returned if
// a subcommand that requires superuser privileges is invoked as a
// user without them
func TestCommandsNeedRoot(t *testing.T) {
	ctx := testutils.NewRktRunCtx()

	uid, gid := ctx.GetUidGidRktBinOwnerNotRoot()

	defer ctx.Cleanup()
	for _, sc := range rootRequiringCommands {
		cmd := fmt.Sprintf("%s %s", ctx.Cmd(), sc)
		runRktAsUidGidAndCheckOutput(t, cmd, "cannot run as unprivileged user", false, true, uid, gid)
	}
}
