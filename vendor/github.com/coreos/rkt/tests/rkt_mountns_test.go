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
	"os"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

func TestMountNSApp(t *testing.T) {
	image := patchTestACI("rkt-test-mount-ns-app.aci", "--exec=/inspect --check-mountns", "--capability=CAP_SYS_PTRACE")
	defer os.Remove(image)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	rktCmd := fmt.Sprintf("%s --insecure-options=image run %s", ctx.Cmd(), image)

	expectedLine := "check-mountns: DIFFERENT"
	runRktAndCheckOutput(t, rktCmd, expectedLine, false)
}

func TestSharedSlave(t *testing.T) {
}
