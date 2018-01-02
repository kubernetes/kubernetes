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

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

// TestMtabExistence tests the happy path for the creation of the /etc/mtab
// symlink
func TestMtabExistence(t *testing.T) {
	imageFile := patchTestACI("rkt-inspect-exit.aci", "--exec=/inspect --print-msg=Hello --file-symlink-target")
	defer os.Remove(imageFile)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	expectedLine := `symlink: "/etc/mtab" -> "../proc/self/mounts"`
	rktCmd := fmt.Sprintf(`%s --insecure-options=image run --set-env=FILE=/etc/mtab %s`, ctx.Cmd(), imageFile)
	runRktAndCheckOutput(t, rktCmd, expectedLine, false)
}
