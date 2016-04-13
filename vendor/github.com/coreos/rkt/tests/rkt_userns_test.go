// Copyright 2015 The rkt Authors
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
	"os"
	"strings"
	"testing"

	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/tests/testutils"
)

var usernsTests = []struct {
	runCmd     string
	file       string
	expectMode string
	expectUid  string
	expectGid  string
}{
	{
		`^RKT_BIN^ --debug --insecure-options=image run ^USERNS^ --no-overlay --set-env=FILE=^FILE^ --mds-register=false ^IMAGE^`,
		"/", // stage2 rootfs ($POD/stage1/rootfs/opt/stage2/rkt-inspect)
		"drwxr-xr-x",
		"0",
		"0",
	},
	{
		`^RKT_BIN^ --debug --insecure-options=image run ^USERNS^ --no-overlay --set-env=FILE=^FILE^ --mds-register=false ^IMAGE^`,
		"/proc/1/root/", // stage1 rootfs ($POD/stage1/rootfs)
		"drwxr-xr-x",
		"0",
		"", // no check: it could be 0 but also the gid of 'rkt', see https://github.com/coreos/rkt/pull/1452
	},
	// TODO test with overlay fs too. We don't test it for now because
	// Semaphore doesn't support it.
}

func TestUserns(t *testing.T) {
	if !common.SupportsUserNS() {
		t.Skip("User namespaces are not supported on this host.")
	}

	if err := checkUserNS(); err != nil {
		t.Skip("User namespaces don't work on this host.")
	}

	image := patchTestACI("rkt-inspect-stat.aci", "--exec=/inspect --stat-file")
	defer os.Remove(image)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	for i, tt := range usernsTests {
		for _, userNsOpt := range []string{"", "--private-users"} {
			runCmd := tt.runCmd
			runCmd = strings.Replace(runCmd, "^IMAGE^", image, -1)
			runCmd = strings.Replace(runCmd, "^RKT_BIN^", ctx.Cmd(), -1)
			runCmd = strings.Replace(runCmd, "^FILE^", tt.file, -1)
			runCmd = strings.Replace(runCmd, "^USERNS^", userNsOpt, -1)

			t.Logf("Running 'run' test #%v: %v", i, runCmd)
			child, err := gexpect.Spawn(runCmd)
			if err != nil {
				t.Fatalf("Cannot exec rkt #%v: %v", i, err)
			}

			err = expectWithOutput(child, tt.file+": mode: "+tt.expectMode)
			if err != nil {
				t.Fatalf("Expected %q but not found: %v", tt.expectMode, err)
			}
			err = expectWithOutput(child, tt.file+": user: "+tt.expectUid)
			if err != nil {
				t.Fatalf("Expected %q but not found: %v", tt.expectUid, err)
			}
			err = expectWithOutput(child, tt.file+": group: "+tt.expectGid)
			if err != nil {
				t.Fatalf("Expected %q but not found: %v", tt.expectGid, err)
			}

			err = child.Wait()
			if err != nil {
				t.Fatalf("rkt didn't terminate correctly: %v", err)
			}

			ctx.Reset()
		}
	}
}
