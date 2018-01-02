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

// +build host coreos src

package main

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/coreos/rkt/tests/testutils"
)

// TestExitCodeSimple is testing a few exit codes on 1 pod containing just 1 app
func TestExitCodeSimple(t *testing.T) {
	for i := 0; i < 3; i++ {
		t.Logf("%d\n", i)
		imageFile := patchTestACI("rkt-inspect-exit.aci", fmt.Sprintf("--exec=/inspect --print-msg=Hello --exit-code=%d", i))
		defer os.Remove(imageFile)
		ctx := testutils.NewRktRunCtx()
		defer ctx.Cleanup()

		cmd := fmt.Sprintf(`%s --debug --insecure-options=image run --mds-register=false %s`,
			ctx.Cmd(), imageFile)
		t.Logf("%s\n", cmd)
		spawnAndWaitOrFail(t, cmd, i)
		checkAppStatus(t, ctx, false, "rkt-inspect", fmt.Sprintf("status=%d", i))
	}
}

// TestExitCodeWithSeveralApps is testing a pod with three apps returning different
// exit codes.
func TestExitCodeWithSeveralApps(t *testing.T) {
	image0File := patchTestACI("rkt-inspect-exit-0.aci", "--name=hello0",
		"--exec=/inspect --print-msg=HelloWorld --exit-code=0 --sleep=1")
	defer os.Remove(image0File)

	image1File := patchTestACI("rkt-inspect-exit-1.aci", "--name=hello1",
		"--exec=/inspect --print-msg=HelloWorld --exit-code=5 --sleep=1")
	defer os.Remove(image1File)

	image2File := patchTestACI("rkt-inspect-exit-2.aci", "--name=hello2",
		"--exec=/inspect --print-msg=HelloWorld --exit-code=6 --sleep=2")
	defer os.Remove(image2File)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf(`%s --debug --insecure-options=image run --mds-register=false %s %s %s`,
		ctx.Cmd(), image0File, image1File, image2File)
	child := spawnOrFail(t, cmd)

	for i := 0; i < 3; i++ {
		// The 3 apps print the same message. We don't have any ordering
		// guarantee but we don't need it.
		if err := expectTimeoutWithOutput(child, "HelloWorld", time.Minute); err != nil {
			t.Fatalf("Could not start the app (#%d): %v", i, err)
		}
	}

	t.Logf("Waiting pod termination\n")
	// Since systemd v227, the exit status is propagated from the app to rkt
	waitOrFail(t, child, 5)

	t.Logf("Check final status\n")

	checkAppStatus(t, ctx, true, "hello0", "status=0")
	checkAppStatus(t, ctx, true, "hello1", "status=5")
	// Currently, hello2 should be stopped correctly (exit code 0) when hello1
	// failed, so it cannot return its exit code 2. This might change with
	// https://github.com/coreos/rkt/issues/1461
	checkAppStatus(t, ctx, true, "hello2", "status=0")
}
