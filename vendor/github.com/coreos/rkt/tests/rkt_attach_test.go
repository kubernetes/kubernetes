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

// +build !fly,!kvm

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/tests/testutils"
)

// TestAttachSmoke is a smoke test for rkt-attach. It exercises several
// features: tty/streams mux, auto-attach, interactive I/O, sandbox support.
func TestAttachSmoke(t *testing.T) {
	actionTimeout := 30 * time.Second
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	imageName := "coreos.com/rkt-inspect/test-attach"
	appName := "test-attach"
	aciPatchArgs := []string{"--name=" + imageName, "--exec=/inspect --read-stdin --check-tty"}
	aciFileName := patchTestACI("rkt-inspect-attach-tty.aci", aciPatchArgs...)
	combinedOutput(t, ctx.ExecCmd("fetch", "--insecure-options=image", aciFileName))
	defer os.Remove(aciFileName)

	var tests = []struct {
		testName   string
		rktRunArgs []string
		expect     string
		sandbox    bool
	}{
		{
			`Check tty with terminal (positive test) - immutable pod`,
			[]string{`--stdin=tty`, `--stdout=tty`, `--stderr=tty`},
			`stdin is a terminal`,
			false,
		},
		{
			`Check tty without terminal (negative test) - immutable pod`,
			[]string{`--stdin=stream`, `--stdout=stream`, `--stderr=stream`},
			`stdin is not a terminal`,
			false,
		},
		{
			`Check tty with terminal (positive test) - sandbox`,
			[]string{`--stdin=tty`, `--stdout=tty`, `--stderr=tty`},
			`stdin is a terminal`,
			true,
		},
		{
			`Check tty without terminal (negative test) - sandbox`,
			[]string{`--stdin=stream`, `--stdout=stream`, `--stderr=stream`},
			`stdin is not a terminal`,
			true,
		},
	}

	for i, tt := range tests {
		t.Logf("Running test #%v: %v", i, tt.testName)

		tmpDir := mustTempDir("rkt-test-attach-")
		uuidFile := filepath.Join(tmpDir, "uuid")
		defer os.RemoveAll(tmpDir)

		err := os.Setenv("RKT_EXPERIMENT_ATTACH", "true")
		if err != nil {
			panic(err)
		}
		defer os.Unsetenv("RKT_EXPERIMENT_ATTACH")

		var uuid string
		var podProc *gexpect.ExpectSubprocess
		if tt.sandbox {
			err := os.Setenv("RKT_EXPERIMENT_APP", "true")
			if err != nil {
				panic(err)
			}
			defer os.Unsetenv("RKT_EXPERIMENT_APP")

			rkt := ctx.Cmd() + " app sandbox --uuid-file-save=" + uuidFile
			podProc = spawnOrFail(t, rkt)

			// wait for the sandbox to start
			uuid, err = waitPodReady(ctx, t, uuidFile, 30*time.Second)
			if err != nil {
				t.Fatal(err)
			}
			addArgs := []string{"app", "add", "--debug", uuid, imageName, "--name=" + appName}
			combinedOutput(t, ctx.ExecCmd(append(addArgs, tt.rktRunArgs...)...))
			combinedOutput(t, ctx.ExecCmd("app", "start", "--debug", uuid, "--app="+appName))
		} else {
			// app starts and blocks, waiting for input
			rktRunCmd := fmt.Sprintf("%s --insecure-options=image run --uuid-file-save=%s %s %s", ctx.Cmd(), uuidFile, aciFileName, strings.Join(tt.rktRunArgs, " "))
			podProc = spawnOrFail(t, rktRunCmd)
			uuid, err = waitPodReady(ctx, t, uuidFile, actionTimeout)
			if err != nil {
				t.Fatal(err)
			}
		}

		// wait for the app to become attachable
		if err := waitAppAttachable(ctx, t, uuid, appName, 30*time.Second); err != nil {
			t.Fatalf("Failed to wait for attachable app #%v: %v", i, err)
		}

		// attach and unblock app by sending some input
		rktAttachCmd := fmt.Sprintf("%s attach %s", ctx.Cmd(), uuid)
		attachProc := spawnOrFail(t, rktAttachCmd)
		input := "some_input"
		if err := attachProc.SendLine(input); err != nil {
			t.Fatalf("Failed to send %q on the prompt #%v: %v", input, i, err)
		}
		feedback := fmt.Sprintf("Received text: %s", input)
		if err := expectTimeoutWithOutput(attachProc, feedback, actionTimeout); err != nil {
			t.Fatalf("Waited for the prompt but not found #%v: %v", i, err)
		}

		// check test result
		if err := expectTimeoutWithOutput(attachProc, tt.expect, actionTimeout); err != nil {
			t.Fatalf("Expected %q but not found #%v: %v", tt.expect, i, err)
		}
		if err := attachProc.Close(); err != nil {
			t.Fatalf("Detach #%v failed: %v", i, err)
		}

		combinedOutput(t, ctx.ExecCmd("stop", uuid))

		waitOrFail(t, podProc, 0)
	}
}

func TestAttachStartStop(t *testing.T) {
	testSandbox(t, func(ctx *testutils.RktRunCtx, child *gexpect.ExpectSubprocess, podUUID string) {
		appName := "attach-start-stop"
		// total retry timeout: 10s
		r := retry{
			n: 20,
			t: 500 * time.Millisecond,
		}

		assertStatus := func(name, status string) error {
			return r.Retry(func() error {
				got := combinedOutput(t, ctx.ExecCmd("app", "status", podUUID, "--app="+name))

				if !strings.Contains(got, status) {
					return fmt.Errorf("unexpected result, got %q", got)
				}

				return nil
			})
		}

		aci := patchTestACI(
			"rkt-inspect-attach-start-stop.aci",
			"--name=coreos.com/rkt-inspect/attach-start-stop",
			"--exec=/inspect -read-stdin -sleep 30",
		)
		defer os.Remove(aci)

		// fetch app
		combinedOutput(t, ctx.ExecCmd("fetch", "--insecure-options=image", aci))

		// add app
		combinedOutput(t, ctx.ExecCmd(
			"app", "add", podUUID,
			"coreos.com/rkt-inspect/attach-start-stop",
			"--name="+appName,
			"--stdin=stream", "--stdout=stream", "--stderr=stream",
		))

		// start app
		combinedOutput(t, ctx.ExecCmd("app", "start", "--debug", podUUID, "--app="+appName))
		if err := assertStatus("attach-start-stop", "running"); err != nil {
			t.Error(err)
			return
		}

		// wait for the app to become attachable
		if err := waitAppAttachable(ctx, t, podUUID, appName, 30*time.Second); err != nil {
			t.Fatalf("Failed to wait for attachable app: %v", err)
		}

		// attach and unblock app by sending some input
		rktAttachCmd := fmt.Sprintf("%s attach %s", ctx.Cmd(), podUUID)
		attachProc := spawnOrFail(t, rktAttachCmd)
		input := "some_input"
		if err := attachProc.SendLine(input); err != nil {
			t.Errorf("Failed to send %q on the prompt: %v", input, err)
			return
		}

		if err := expectTimeoutWithOutput(attachProc, input, 30*time.Second); err != nil {
			t.Errorf("Waited for feedback %q but not found: %v", input, err)
			return
		}

		// stop after entering input
		combinedOutput(t, ctx.ExecCmd("app", "stop", "--debug", podUUID, "--app="+appName))
		if err := assertStatus("attach-start-stop", "exited"); err != nil {
			t.Error(err)
			return
		}

		// restart app
		combinedOutput(t, ctx.ExecCmd("app", "start", "--debug", podUUID, "--app="+appName))
		if err := assertStatus("attach-start-stop", "running"); err != nil {
			t.Error(err)
			return
		}

		// stop without entering input
		combinedOutput(t, ctx.ExecCmd("app", "stop", "--debug", podUUID, "--app="+appName))
		if err := assertStatus("attach-start-stop", "exited"); err != nil {
			t.Error(err)
			return
		}
	})
}
