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

// +build host coreos src kvm

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/tests/testutils"
)

func preparePidFileRace(t *testing.T, ctx *testutils.RktRunCtx, pidFileName, sleepImage string) (*gexpect.ExpectSubprocess, *gexpect.ExpectSubprocess, string, string) {
	// Start the pod
	runCmd := fmt.Sprintf("%s --debug --insecure-options=image run --mds-register=false --interactive %s", ctx.Cmd(), sleepImage)
	runChild := spawnOrFail(t, runCmd)

	if err := expectWithOutput(runChild, "Enter text:"); err != nil {
		t.Fatalf("Waited for the prompt but not found: %v", err)
	}

	// Check the ppid file is really created
	cmd := fmt.Sprintf(`%s list --full|grep running`, ctx.Cmd())
	output, err := exec.Command("/bin/sh", "-c", cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("Couldn't list the pods: %v", err)
	}
	UUID := strings.Split(string(output), "\t")[0]

	pidFileNamePath := filepath.Join(ctx.DataDir(), "pods/run", UUID, pidFileName)
	if _, err := os.Stat(pidFileNamePath); err != nil {
		t.Fatalf("Pid file missing: %v", err)
	}

	// Temporarily move the ppid file away
	pidFileNameBackup := pidFileNamePath + ".backup"
	if err := os.Rename(pidFileNamePath, pidFileNameBackup); err != nil {
		t.Fatalf("Cannot move ppid file away: %v", err)
	}

	// Start the "enter" command without the pidfile
	enterCmd := fmt.Sprintf("%s --debug enter %s /inspect --print-msg=RktEnterWorksFine", ctx.Cmd(), UUID)
	t.Logf("%s", enterCmd)
	enterChild := spawnOrFail(t, enterCmd)

	// Enter should be able to wait until the ppid file appears
	time.Sleep(1 * time.Second)

	return runChild, enterChild, pidFileNamePath, pidFileNameBackup
}

// Check that "enter" is able to wait for the ppid file to be created
func NewPidFileDelayedStartTest(pidFileName string) testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {
		sleepImage := patchTestACI("rkt-inspect-sleep.aci", "--exec=/inspect --read-stdin")
		defer os.Remove(sleepImage)

		ctx := testutils.NewRktRunCtx()
		defer ctx.Cleanup()

		runChild, enterChild, pidFileName, pidFileNameBackup := preparePidFileRace(t, ctx, sleepImage, pidFileName)

		// Restore ppid file so the "enter" command can find it
		if err := os.Rename(pidFileNameBackup, pidFileName); err != nil {
			t.Fatalf("Cannot restore ppid file: %v", err)
		}

		// Now the "enter" command works and can complete
		if err := expectWithOutput(enterChild, "RktEnterWorksFine"); err != nil {
			t.Fatalf("Waited for enter to works but failed: %v", err)
		}
		if err := enterChild.Wait(); err != nil {
			t.Fatalf("rkt enter didn't terminate correctly: %v", err)
		}

		// Terminate the pod
		if err := runChild.SendLine("Bye"); err != nil {
			t.Fatalf("rkt couldn't write to the container: %v", err)
		}
		if err := expectWithOutput(runChild, "Received text: Bye"); err != nil {
			t.Fatalf("Expected Bye but not found: %v", err)
		}
		if err := runChild.Wait(); err != nil {
			t.Fatalf("rkt didn't terminate correctly: %v", err)
		}
	})
}

// Check that "enter" doesn't wait forever for the ppid file when the pod is terminated
func NewPidFileAbortedStartTest(pidFileName, escapeSequence string, processExitCode int) testutils.Test {
	return testutils.TestFunc(func(t *testing.T) {

		sleepImage := patchTestACI("rkt-inspect-sleep.aci", "--exec=/inspect --read-stdin")
		defer os.Remove(sleepImage)

		ctx := testutils.NewRktRunCtx()
		defer ctx.Cleanup()

		runChild, enterChild, _, _ := preparePidFileRace(t, ctx, sleepImage, pidFileName)

		// Terminate the pod with the escape sequence
		if err := runChild.SendLine(escapeSequence); err != nil {
			t.Fatalf("Failed to terminate the pod: %v", err)
		}
		waitOrFail(t, runChild, processExitCode)

		// Now the "enter" command terminates quickly
		before := time.Now()
		if err := enterChild.Wait(); err.Error() != "exit status 1" {
			t.Fatalf("rkt enter didn't terminate as expected: %v", err)
		}
		delay := time.Now().Sub(before)
		t.Logf("rkt enter terminated %v after the pod was terminated", delay)
		if delay > time.Second {
			// 1 second shall be enough: it takes less than 50ms on my computer
			t.Fatalf("rkt enter didn't terminate quickly enough: %v", delay)
		}
	})
}
