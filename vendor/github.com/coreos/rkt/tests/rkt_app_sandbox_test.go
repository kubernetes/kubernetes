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

// Disabled on kvm due to https://github.com/coreos/rkt/issues/3382
// +build !fly,!kvm

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/tests/testutils"
)

// TestAppSandboxOneApp is a basic test for `rkt app` sandbox.
// It starts the sandbox, adds one app, starts it, and removes it.
func TestAppSandboxAddStartRemove(t *testing.T) {
	testSandbox(t, func(ctx *testutils.RktRunCtx, child *gexpect.ExpectSubprocess, podUUID string) {
		actionTimeout := 30 * time.Second
		imageName := "coreos.com/rkt-inspect/hello"
		appName := "hello-app"
		msg := "HelloFromAppInSandbox"

		aciHello := patchTestACI("rkt-inspect-hello.aci", "--name="+imageName, "--exec=/inspect --print-msg="+msg)
		defer os.Remove(aciHello)

		combinedOutput(t, ctx.ExecCmd("fetch", "--insecure-options=image", aciHello))
		combinedOutput(t, ctx.ExecCmd("app", "add", "--debug", podUUID, imageName, "--name="+appName))
		combinedOutput(t, ctx.ExecCmd("app", "start", "--debug", podUUID, "--app="+appName))

		if err := expectTimeoutWithOutput(child, msg, actionTimeout); err != nil {
			t.Fatalf("Expected %q but not found: %v", msg, err)
		}

		combinedOutput(t, ctx.ExecCmd("app", "rm", "--debug", podUUID, "--app="+appName))

		out := combinedOutput(t, ctx.ExecCmd("app", "list", "--no-legend", podUUID))
		if out != "\n" {
			t.Errorf("unexpected output %q", out)
			return
		}
	})
}

// TestAppSandboxMultipleApps tests multiple apps in a sandbox:
// one that exits successfully, one that exits with an error, and one that keeps running.
func TestAppSandboxMultipleApps(t *testing.T) {
	testSandbox(t, func(ctx *testutils.RktRunCtx, child *gexpect.ExpectSubprocess, podUUID string) {
		actionTimeout := 30 * time.Second

		type app struct {
			name, image, exec, aci string
		}

		apps := []app{
			{
				name:  "winner",
				image: "coreos.com/rkt-inspect/success",
				exec:  "/inspect -print-msg=SUCCESS",
				aci:   "rkt-inspect-success.aci",
			},
			{
				name:  "loser",
				image: "coreos.com/rkt-inspect/fail",
				exec:  "/inspect -print-msg=FAILED -exit-code=12",
				aci:   "rkt-inspect-loser.aci",
			},
			{
				name:  "sleeper",
				image: "coreos.com/rkt-inspect/sleep",
				exec:  "/inspect -print-msg=SLEEP -sleep=120",
				aci:   "rkt-inspect-sleep.aci",
			},
		}

		// create, fetch, add, and start all apps in the sandbox
		for _, app := range apps {
			aci := patchTestACI(app.aci, "--name="+app.image, "--exec="+app.exec)
			defer os.Remove(aci)

			combinedOutput(t, ctx.ExecCmd("fetch", "--insecure-options=image", aci))
			combinedOutput(t, ctx.ExecCmd("app", "add", "--debug", podUUID, app.image, "--name="+app.name))
			combinedOutput(t, ctx.ExecCmd("app", "start", "--debug", podUUID, "--app="+app.name))
		}

		// check for app output messages
		for _, msg := range []string{
			"SUCCESS",
			"FAILED",
			"SLEEP",
		} {
			if err := expectTimeoutWithOutput(child, msg, actionTimeout); err != nil {
				t.Fatalf("Expected %q but not found: %v", msg, err)
			}
		}

		// total retry timeout: 10s
		r := retry{
			n: 20,
			t: 500 * time.Millisecond,
		}

		// assert `rkt app list` for the apps
		if err := r.Retry(func() error {
			got := combinedOutput(t, ctx.ExecCmd("app", "list", "--no-legend", podUUID))

			if strings.Contains(got, "winner\texited") &&
				strings.Contains(got, "loser\texited") &&
				strings.Contains(got, "sleeper\trunning") {
				return nil
			}

			return fmt.Errorf("unexpected result, got %q", got)
		}); err != nil {
			t.Error(err)
			return
		}

		// assert `rkt app status` for the apps
		for _, app := range []struct {
			name          string
			checkExitCode bool
			exitCode      int
			state         string
		}{
			{
				name:          "winner",
				checkExitCode: true,
				exitCode:      0,
				state:         "exited",
			},
			{
				name:          "loser",
				checkExitCode: true,
				exitCode:      12,
				state:         "exited",
			},
			{
				name:  "sleeper",
				state: "running",
			},
		} {
			if err := r.Retry(func() error {
				got := combinedOutput(t, ctx.ExecCmd("app", "status", podUUID, "--app="+app.name))
				ok := true

				if app.checkExitCode {
					ok = ok && strings.Contains(got, "exit_code="+strconv.Itoa(app.exitCode))
				}

				ok = ok && strings.Contains(got, "state="+app.state)

				if !ok {
					return fmt.Errorf("unexpected result, got %q", got)
				}

				return nil
			}); err != nil {
				t.Error(err)
				return
			}
		}

		// remove all apps
		for _, app := range apps {
			combinedOutput(t, ctx.ExecCmd("app", "rm", "--debug", podUUID, "--app="+app.name))
		}

		// assert empty `rkt app list`, no need for retrying,
		// as after removal no leftovers are expected to be present
		got := combinedOutput(t, ctx.ExecCmd("app", "list", "--no-legend", podUUID))
		if got != "\n" {
			t.Errorf("unexpected result, got %q", got)
			return
		}
	})
}

// TestAppSandboxRestart tests multiple apps in a sandbox and restarts one of them.
func TestAppSandboxRestart(t *testing.T) {
	testSandbox(t, func(ctx *testutils.RktRunCtx, child *gexpect.ExpectSubprocess, podUUID string) {
		type app struct {
			name, image, exec, aci string
		}

		apps := []app{
			{
				name:  "app1",
				image: "coreos.com/rkt-inspect/app1",
				exec:  "/inspect -sleep=120",
				aci:   "rkt-inspect-app1.aci",
			},
			{
				name:  "app2",
				image: "coreos.com/rkt-inspect/app1",
				exec:  "/inspect -sleep=120",
				aci:   "rkt-inspect-app1.aci",
			},
		}

		// create, fetch, add, and start all apps in the sandbox
		for _, app := range apps {
			aci := patchTestACI(app.aci, "--name="+app.image, "--exec="+app.exec)
			defer os.Remove(aci)

			combinedOutput(t, ctx.ExecCmd("fetch", "--insecure-options=image", aci))
			combinedOutput(t, ctx.ExecCmd("app", "add", "--debug", podUUID, app.image, "--name="+app.name))
			combinedOutput(t, ctx.ExecCmd("app", "start", "--debug", podUUID, "--app="+app.name))
		}

		// total retry timeout: 10s
		r := retry{
			n: 20,
			t: 500 * time.Millisecond,
		}

		// assert `rkt app list` for the apps
		if err := r.Retry(func() error {
			got := combinedOutput(t, ctx.ExecCmd("app", "list", "--no-legend", podUUID))

			if strings.Contains(got, "app1\trunning") &&
				strings.Contains(got, "app2\trunning") {
				return nil
			}

			return fmt.Errorf("unexpected result, got %q", got)
		}); err != nil {
			t.Error(err)
			return
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

		// assert `rkt app status` for the apps
		for _, app := range apps {
			if err := assertStatus(app.name, "state=running"); err != nil {
				t.Error(err)
				return
			}
		}

		// stop app1
		combinedOutput(t, ctx.ExecCmd("app", "stop", podUUID, "--app=app1"))

		// assert `rkt app status` for the apps
		for _, app := range []struct {
			name   string
			status string
		}{
			{
				name:   "app1",
				status: "state=exited",
			},
			{
				name:   "app2",
				status: "state=running",
			},
		} {
			if err := assertStatus(app.name, app.status); err != nil {
				t.Error(err)
				return
			}
		}

		// assert `rkt app list` for the apps
		if err := r.Retry(func() error {
			got := combinedOutput(t, ctx.ExecCmd("app", "list", "--no-legend", podUUID))

			if strings.Contains(got, "app1\texited") &&
				strings.Contains(got, "app2\trunning") {
				return nil
			}

			return fmt.Errorf("unexpected result, got %q", got)
		}); err != nil {
			t.Error(err)
			return
		}

		// start app1
		combinedOutput(t, ctx.ExecCmd("app", "start", podUUID, "--app=app1"))

		// assert `rkt app status` for the apps
		for _, app := range []struct {
			name   string
			status string
		}{
			{
				name:   "app1",
				status: "state=running",
			},
			{
				name:   "app2",
				status: "state=running",
			},
		} {
			if err := assertStatus(app.name, app.status); err != nil {
				t.Error(err)
				return
			}
		}

		// assert `rkt app list` for the apps
		if err := r.Retry(func() error {
			got := combinedOutput(t, ctx.ExecCmd("app", "list", "--no-legend", podUUID))

			if strings.Contains(got, "app1\trunning") &&
				strings.Contains(got, "app2\trunning") {
				return nil
			}

			return fmt.Errorf("unexpected result, got %q", got)
		}); err != nil {
			t.Error(err)
			return
		}
	})
}

func TestAppSandboxMount(t *testing.T) {
	// this test hast to be skipped on semaphore for now,
	// because it uses an outdated kernel hindering mount propagation,
	// letting this test fail.
	if os.Getenv("SEMAPHORE") == "true" {
		t.Skip("skipped on semaphore")
	}

	mntSrcDir := mustTempDir("rkt-mount-test-")
	defer os.RemoveAll(mntSrcDir)

	mntSrcFile := filepath.Join(mntSrcDir, "test")
	if err := ioutil.WriteFile(mntSrcFile, []byte("content"), 0666); err != nil {
		t.Fatalf("Cannot write file: %v", err)
	}

	testSandbox(t, func(ctx *testutils.RktRunCtx, child *gexpect.ExpectSubprocess, podUUID string) {
		aci := patchTestACI(
			"rkt-inspect-mounter.aci",
			"--name=coreos.com/rkt-inspect/mounter",
			"--exec=/inspect -read-file",
		)
		defer os.Remove(aci)
		combinedOutput(t, ctx.ExecCmd("fetch", "--insecure-options=image", aci))

		for _, tt := range []struct {
			mntTarget    string
			expectedFile string
		}{
			{
				mntTarget:    "/dir2",
				expectedFile: "/dir2/test",
			},
			{
				mntTarget:    "/dir1/link_rel_dir2",
				expectedFile: "/dir2/test",
			},
			{
				mntTarget:    "/dir1/link_abs_dir2",
				expectedFile: "/dir2/test",
			},
			{
				mntTarget:    "/dir1/link_abs_root/notexists",
				expectedFile: "/notexists/test",
			},
			{
				mntTarget:    "/../../../../../../../../notexists",
				expectedFile: "/notexists/test",
			},
			{
				mntTarget:    "../../../../../../../../notexists",
				expectedFile: "/notexists/test",
			},
		} {
			combinedOutput(t, ctx.ExecCmd(
				"app", "add", "--debug", podUUID,
				"coreos.com/rkt-inspect/mounter",
				"--name=mounter",
				"--environment=FILE="+tt.expectedFile,
				"--mnt-volume=name=test,kind=host,source="+mntSrcDir+",target="+tt.mntTarget,
			))

			combinedOutput(t, ctx.ExecCmd("app", "start", "--debug", podUUID, "--app=mounter"))

			if err := expectTimeoutWithOutput(child, "content", 10*time.Second); err != nil {
				t.Fatalf("Expected \"content\" but not found: %v", err)
			}

			combinedOutput(t, ctx.ExecCmd("app", "rm", "--debug", podUUID, "--app=mounter"))
		}
	})
}

func testSandbox(t *testing.T, testFunc func(*testutils.RktRunCtx, *gexpect.ExpectSubprocess, string)) {
	if err := os.Setenv("RKT_EXPERIMENT_APP", "true"); err != nil {
		panic(err)
	}
	defer os.Unsetenv("RKT_EXPERIMENT_APP")

	if err := os.Setenv("RKT_EXPERIMENT_ATTACH", "true"); err != nil {
		panic(err)
	}
	defer os.Unsetenv("RKT_EXPERIMENT_ATTACH")

	tmpDir := mustTempDir("rkt-test-cri-")
	uuidFile := filepath.Join(tmpDir, "uuid")
	defer os.RemoveAll(tmpDir)

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	rkt := ctx.Cmd() + " app sandbox --uuid-file-save=" + uuidFile
	child := spawnOrFail(t, rkt)

	// wait for the sandbox to start
	podUUID, err := waitPodReady(ctx, t, uuidFile, 30*time.Second)
	if err != nil {
		t.Fatal(err)
	}

	testFunc(ctx, child, podUUID)

	// assert that the pod is still running
	got := combinedOutput(t, ctx.ExecCmd("status", podUUID))
	if !strings.Contains(got, "state=running") {
		t.Errorf("unexpected result, got %q", got)
		return
	}

	combinedOutput(t, ctx.ExecCmd("stop", podUUID))

	waitOrFail(t, child, 0)
}
