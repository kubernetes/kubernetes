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
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/tests/testutils"
)

func TestGC(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	imagePath := getInspectImagePath()
	// Finished pods.
	importImageAndRun(imagePath, t, ctx)

	// Prepared pods.
	importImageAndPrepare(imagePath, t, ctx)

	// Abort prepare.
	cmd := fmt.Sprintf("%s --insecure-options=image prepare %s %s", ctx.Cmd(), imagePath, imagePath)
	spawnAndWaitOrFail(t, cmd, 254)

	gcCmd := fmt.Sprintf("%s gc --mark-only=true --expire-prepared=0 --grace-period=0", ctx.Cmd())
	spawnAndWaitOrFail(t, gcCmd, 0)

	pods := podsRemaining(t, ctx)
	if len(pods) == 0 {
		t.Fatalf("pods should still be present in rkt's data directory")
	}

	gcCmd = fmt.Sprintf("%s gc --mark-only=false --expire-prepared=0 --grace-period=0", ctx.Cmd())
	spawnAndWaitOrFail(t, gcCmd, 0)

	pods = podsRemaining(t, ctx)
	if len(pods) != 0 {
		t.Fatalf("no pods should exist in rkt data directory, but found: %s", pods)
	}
}

func podsRemaining(t *testing.T, ctx *testutils.RktRunCtx) []os.FileInfo {
	gcDirs := []string{
		filepath.Join(ctx.DataDir(), "pods", "exited-garbage"),
		filepath.Join(ctx.DataDir(), "pods", "prepared"),
		filepath.Join(ctx.DataDir(), "pods", "garbage"),
		filepath.Join(ctx.DataDir(), "pods", "run"),
	}

	var remainingPods []os.FileInfo
	for _, dir := range gcDirs {
		pods, err := ioutil.ReadDir(dir)
		if err != nil {
			t.Fatalf("cannot read gc directory %q: %v", dir, err)
		}
		remainingPods = append(remainingPods, pods...)
	}

	return remainingPods
}

func TestGCAfterUnmount(t *testing.T) {
	if err := common.SupportsOverlay(); err != nil {
		t.Skipf("Overlay fs not supported: %v", err)
	}

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	imagePath := getInspectImagePath()

	for _, rmNetns := range []bool{false, true} {
		_, err := importImageAndFetchHash(t, ctx, "", imagePath)
		if err != nil {
			t.Fatalf("%v", err)
		}
		cmd := fmt.Sprintf("%s --insecure-options=image prepare %s", ctx.Cmd(), imagePath)
		uuid := runRktAndGetUUID(t, cmd)

		cmd = fmt.Sprintf("%s run-prepared %s", ctx.Cmd(), uuid)
		runRktAndCheckOutput(t, cmd, "", false)

		unmountPod(t, ctx, uuid, rmNetns)

		pods := podsRemaining(t, ctx)
		if len(pods) == 0 {
			t.Fatalf("pods should still be present in rkt's data directory")
		}

		gcCmd := fmt.Sprintf("%s gc --debug --mark-only=false --expire-prepared=0 --grace-period=0", ctx.Cmd())
		// check we don't get any output (an error) after "executing net-plugin..."
		runRktAndCheckRegexOutput(t, gcCmd, `executing net-plugin .*\n\z`)

		pods = podsRemaining(t, ctx)
		if len(pods) != 0 {
			t.Fatalf("no pods should exist rkt's data directory, but found: %v", pods)
		}

	}
}
