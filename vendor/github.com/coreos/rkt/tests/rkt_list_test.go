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
	"strings"
	"testing"
	"time"

	"github.com/coreos/rkt/tests/testutils"
)

const delta = 4 * time.Second
const precision = 2 * time.Second

// compareTime checks if a and b are roughly equal
func compareTime(a time.Time, b time.Time) bool {
	diff := a.Sub(b)
	if diff < 0 {
		diff = -diff
	}
	return diff < precision
}

func TestRktList(t *testing.T) {
	const imgName = "rkt-list-test"

	image := patchTestACI(fmt.Sprintf("%s.aci", imgName), fmt.Sprintf("--name=%s", imgName))
	defer os.Remove(image)

	imageHash := getHashOrPanic(image)
	imgID := ImageID{image, imageHash}

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	// Prepare image
	cmd := fmt.Sprintf("%s --insecure-options=image prepare %s", ctx.Cmd(), imgID.path)
	podUuid := runRktAndGetUUID(t, cmd)

	// Get hash
	imageID := fmt.Sprintf("sha512-%s", imgID.hash[:12])

	// Define tests
	tests := []struct {
		cmd           string
		shouldSucceed bool
		expect        string
	}{
		// Test that pod UUID is in output
		{
			"list --full",
			true,
			podUuid,
		},
		// Test that image name is in output
		{
			"list",
			true,
			imgName,
		},
		// Test that imageID is in output
		{
			"list --full",
			true,
			imageID,
		},
		// Remove the image
		{
			fmt.Sprintf("image rm %s", imageID),
			true,
			"successfully removed",
		},
		// Name should still show up in rkt list
		{
			"list",
			true,
			imgName,
		},
		// Test that imageID is still in output
		{
			"list --full",
			true,
			imageID,
		},
	}

	// Run tests
	for i, tt := range tests {
		runCmd := fmt.Sprintf("%s %s", ctx.Cmd(), tt.cmd)
		t.Logf("Running test #%d, %s", i, runCmd)
		runRktAndCheckOutput(t, runCmd, tt.expect, !tt.shouldSucceed)
	}
}

func getCreationStartTime(t *testing.T, ctx *testutils.RktRunCtx, imageID string) (creation time.Time, start time.Time) {
	// Run rkt list --full
	rktCmd := fmt.Sprintf("%s list --full", ctx.Cmd())
	child := spawnOrFail(t, rktCmd)
	child.Wait()

	// Get creation time
	match := fmt.Sprintf(".*%s\t.*\t(.*)\t(.*)\t", imageID)
	result, out, err := expectRegexWithOutput(child, match)
	if err != nil {
		t.Fatalf("%q regex not found, Error: %v\nOutput: %v", match, err, out)
	}
	tmStr := strings.TrimSpace(result[1])
	creation, err = time.Parse(defaultTimeLayout, tmStr)
	if err != nil {
		t.Fatalf("Error parsing creation time: %q", err)
	}

	tmStr = strings.TrimSpace(result[2])
	start, err = time.Parse(defaultTimeLayout, tmStr)
	if err != nil {
		t.Fatalf("Error parsing start time: %q", err)
	}

	return creation, start
}

func TestRktListCreatedStarted(t *testing.T) {
	const imgName = "rkt-list-creation-start-time-test"

	image := patchTestACI(fmt.Sprintf("%s.aci", imgName), fmt.Sprintf("--exec=/inspect --exit-code=0"))
	defer os.Remove(image)

	imageHash := getHashOrPanic(image)
	imgID := ImageID{image, imageHash}

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	// Prepare image
	cmd := fmt.Sprintf("%s --insecure-options=image prepare %s", ctx.Cmd(), imgID.path)
	podUuid := runRktAndGetUUID(t, cmd)

	// t0: prepare
	expectPrepare := time.Now()

	// Get hash
	imageID := fmt.Sprintf("sha512-%s", imgID.hash[:12])

	tmpDir := mustTempDir(imgName)
	defer os.RemoveAll(tmpDir)

	time.Sleep(delta)

	// Run image
	cmd = fmt.Sprintf("%s run-prepared %s", ctx.Cmd(), podUuid)
	rktChild := spawnOrFail(t, cmd)

	// t1: run
	expectRun := time.Now()

	waitOrFail(t, rktChild, 0)

	creation, start := getCreationStartTime(t, ctx, imageID)
	if !compareTime(expectPrepare, creation) {
		t.Fatalf("rkt list returned an incorrect creation time. Got: %q Expect: %q", creation, expectPrepare)
	}
	if !compareTime(expectRun, start) {
		t.Fatalf("rkt list returned an incorrect start time. Got: %q Expect: %q", start, expectRun)
	}
}
