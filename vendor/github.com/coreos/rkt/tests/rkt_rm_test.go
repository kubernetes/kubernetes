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

	"github.com/coreos/rkt/tests/testutils"
)

func TestRm(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	var uuids []string

	img := getInspectImagePath()
	prepareCmd := fmt.Sprintf("%s --insecure-options=image prepare %s", ctx.Cmd(), img)

	// Finished pod.
	uuid := runRktAndGetUUID(t, prepareCmd)
	runPreparedCmd := fmt.Sprintf("%s --insecure-options=image run-prepared %s", ctx.Cmd(), uuid)
	runRktAndCheckOutput(t, runPreparedCmd, "", false)

	uuids = append(uuids, uuid)

	// Prepared pod.
	uuid = runRktAndGetUUID(t, prepareCmd)
	uuids = append(uuids, uuid)

	podDirs := []string{
		filepath.Join(ctx.DataDir(), "pods", "run"),
		filepath.Join(ctx.DataDir(), "pods", "prepared"),
	}

	for _, dir := range podDirs {
		pods, err := ioutil.ReadDir(dir)
		if err != nil {
			t.Fatalf("cannot read pods directory %q: %v", dir, err)
		}
		if len(pods) == 0 {
			t.Fatalf("pods should still exist in directory %q: %v", dir, pods)
		}
	}

	for _, u := range uuids {
		cmd := fmt.Sprintf("%s rm %s", ctx.Cmd(), u)
		spawnAndWaitOrFail(t, cmd, 0)
	}

	podDirs = append(podDirs, filepath.Join(ctx.DataDir(), "pods", "exited-garbage"))

	for _, dir := range podDirs {
		pods, err := ioutil.ReadDir(dir)
		if err != nil {
			t.Fatalf("cannot read pods directory %q: %v", dir, err)
		}
		if len(pods) != 0 {
			t.Errorf("no pods should exist in directory %q, but found: %v", dir, pods)
		}
	}
}

func TestRmInvalid(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	nonexistentUUID := "0f746094-3438-42bc-ab37-3cf85f132e60"
	tmpDir := mustTempDir("rkt_rm_test")
	defer os.RemoveAll(tmpDir)
	uuidFile := filepath.Join(tmpDir, "uuid-file")
	if err := ioutil.WriteFile(uuidFile, []byte(nonexistentUUID), 0600); err != nil {
		t.Fatalf("cannot write uuid-file: %v", err)
	}

	expected := fmt.Sprintf("no matches found for %q", nonexistentUUID)

	cmd := fmt.Sprintf("%s rm %s", ctx.Cmd(), nonexistentUUID)
	runRktAndCheckOutput(t, cmd, expected, true)

	cmd = fmt.Sprintf("%s rm --uuid-file=%s", ctx.Cmd(), uuidFile)
	runRktAndCheckOutput(t, cmd, expected, true)
}

func TestRmEmptyUUID(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	emptyUUID := "\"\""
	expected := fmt.Sprintf("UUID cannot be empty")

	cmd := fmt.Sprintf("%s rm %s", ctx.Cmd(), emptyUUID)
	runRktAndCheckOutput(t, cmd, expected, true)
}
