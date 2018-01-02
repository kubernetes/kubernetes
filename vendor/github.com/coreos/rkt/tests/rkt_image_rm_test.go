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
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/tests/testutils"
)

const (
	rmImageOk = "rm: 1 image(s) successfully removed"

	unreferencedACI = "rkt-unreferencedACI.aci"
	unreferencedApp = "coreos.com/rkt-unreferenced"
	referencedApp   = "coreos.com/rkt-inspect"

	stage1App = "coreos.com/rkt/stage1"
)

func TestImageRunRmName(t *testing.T) {
	imageFile := patchTestACI(unreferencedACI, fmt.Sprintf("--name=%s", unreferencedApp))
	defer os.Remove(imageFile)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --insecure-options=image fetch %s", ctx.Cmd(), imageFile)
	spawnAndWaitOrFail(t, cmd, 0)

	// at this point we know that RKT_INSPECT_IMAGE env var is not empty
	referencedACI := os.Getenv("RKT_INSPECT_IMAGE")
	cmd = fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s", ctx.Cmd(), referencedACI)
	spawnAndWaitOrFail(t, cmd, 0)

	t.Logf("Retrieving stage1 image name")
	stage1ImageName := getImageName(t, ctx, stage1App)

	t.Logf("Removing stage1 image (should work)")
	removeImage(t, ctx, stage1ImageName)

	t.Logf("Removing image for app %s (should work)", referencedApp)
	removeImage(t, ctx, referencedApp)

	t.Logf("Removing image for app %s (should work)", unreferencedApp)
	removeImage(t, ctx, unreferencedApp)
}

func TestImageRunRmID(t *testing.T) {
	imageFile := patchTestACI(unreferencedACI, fmt.Sprintf("--name=%s", unreferencedApp))
	defer os.Remove(imageFile)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --insecure-options=image fetch %s", ctx.Cmd(), imageFile)
	t.Logf("Fetching %s", imageFile)
	spawnAndWaitOrFail(t, cmd, 0)

	// at this point we know that RKT_INSPECT_IMAGE env var is not empty
	referencedACI := os.Getenv("RKT_INSPECT_IMAGE")
	cmd = fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s", ctx.Cmd(), referencedACI)
	t.Logf("Running %s", referencedACI)
	spawnAndWaitOrFail(t, cmd, 0)

	t.Logf("Retrieving stage1 image ID")
	stage1ImageID, err := getImageID(ctx, stage1App)
	if err != nil {
		t.Fatalf("rkt didn't terminate correctly: %v", err)
	}

	t.Logf("Retrieving %s image ID", referencedApp)
	referencedImageID, err := getImageID(ctx, referencedApp)
	if err != nil {
		t.Fatalf("rkt didn't terminate correctly: %v", err)
	}

	t.Logf("Retrieving %s image ID", unreferencedApp)
	unreferencedImageID, err := getImageID(ctx, unreferencedApp)
	if err != nil {
		t.Fatalf("rkt didn't terminate correctly: %v", err)
	}

	t.Logf("Removing stage1 image (should work)")
	removeImage(t, ctx, stage1ImageID)

	t.Logf("Removing image for app %s (should work)", referencedApp)
	removeImage(t, ctx, referencedImageID)

	t.Logf("Removing image for app %s (should work)", unreferencedApp)
	removeImage(t, ctx, unreferencedImageID)
}

func TestImageRunRmDuplicate(t *testing.T) {
	imageFile := patchTestACI(unreferencedACI, fmt.Sprintf("--name=%s", unreferencedApp))
	defer os.Remove(imageFile)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --insecure-options=image fetch %s", ctx.Cmd(), imageFile)
	t.Logf("Fetching %s", imageFile)
	spawnAndWaitOrFail(t, cmd, 0)

	// at this point we know that RKT_INSPECT_IMAGE env var is not empty
	referencedACI := os.Getenv("RKT_INSPECT_IMAGE")
	cmd = fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s", ctx.Cmd(), referencedACI)
	t.Logf("Running %s", referencedACI)
	spawnAndWaitOrFail(t, cmd, 0)

	t.Logf("Retrieving %s image ID", referencedApp)
	referencedImageID, err := getImageID(ctx, referencedApp)
	if err != nil {
		t.Fatalf("rkt didn't terminate correctly: %v", err)
	}

	t.Logf("Retrieving %s image ID", unreferencedApp)
	unreferencedImageID, err := getImageID(ctx, unreferencedApp)
	if err != nil {
		t.Fatalf("rkt didn't terminate correctly: %v", err)
	}

	t.Logf("Removing image for app %s (should work)", referencedApp)
	removeImage(t, ctx, referencedApp, referencedImageID)

	t.Logf("Removing image for app %s (should work)", unreferencedApp)
	removeImage(t, ctx, unreferencedImageID, unreferencedApp)
}

func TestImagePrepareRmNameRun(t *testing.T) {
	imageFile := patchTestACI(unreferencedACI, fmt.Sprintf("--name=%s", unreferencedApp))
	defer os.Remove(imageFile)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --insecure-options=image fetch %s", ctx.Cmd(), imageFile)
	t.Logf("Fetching %s", imageFile)
	spawnAndWaitOrFail(t, cmd, 0)

	// at this point we know that RKT_INSPECT_IMAGE env var is not empty
	referencedACI := os.Getenv("RKT_INSPECT_IMAGE")
	cmds := strings.Fields(ctx.Cmd())
	prepareCmd := exec.Command(cmds[0], cmds[1:]...)
	prepareCmd.Args = append(prepareCmd.Args, "--insecure-options=image", "prepare", referencedACI)
	output, err := prepareCmd.Output()
	if err != nil {
		t.Fatalf("Cannot read the output: %v", err)
	}

	podIDStr := strings.TrimSpace(string(output))
	podID, err := types.NewUUID(podIDStr)
	if err != nil {
		t.Fatalf("%q is not a valid UUID: %v", podIDStr, err)
	}

	t.Logf("Retrieving stage1 image name")
	stage1ImageName := getImageName(t, ctx, stage1App)

	t.Logf("Removing stage1 image (should work)")
	removeImage(t, ctx, stage1ImageName)

	t.Logf("Removing image for app %s (should work)", referencedApp)
	removeImage(t, ctx, referencedApp)

	t.Logf("Removing image for app %s (should work)", unreferencedApp)
	removeImage(t, ctx, unreferencedApp)

	cmd = fmt.Sprintf("%s run-prepared --mds-register=false %s", ctx.Cmd(), podID.String())
	t.Logf("Running %s", referencedACI)
	spawnAndWaitOrFail(t, cmd, 0)
}

func TestImagePrepareRmIDRun(t *testing.T) {
	imageFile := patchTestACI(unreferencedACI, fmt.Sprintf("--name=%s", unreferencedApp))
	defer os.Remove(imageFile)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --insecure-options=image fetch %s", ctx.Cmd(), imageFile)
	t.Logf("Fetching %s", imageFile)
	spawnAndWaitOrFail(t, cmd, 0)

	// at this point we know that RKT_INSPECT_IMAGE env var is not empty
	referencedACI := os.Getenv("RKT_INSPECT_IMAGE")
	cmds := strings.Fields(ctx.Cmd())
	prepareCmd := exec.Command(cmds[0], cmds[1:]...)
	prepareCmd.Args = append(prepareCmd.Args, "--insecure-options=image", "prepare", referencedACI)
	output, err := prepareCmd.Output()
	if err != nil {
		t.Fatalf("Cannot read the output: %v", err)
	}

	podIDStr := strings.TrimSpace(string(output))
	podID, err := types.NewUUID(podIDStr)
	if err != nil {
		t.Fatalf("%q is not a valid UUID: %v", podIDStr, err)
	}

	t.Logf("Retrieving stage1 imageID")
	stage1ImageID, err := getImageID(ctx, stage1App)
	if err != nil {
		t.Fatalf("rkt didn't terminate correctly: %v", err)
	}

	t.Logf("Retrieving %s image ID", referencedApp)
	referencedImageID, err := getImageID(ctx, referencedApp)
	if err != nil {
		t.Fatalf("rkt didn't terminate correctly: %v", err)
	}

	t.Logf("Retrieving %s image ID", unreferencedApp)
	unreferencedImageID, err := getImageID(ctx, unreferencedApp)
	if err != nil {
		t.Fatalf("rkt didn't terminate correctly: %v", err)
	}

	t.Logf("Removing stage1 image (should work)")
	removeImage(t, ctx, stage1ImageID)

	t.Logf("Removing image for app %s (should work)", referencedApp)
	removeImage(t, ctx, referencedImageID)

	t.Logf("Removing image for app %s (should work)", unreferencedApp)
	removeImage(t, ctx, unreferencedImageID)

	cmd = fmt.Sprintf("%s run-prepared --mds-register=false %s", ctx.Cmd(), podID.String())
	t.Logf("Running %s", referencedACI)
	spawnAndWaitOrFail(t, cmd, 0)
}

func TestImagePrepareRmDuplicate(t *testing.T) {
	imageFile := patchTestACI(unreferencedACI, fmt.Sprintf("--name=%s", unreferencedApp))
	defer os.Remove(imageFile)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	cmd := fmt.Sprintf("%s --insecure-options=image fetch %s", ctx.Cmd(), imageFile)
	t.Logf("Fetching %s", imageFile)
	spawnAndWaitOrFail(t, cmd, 0)

	// at this point we know that RKT_INSPECT_IMAGE env var is not empty
	referencedACI := os.Getenv("RKT_INSPECT_IMAGE")
	cmds := strings.Fields(ctx.Cmd())
	prepareCmd := exec.Command(cmds[0], cmds[1:]...)
	prepareCmd.Args = append(prepareCmd.Args, "--insecure-options=image", "prepare", referencedACI)
	output, err := prepareCmd.Output()
	if err != nil {
		t.Fatalf("Cannot read the output: %v", err)
	}

	podIDStr := strings.TrimSpace(string(output))
	podID, err := types.NewUUID(podIDStr)
	if err != nil {
		t.Fatalf("%q is not a valid UUID: %v", podIDStr, err)
	}

	t.Logf("Retrieving %s image ID", referencedApp)
	referencedImageID, err := getImageID(ctx, referencedApp)
	if err != nil {
		t.Fatalf("rkt didn't terminate correctly: %v", err)
	}

	t.Logf("Retrieving %s image ID", unreferencedApp)
	unreferencedImageID, err := getImageID(ctx, unreferencedApp)
	if err != nil {
		t.Fatalf("rkt didn't terminate correctly: %v", err)
	}

	t.Logf("Removing image for app %s (should work)", referencedApp)
	removeImage(t, ctx, referencedApp, referencedImageID)

	t.Logf("Removing image for app %s (should work)", unreferencedApp)
	removeImage(t, ctx, unreferencedImageID, unreferencedApp)

	cmd = fmt.Sprintf("%s run-prepared --mds-register=false %s", ctx.Cmd(), podID.String())
	t.Logf("Running %s", referencedACI)
	spawnAndWaitOrFail(t, cmd, 0)
}

func getImageName(t *testing.T, ctx *testutils.RktRunCtx, name string) string {
	cmd := fmt.Sprintf(`/bin/sh -c "%s image list --fields=name --no-legend | grep %s"`, ctx.Cmd(), name)
	child := spawnOrFail(t, cmd)
	imageName, err := child.ReadLine()
	imageName = strings.TrimSpace(imageName)
	imageName = string(bytes.Trim([]byte(imageName), "\x00"))
	if err != nil {
		t.Fatalf("Cannot exec: %v", err)
	}
	waitOrFail(t, child, 0)
	return imageName
}

func getImageID(ctx *testutils.RktRunCtx, name string) (string, error) {
	cmd := fmt.Sprintf(`/bin/sh -c "%s image list --fields=id,name --no-legend | grep %s | awk '{print $1}'"`, ctx.Cmd(), name)
	child, err := gexpect.Spawn(cmd)
	if err != nil {
		return "", fmt.Errorf("Cannot exec rkt: %v", err)
	}
	imageID, err := child.ReadLine()
	imageID = strings.TrimSpace(imageID)
	imageID = string(bytes.Trim([]byte(imageID), "\x00"))
	if err != nil {
		return "", fmt.Errorf("Cannot exec: %v", err)
	}
	if err := child.Wait(); err != nil {
		return "", fmt.Errorf("rkt didn't terminate correctly: %v", err)
	}
	return imageID, nil
}

func removeImage(t *testing.T, ctx *testutils.RktRunCtx, images ...string) {
	cmd := fmt.Sprintf("%s image rm %s", ctx.Cmd(), strings.Join(images, " "))
	child, err := gexpect.Spawn(cmd)
	if err != nil {
		t.Fatalf("Cannot exec: %v", err)
	}
	if err := expectWithOutput(child, rmImageOk); err != nil {
		t.Fatalf("Expected %q but not found: %v", rmImageOk, err)
	}
	if err := child.Wait(); err != nil {
		t.Fatalf("rkt didn't terminate correctly: %v", err)
	}
}
