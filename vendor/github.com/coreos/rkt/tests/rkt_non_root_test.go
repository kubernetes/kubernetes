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
	"fmt"
	"os"
	"testing"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/tests/testutils"
)

// TestNonRootReadInfo tests that non-root users that can do rkt list, rkt image list.
func TestNonRootReadInfo(t *testing.T) {
	if !common.SupportsUserNS() {
		t.Skip("User namespaces are not supported on this host.")
	}

	if err := checkUserNS(); err != nil {
		t.Skip("User namespaces don't work on this host.")
	}

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	gid, err := common.LookupGid(common.RktGroup)
	if err != nil {
		t.Skipf("Skipping the test because there's no %q group", common.RktGroup)
	}

	if err := ctx.SetupDataDir(); err != nil {
		t.Fatalf("failed to setup data dir: %v", err)
	}

	// Launch some pods, this creates the environment for later testing.
	imgs := []struct {
		name    string
		msg     string
		imgFile string
	}{
		{name: "inspect-1", msg: "foo-1"},
		{name: "inspect-2", msg: "foo-2"},
		{name: "inspect-3", msg: "foo-3"},
	}

	for i, img := range imgs {
		imgName := fmt.Sprintf("rkt-%s.aci", img.name)
		imgs[i].imgFile = patchTestACI(imgName, fmt.Sprintf("--name=%s", img.name), fmt.Sprintf("--exec=/inspect --print-msg=%s --exit-code=0", img.msg))
		defer os.Remove(imgs[i].imgFile)
	}

	runCmds := []string{
		// Run with overlay, without private-users.
		fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s", ctx.Cmd(), imgs[0].imgFile),

		// Run without overlay, without private-users.
		fmt.Sprintf("%s --insecure-options=image run --no-overlay --mds-register=false %s", ctx.Cmd(), imgs[1].imgFile),

		// Run without overlay, with private-users.
		fmt.Sprintf("%s --insecure-options=image run --no-overlay --private-users --mds-register=false %s", ctx.Cmd(), imgs[2].imgFile),
	}

	for i, cmd := range runCmds {
		t.Logf("#%d: Running %s", i, cmd)
		runRktAndCheckOutput(t, cmd, imgs[i].msg, false)
	}

	imgListCmd := fmt.Sprintf("%s image list", ctx.Cmd())
	t.Logf("Running %s", imgListCmd)
	runRktAsGidAndCheckOutput(t, imgListCmd, "inspect-", false, gid)
}

// TestNonRootFetchRmGCImage tests that non-root users can remove images fetched by themselves but
// cannot remove images fetched by root, or gc any images.
func TestNonRootFetchRmGCImage(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	gid, err := common.LookupGid(common.RktGroup)
	if err != nil {
		t.Skipf("Skipping the test because there's no %q group", common.RktGroup)
	}

	if err := ctx.SetupDataDir(); err != nil {
		t.Fatalf("failed to setup data dir: %v", err)
	}

	rootImg := patchTestACI("rkt-inspect-root-rm.aci", "--exec=/inspect --print-msg=foobar")
	defer os.Remove(rootImg)
	rootImgHash := importImageAndFetchHash(t, ctx, "", rootImg)

	// Launch/gc a pod so we can test non-root image gc.
	runCmd := fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s", ctx.Cmd(), rootImg)
	runRktAndCheckOutput(t, runCmd, "foobar", false)

	ctx.RunGC()

	// Should not be able to do image gc.
	imgGCCmd := fmt.Sprintf("%s image gc", ctx.Cmd())
	t.Logf("Running %s", imgGCCmd)
	runRktAsGidAndCheckOutput(t, imgGCCmd, "permission denied", true, gid)

	// Should not be able to remove the image fetched by root.
	imgRmCmd := fmt.Sprintf("%s image rm %s", ctx.Cmd(), rootImgHash)
	t.Logf("Running %s", imgRmCmd)
	runRktAsGidAndCheckOutput(t, imgRmCmd, "permission denied", true, gid)

	// Should be able to remove the image fetched by ourselves.
	nonrootImg := patchTestACI("rkt-inspect-non-root-rm.aci", "--exec=/inspect")
	defer os.Remove(nonrootImg)
	nonrootImgHash := importImageAndFetchHashAsGid(t, ctx, nonrootImg, "", gid)

	imgRmCmd = fmt.Sprintf("%s image rm %s", ctx.Cmd(), nonrootImgHash)
	t.Logf("Running %s", imgRmCmd)
	runRktAsGidAndCheckOutput(t, imgRmCmd, "successfully removed", false, gid)
}
