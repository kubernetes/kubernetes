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
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coreos/rkt/tests/testutils"
)

var volTests = []struct {
	rktCmd       string
	expect       string
	expectedExit int
}{
	// Check that we can read files in the ACI
	{
		`/bin/sh -c "export FILE=/dir1/file ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true ^READ_FILE^"`,
		`<<<dir1>>>`,
		0,
	},
	// Check that we can read files from a volume (both ro and rw)
	{
		`/bin/sh -c "export FILE=/dir1/file ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true --volume=dir1,kind=host,source=^TMPDIR^ --mount=volume=dir1,target=dir1 ^VOL_RW_READ_FILE^"`,
		`<<<host>>>`,
		0,
	},
	{
		`/bin/sh -c "export FILE=/dir1/file ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true --volume=dir1,kind=host,source=^TMPDIR^ --mount=volume=dir1,target=dir1 ^VOL_RO_READ_FILE^"`,
		`<<<host>>>`,
		0,
	},
	// Check that we can write to files in the ACI
	{
		`/bin/sh -c "export FILE=/dir1/file CONTENT=1 ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true ^WRITE_FILE^"`,
		`<<<1>>>`,
		0,
	},
	// Check that we can write files to a volume (both ro and rw)
	{
		`/bin/sh -c "export FILE=/dir1/file CONTENT=2 ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true --volume=dir1,kind=host,source=^TMPDIR^ --mount=volume=dir1,target=dir1 ^VOL_RW_WRITE_FILE^"`,
		`<<<2>>>`,
		0,
	},
	{
		`/bin/sh -c "export FILE=/dir1/file CONTENT=3 ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true --volume=dir1,kind=host,source=^TMPDIR^ --mount=volume=dir1,target=dir1 ^VOL_RO_WRITE_FILE^"`,
		`Cannot write to file "/dir1/file": open /dir1/file: read-only file system`,
		1,
	},
	// Check that the volume still contains the file previously written
	{
		`/bin/sh -c "export FILE=/dir1/file ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true --volume=dir1,kind=host,source=^TMPDIR^ --mount=volume=dir1,target=dir1 ^VOL_RO_READ_FILE^"`,
		`<<<2>>>`,
		0,
	},
	// Check that injecting a rw mount/volume works without any mountpoint in the image manifest
	{
		`/bin/sh -c "export FILE=/dir1/file CONTENT=1 ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true --volume=dir1,kind=host,source=^TMPDIR^ ^VOL_ADD_MOUNT_RW^ --mount=volume=dir1,target=dir1"`,
		`<<<1>>>`,
		0,
	},
	// Check that injecting a ro mount/volume works without any mountpoint in the image manifest
	{
		`/bin/sh -c "export FILE=/dir1/file CONTENT=1 ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true --volume=dir1,kind=host,source=^TMPDIR^,readOnly=true ^VOL_ADD_MOUNT_RO^ --mount=volume=dir1,target=dir1"`,
		`Cannot write to file "/dir1/file": open /dir1/file: read-only file system`,
		1,
	},
	// Check that an implicit empty volume is created if the user didn't provide it but there's a mount point in the app
	{
		`/bin/sh -c "export FILE=/dir1/file CONTENT=1 ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true ^VOL_RW_WRITE_FILE^"`,
		`<<<1>>>`,
		0,
	},
}

func TestVolumes(t *testing.T) {
	readFileImage := patchTestACI("rkt-inspect-read-file.aci", "--exec=/inspect --read-file")
	defer os.Remove(readFileImage)
	writeFileImage := patchTestACI("rkt-inspect-write-file.aci", "--exec=/inspect --write-file --read-file")
	defer os.Remove(writeFileImage)
	volRwReadFileImage := patchTestACI("rkt-inspect-vol-rw-read-file.aci", "--exec=/inspect --read-file", "--mounts=dir1,path=/dir1,readOnly=false")
	defer os.Remove(volRwReadFileImage)
	volRwWriteFileImage := patchTestACI("rkt-inspect-vol-rw-write-file.aci", "--exec=/inspect --write-file --read-file", "--mounts=dir1,path=/dir1,readOnly=false")
	defer os.Remove(volRwWriteFileImage)
	volRoReadFileImage := patchTestACI("rkt-inspect-vol-ro-read-file.aci", "--exec=/inspect --read-file", "--mounts=dir1,path=/dir1,readOnly=true")
	defer os.Remove(volRoReadFileImage)
	volRoWriteFileImage := patchTestACI("rkt-inspect-vol-ro-write-file.aci", "--exec=/inspect --write-file --read-file", "--mounts=dir1,path=/dir1,readOnly=true")
	defer os.Remove(volRoWriteFileImage)
	volAddMountRwImage := patchTestACI("rkt-inspect-vol-add-mount-rw.aci", "--exec=/inspect --write-file --read-file")
	defer os.Remove(volAddMountRwImage)
	volAddMountRoImage := patchTestACI("rkt-inspect-vol-add-mount-ro.aci", "--exec=/inspect --write-file --read-file")
	defer os.Remove(volAddMountRoImage)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	tmpdir := createTempDirOrPanic("rkt-tests.")
	defer os.RemoveAll(tmpdir)

	tmpfile := filepath.Join(tmpdir, "file")
	if err := ioutil.WriteFile(tmpfile, []byte("host"), 0600); err != nil {
		t.Fatalf("Cannot create temporary file: %v", err)
	}

	for i, tt := range volTests {
		cmd := strings.Replace(tt.rktCmd, "^TMPDIR^", tmpdir, -1)
		cmd = strings.Replace(cmd, "^RKT_BIN^", ctx.Cmd(), -1)
		cmd = strings.Replace(cmd, "^READ_FILE^", readFileImage, -1)
		cmd = strings.Replace(cmd, "^WRITE_FILE^", writeFileImage, -1)
		cmd = strings.Replace(cmd, "^VOL_RO_READ_FILE^", volRoReadFileImage, -1)
		cmd = strings.Replace(cmd, "^VOL_RO_WRITE_FILE^", volRoWriteFileImage, -1)
		cmd = strings.Replace(cmd, "^VOL_RW_READ_FILE^", volRwReadFileImage, -1)
		cmd = strings.Replace(cmd, "^VOL_RW_WRITE_FILE^", volRwWriteFileImage, -1)
		cmd = strings.Replace(cmd, "^VOL_ADD_MOUNT_RW^", volAddMountRwImage, -1)
		cmd = strings.Replace(cmd, "^VOL_ADD_MOUNT_RO^", volAddMountRoImage, -1)

		t.Logf("Running test #%v", i)
		child := spawnOrFail(t, cmd)
		defer waitOrFail(t, child, tt.expectedExit)

		if err := expectTimeoutWithOutput(child, tt.expect, time.Minute); err != nil {
			fmt.Printf("Command: %s\n", cmd)
			t.Fatalf("Expected %q but not found #%v: %v", tt.expect, i, err)
		}
	}
}
