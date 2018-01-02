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

// +build host coreos src

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

func TestVolumes(t *testing.T) {
	NewVolumesTest().Execute(t)
}

var volSysfsTests = []struct {
	name          string
	imgPatch      string
	cmdParams     string
	expected      string
	expectedError bool
}{
	{
		"Without any mount",
		"--name=base",
		"",
		"/sys/class: mode: drwxr-xr-x",
		false,
	},
	{
		"With /sys as an empty volume",
		"--mounts=sysdir,path=/sys,readOnly=false",
		"",
		"stat /sys/class: no such file or directory",
		true,
	},
	{
		"With /sys bind-mounted from the host",
		"--mounts=sysdir,path=/sys,readOnly=false",
		"--volume=sysdir,kind=host,source=/sys,readOnly=false",
		"/sys/class: mode: drwxr-xr-x",
		false,
	},
	{
		"With /sys/class bind-mounted from the host",
		"--mounts=sysdir,path=/sys/class,readOnly=false",
		"--volume=sysdir,kind=host,source=/dev/null,readOnly=false",
		"/sys/class: mode: Dcrw-rw-rw-",
		false,
	},
}

// TestVolumeSysfs checks that sysfs is available for the apps, but only if
// the app does not have mount points in /sys or a subdirectory.
// See: https://github.com/coreos/rkt/issues/2874
func TestVolumeSysfs(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	for i, tt := range volSysfsTests {
		t.Logf("Running test #%v: %s", i, tt.name)

		img := patchTestACI("rkt-sysfs.aci", tt.imgPatch)
		defer os.Remove(img)
		cmd := fmt.Sprintf(`%s --debug --insecure-options=image --set-env=FILE=/sys/class run %s %s --exec /inspect -- --stat-file`,
			ctx.Cmd(), tt.cmdParams, img)

		runRktAndCheckOutput(t, cmd, tt.expected, tt.expectedError)
	}
}
