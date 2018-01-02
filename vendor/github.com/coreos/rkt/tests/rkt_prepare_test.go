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
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

// TestPrepareConflictingFlags tests that 'rkt prepare' will complain and abort
// if conflicting flags are specified together with a pod manifest.
func TestPrepareConflictingFlags(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	prepareConflictingFlagsMsg := "conflicting flags set with --pod-manifest (see --help)"
	podManifestFlag := "--pod-manifest=/dev/null"
	conflictingFlags := []struct {
		flag string
		args string
	}{
		{"--inherit-env", ""},
		{"--no-store", ""},
		{"--store-only", ""},
		{"--port=", "foo:80"},
		{"--set-env=", "foo=bar"},
		{"--volume=", "foo,kind=host,source=/tmp"},
		{"--mount=", "volume=foo,target=/tmp --volume=foo,kind=host,source=/tmp"},
	}
	imageConflictingFlags := []struct {
		flag string
		args string
	}{
		{"--exec=", "/bin/sh"},
		{"--user=", "user_foo"},
		{"--group=", "group_foo"},
	}

	for _, cf := range conflictingFlags {
		cmd := fmt.Sprintf("%s prepare %s %s%s", ctx.Cmd(), podManifestFlag, cf.flag, cf.args)
		runRktAndCheckOutput(t, cmd, prepareConflictingFlagsMsg, true)
	}
	for _, icf := range imageConflictingFlags {
		cmd := fmt.Sprintf("%s prepare dummy-image.aci %s %s%s", ctx.Cmd(), podManifestFlag, icf.flag, icf.args)
		runRktAndCheckOutput(t, cmd, prepareConflictingFlagsMsg, true)
	}
}
