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
	"io/ioutil"
	"os"
	"testing"

	"github.com/coreos/rkt/pkg/aci/acitest"
	"github.com/coreos/rkt/tests/testutils"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

func TestRunOverrideExec(t *testing.T) {
	// noappManifest specifies an image manifest configuration without
	// an application section.
	noappManifest := schema.ImageManifest{
		Name: "coreos.com/rkt-inspect",
		Labels: types.Labels{
			{"version", "1.25.0"},
			{"arch", "amd64"},
			{"os", "linux"},
		},
	}

	noappManifestStr, err := acitest.ImageManifestString(&noappManifest)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	noappManifestFile := "noapp-manifest.json"
	if err := ioutil.WriteFile(noappManifestFile, []byte(noappManifestStr), 0600); err != nil {
		t.Fatalf("Cannot write noapp manifest: %v", err)
	}
	defer os.Remove(noappManifestFile)
	noappImage := patchTestACI("rkt-image-without-exec.aci", fmt.Sprintf("--manifest=%s", noappManifestFile))
	defer os.Remove(noappImage)
	execImage := patchTestACI("rkt-exec-override.aci", "--exec=/inspect")
	defer os.Remove(execImage)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	for _, tt := range []struct {
		rktCmd        string
		expectedRegex string
	}{
		{
			// Sanity check - make sure no --exec override prints the expected exec invocation
			rktCmd:        fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s -- --print-exec", ctx.Cmd(), execImage),
			expectedRegex: "inspect execed as: /inspect",
		},
		{
			// Now test overriding the entrypoint (which is a symlink to /inspect so should behave identically)
			rktCmd:        fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s --exec /inspect-link -- --print-exec", ctx.Cmd(), execImage),
			expectedRegex: "inspect execed as: /inspect-link",
		},
		{
			// Test overriding the entrypoint with a relative path
			rktCmd:        fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s --exec inspect-link-bin -- --print-exec", ctx.Cmd(), execImage),
			expectedRegex: "inspect execed as: .*inspect-link-bin",
		},

		{
			// Test overriding the entrypoint with a missing app section
			rktCmd:        fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s --exec /inspect -- --print-exec", ctx.Cmd(), noappImage),
			expectedRegex: "inspect execed as: /inspect",
		},
	} {
		runRktAndCheckRegexOutput(t, tt.rktCmd, tt.expectedRegex)
	}
}

func TestRunPreparedOverrideExec(t *testing.T) {
	execImage := patchTestACI("rkt-exec-override.aci", "--exec=/inspect")
	defer os.Remove(execImage)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	var rktCmd, uuid, expected string

	// Sanity check - make sure no --exec override prints the expected exec invocation
	rktCmd = fmt.Sprintf("%s prepare --insecure-options=image %s -- --print-exec", ctx.Cmd(), execImage)
	uuid = runRktAndGetUUID(t, rktCmd)

	rktCmd = fmt.Sprintf("%s run-prepared --mds-register=false %s", ctx.Cmd(), uuid)
	expected = "inspect execed as: /inspect"
	runRktAndCheckOutput(t, rktCmd, expected, false)

	// Now test overriding the entrypoint (which is a symlink to /inspect so should behave identically)
	rktCmd = fmt.Sprintf("%s prepare --insecure-options=image %s --exec /inspect-link -- --print-exec", ctx.Cmd(), execImage)
	uuid = runRktAndGetUUID(t, rktCmd)

	rktCmd = fmt.Sprintf("%s run-prepared --mds-register=false %s", ctx.Cmd(), uuid)
	expected = "inspect execed as: /inspect-link"
	runRktAndCheckOutput(t, rktCmd, expected, false)
}
