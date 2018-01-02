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
	"os"
	"strings"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

var envTests = []struct {
	runCmd    string
	runExpect string
	sleepCmd  string
	enterCmd  string
}{
	{
		`^RKT_BIN^ --debug --insecure-options=image run --mds-register=false ^PRINT_VAR_FROM_MANIFEST^`,
		"VAR_FROM_MANIFEST=manifest",
		`^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --interactive ^SLEEP^`,
		`/bin/sh -c "^RKT_BIN^ --debug enter $(^RKT_BIN^ list --full|grep running|awk '{print $1}') /inspect --print-env=VAR_FROM_MANIFEST"`,
	},
	{
		`^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --set-env=VAR_OTHER=setenv ^PRINT_VAR_OTHER^`,
		"VAR_OTHER=setenv",
		`^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --interactive --set-env=VAR_OTHER=setenv ^SLEEP^`,
		`/bin/sh -c "^RKT_BIN^ --debug enter $(^RKT_BIN^ list --full|grep running|awk '{print $1}') /inspect --print-env=VAR_OTHER"`,
	},
	{
		`^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --set-env=VAR_FROM_MANIFEST=setenv ^PRINT_VAR_FROM_MANIFEST^`,
		"VAR_FROM_MANIFEST=setenv",
		`^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --interactive --set-env=VAR_FROM_MANIFEST=setenv ^SLEEP^`,
		`/bin/sh -c "^RKT_BIN^ --debug enter $(^RKT_BIN^ list --full|grep running|awk '{print $1}') /inspect --print-env=VAR_FROM_MANIFEST"`,
	},
	{
		`/bin/sh -c "export VAR_OTHER=host ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true ^PRINT_VAR_OTHER^"`,
		"VAR_OTHER=host",
		`/bin/sh -c "export VAR_OTHER=host ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --interactive --inherit-env=true ^SLEEP^"`,
		`/bin/sh -c "export VAR_OTHER=host ; ^RKT_BIN^ --debug enter $(^RKT_BIN^ list --full|grep running|awk '{print $1}') /inspect --print-env=VAR_OTHER"`,
	},
	{
		`^RKT_BIN^ --insecure-options=image run --mds-register=false --set-env=TERM=dumb ^PRINT_TERM_HOST^`,
		"TERM=dumb",
		`^RKT_BIN^ --insecure-options=image run --mds-register=false --interactive --inherit-env=false ^SLEEP^`,
		`/bin/sh -c "export TERM=dumb ; ^RKT_BIN^ enter $(^RKT_BIN^ list --full|grep running|awk '{print $1}') /inspect --print-env=TERM"`,
	},
	{
		`^RKT_BIN^ --insecure-options=image run --mds-register=false --set-env=TERM=^HOST_TERM^ ^PRINT_TERM_HOST^`,
		"TERM=^HOST_TERM^",
		`^RKT_BIN^ --insecure-options=image run --mds-register=false --interactive --inherit-env=false ^SLEEP^`,
		`/bin/sh -c "^RKT_BIN^ enter $(^RKT_BIN^ list --full|grep running|awk '{print $1}') /inspect --print-env=TERM"`,
	},
	{
		`/bin/sh -c "export VAR_FROM_MANIFEST=host ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true ^PRINT_VAR_FROM_MANIFEST^"`,
		"VAR_FROM_MANIFEST=manifest",
		`/bin/sh -c "export VAR_FROM_MANIFEST=host ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --interactive --inherit-env=true ^SLEEP^"`,
		`/bin/sh -c "export VAR_FROM_MANIFEST=host ; ^RKT_BIN^ --debug enter $(^RKT_BIN^ list --full|grep running|awk '{print $1}') /inspect --print-env=VAR_FROM_MANIFEST"`,
	},
	{
		`/bin/sh -c "export VAR_OTHER=host ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true --set-env=VAR_OTHER=setenv ^PRINT_VAR_OTHER^"`,
		"VAR_OTHER=setenv",
		`/bin/sh -c "export VAR_OTHER=host ; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --interactive --inherit-env=true --set-env=VAR_OTHER=setenv ^SLEEP^"`,
		`/bin/sh -c "export VAR_OTHER=host ; ^RKT_BIN^ --debug enter $(^RKT_BIN^ list --full|grep running|awk '{print $1}') /inspect --print-env=VAR_OTHER"`,
	},
	{
		`/bin/sh -c "^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --set-env=VAR_OTHER=setenv --set-env-file=env_file_test.conf ^PRINT_VAR_OTHER^"`,
		"VAR_OTHER=setenv",
		`/bin/sh -c "^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --interactive --set-env=VAR_OTHER=setenv --set-env-file=env_file_test.conf ^SLEEP^"`,
		`/bin/sh -c "export VAR_OTHER=host ; ^RKT_BIN^ --debug enter $(^RKT_BIN^ list --full|grep running|awk '{print $1}') /inspect --print-env=VAR_OTHER"`,
	},
	{
		`/bin/sh -c "^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --set-env-file=env_file_test.conf ^PRINT_VAR_OTHER^"`,
		"VAR_OTHER=file",
		`/bin/sh -c "^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --interactive --set-env-file=env_file_test.conf ^SLEEP^"`,
		`/bin/sh -c "^RKT_BIN^ --debug enter $(^RKT_BIN^ list --full|grep running|awk '{print $1}') /inspect --print-env=VAR_OTHER"`,
	},
	{
		`/bin/sh -c "^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --set-env-file=env_file_test.conf ^CHECK_PATH^"`,
		"PATH is good",
		`/bin/sh -c "^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --interactive --set-env-file=env_file_test.conf ^SLEEP^"`,
		`/bin/sh -c "^RKT_BIN^ --debug enter $(^RKT_BIN^ list --full|grep running|awk '{print $1}') /inspect --check-path"`,
	},
}

func TestEnv(t *testing.T) {
	printVarFromManifestImage := patchTestACI("rkt-inspect-print-var-from-manifest.aci", "--exec=/inspect --print-env=VAR_FROM_MANIFEST")
	defer os.Remove(printVarFromManifestImage)
	printVarOtherImage := patchTestACI("rkt-inspect-print-var-other.aci", "--exec=/inspect --print-env=VAR_OTHER")
	defer os.Remove(printVarOtherImage)
	printTermHostImage := patchTestACI("rkt-inspect-print-term-host.aci", "--exec=/inspect --print-env=TERM")
	defer os.Remove(printTermHostImage)
	checkPathImage := patchTestACI("rkt-inspect-check-path.aci", "--exec=/inspect --check-path")
	defer os.Remove(checkPathImage)
	sleepImage := patchTestACI("rkt-inspect-sleep.aci", "--exec=/inspect --read-stdin")
	defer os.Remove(sleepImage)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	term := testutils.GetValueFromEnvOrPanic("TERM")
	replacePlaceholders := func(cmd string) string {
		fixed := cmd
		fixed = strings.Replace(fixed, "^RKT_BIN^", ctx.Cmd(), -1)
		fixed = strings.Replace(fixed, "^PRINT_VAR_FROM_MANIFEST^", printVarFromManifestImage, -1)
		fixed = strings.Replace(fixed, "^PRINT_VAR_OTHER^", printVarOtherImage, -1)
		fixed = strings.Replace(fixed, "^PRINT_TERM_HOST^", printTermHostImage, -1)
		fixed = strings.Replace(fixed, "^CHECK_PATH^", checkPathImage, -1)
		fixed = strings.Replace(fixed, "^SLEEP^", sleepImage, -1)
		fixed = strings.Replace(fixed, "^HOST_TERM^", term, -1)
		return fixed
	}
	for i, tt := range envTests {
		// change dynamic variables from expected result
		tt.runExpect = replacePlaceholders(tt.runExpect)

		// 'run' tests
		runCmd := replacePlaceholders(tt.runCmd)
		t.Logf("Running 'run' test #%v", i)
		runRktAndCheckOutput(t, runCmd, tt.runExpect, false)

		// 'enter' tests
		sleepCmd := replacePlaceholders(tt.sleepCmd)
		t.Logf("Running 'enter' test #%v", i)
		child := spawnOrFail(t, sleepCmd)

		if err := expectWithOutput(child, "Enter text:"); err != nil {
			t.Fatalf("Waited for the prompt but not found #%v: %v", i, err)
		}

		enterCmd := replacePlaceholders(tt.enterCmd)
		t.Logf("Running 'enter' test #%v", i)
		enterChild := spawnOrFail(t, enterCmd)

		if err := expectWithOutput(enterChild, tt.runExpect); err != nil {
			t.Fatalf("Expected %q but not found: %v", tt.runExpect, err)
		}

		waitOrFail(t, enterChild, 0)

		if err := child.SendLine("Bye"); err != nil {
			t.Fatalf("rkt couldn't write to the container: %v", err)
		}
		if err := expectWithOutput(child, "Received text: Bye"); err != nil {
			t.Fatalf("Expected Bye but not found #%v: %v", i, err)
		}

		waitOrFail(t, child, 0)
		ctx.Reset()
	}
}
