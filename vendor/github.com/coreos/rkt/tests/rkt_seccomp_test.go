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
	"strings"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

const (
	baseApp = `--exec=/inspect -file-name / -stat-file`
)

var seccompTestCases = []struct {
	name           string
	aciBuildArgs   []string
	runArgs        []string
	expectedOutput string
	expectErr      bool
}{
	{
		`default: aci seccomp filter or rkt whitelist`,
		[]string{baseApp},
		nil,
		`/: mode: d`,
		false,
	},
	{
		`seccomp opt-out: no filtering`,
		[]string{baseApp, "--seccomp-mode=retain", "--seccomp-set=@appc.io/all"},
		nil,
		`/: mode: d`,
		false,
	},
	{
		`seccomp opt-in: rkt whitelist`,
		[]string{baseApp, "--seccomp-mode=remove", "--seccomp-set=@appc.io/empty"},
		nil,
		`/: mode: d`,
		false,
	},
	{
		`remove-set rkt blacklist`,
		[]string{baseApp, "--seccomp-mode=remove", "--seccomp-set=@rkt/default-blacklist"},
		nil,
		`/: mode: d`,
		false,
	},
	{
		`remove-set unrelated blacklist`,
		[]string{baseApp, "--seccomp-mode=remove", "--seccomp-set=reboot"},
		nil,
		`/: mode: d`,
		false,
	},
	{
		`remove-set blacklist stat with custom error`,
		[]string{baseApp, "--seccomp-mode=remove,errno=EXFULL", "--seccomp-set=stat"},
		nil,
		"exchange full",
		true,
	},
	{
		`retain-set rkt whitelist`,
		[]string{baseApp, "--seccomp-mode=retain", "--seccomp-set=@rkt/default-whitelist"},
		nil,
		`/: mode: d`,
		false,
	},
	{
		`retain-set docker whitelist`,
		[]string{baseApp, "--seccomp-mode=retain", "--seccomp-set=@docker/default-whitelist"},
		nil,
		`/: mode: d`,
		false,
	},
	{
		`remove-set unprivileged group`,
		[]string{baseApp, "--seccomp-mode=remove", "--seccomp-set=reboot"},
		[]string{"--group=100"},
		`/: mode: d`,
		false,
	},
	{
		`remove-set unprivileged user`,
		[]string{baseApp, "--seccomp-mode=remove", "--seccomp-set=reboot"},
		[]string{"--user=1000"},
		`/: mode: d`,
		false,
	},
	{
		`CLI override whitelist all`,
		[]string{baseApp, "--seccomp-mode=remove,errno=EXFULL", "--seccomp-set=stat"},
		[]string{"--seccomp=mode=retain,@appc.io/all"},
		`/: mode: d`,
		false,
	},
	{
		`CLI override blacklist stat with custom error`,
		[]string{baseApp},
		[]string{"--seccomp=mode=remove,errno=EXFULL,stat"},
		"exchange full",
		true,
	},
	{
		`insecure-options fake override: remove-set blacklist stat with custom error`,
		[]string{baseApp, "--seccomp-mode=remove,errno=EMULTIHOP", "--seccomp-set=stat"},
		[]string{"--insecure-options=image,ondisk,capabilities,paths"},
		"multihop attempted",
		true,
	},
	{
		`insecure-options simple override: remove-set blacklist stat with custom error`,
		[]string{baseApp, "--seccomp-mode=remove,errno=EMULTIHOP", "--seccomp-set=stat"},
		[]string{"--insecure-options=image,seccomp"},
		`/: mode: d`,
		false,
	},
	{
		`insecure-options complete override: remove-set blacklist stat with custom error`,
		[]string{baseApp, "--seccomp-mode=remove,errno=EMULTIHOP", "--seccomp-set=stat"},
		[]string{"--insecure-options=image,all-run"},
		`/: mode: d`,
		false,
	},
}

func TestAppIsolatorSeccomp(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	for i, ti := range seccompTestCases {
		t.Logf("Running seccomp test #%d: %v", i, ti.name)
		aciFileName := patchTestACI(fmt.Sprintf("rkt-inspect-isolators-%d.aci", i), ti.aciBuildArgs...)
		defer os.Remove(aciFileName)

		rktCmd := fmt.Sprintf("%s --insecure-options=image run --debug %s %s", ctx.Cmd(), aciFileName, strings.Join(ti.runArgs, " "))
		runRktAndCheckOutput(t, rktCmd, ti.expectedOutput, ti.expectErr)
	}
}
