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

// +build !fly

package main

import (
	"crypto/sha1"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/tests/testutils"
)

// stage1/prepare-app will populate /etc/hosts with a 127.0.0.1 entry when absent,
// or leave it alone when present.  These tests verify that behavior.

/*
	The stage0 can create an etc hosts

	prepare-app will overwrite the app's /etc/hosts if the stage0 created one
	Otherwise, it will create a fallback /etc/hosts if the stage2 does not have one
*/
func TestEtcHosts(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	tmpdir := mustTempDir("rkt-tests.")
	defer os.RemoveAll(tmpdir)
	tmpetc := mustTempDir("rkt-tests-etc.")
	defer os.RemoveAll(tmpetc)

	tmpfile := filepath.Join(tmpetc, "hosts")
	if err := ioutil.WriteFile(tmpfile, []byte(`<<<preexisting etc>>>`), 0600); err != nil {
		t.Fatalf("Cannot create etc/hosts file: %v", err)
	}

	dat, err := ioutil.ReadFile("/etc/hosts")
	if err != nil {
		t.Fatal("Could not read host's /etc/hosts", err)
	}

	sum := fmt.Sprintf("%x", sha1.Sum(dat))

	tests := []struct {
		rktArgs     string
		inspectArgs string
		expectRegEx string
	}{
		{
			fmt.Sprintf("--volume hosts,kind=host,source=%s", tmpfile),
			"--mount volume=hosts,target=/etc/hosts --exec=/inspect -- -file-name=/etc/hosts -read-file",
			"<<<preexisting etc>>>",
		},
		{ // Test that with no /etc/hosts, the fallback is created with container hostname
			"",
			"--exec=/inspect -- -file-name=/etc/hosts -read-file",
			"127.0.0.1 localhost localhost.domain localhost4 localhost4.localdomain4 rkt-[a-z0-9-]{36}",
		},
		{ // Test that with --hosts-entry=host, the host's is copied
			"--hosts-entry=host",
			"--exec=/inspect -- -file-name=/etc/hosts -hash-file",
			sum,
		},
		{ // test that we create our own
			"--hosts-entry=128.66.0.99=host1",
			"--exec=/inspect -- -file-name=/etc/hosts -read-file",
			"128.66.0.99 host1",
		},
	}

	for i, tt := range tests {
		cmd := fmt.Sprintf(
			"%s run --insecure-options=image %s %s %s",
			ctx.Cmd(),
			tt.rktArgs,
			getInspectImagePath(),
			tt.inspectArgs)

		t.Logf("Running test #%v: %v", i, cmd)

		child, err := gexpect.Spawn(cmd)
		if err != nil {
			t.Fatalf("Cannot exec rkt #%v: %v", i, err)
		}

		_, out, err := expectRegexTimeoutWithOutput(child, tt.expectRegEx, 30*time.Second)
		if err != nil {
			t.Fatalf("Test %d %v output: %v", i, err, out)
		}

		waitOrFail(t, child, 0)
	}
}

// stage1/prepare-app will bind mount /proc/sys/kernel/hostname on /etc/hostname,
// see https://github.com/coreos/rkt/issues/2657
var etcHostnameTests = []struct {
	aciBuildArgs   []string
	runArgs        []string
	expectedOutput string
	expectErr      bool
}{
	{
		[]string{"--exec=/inspect -file-name /etc/hostname -stat-file"},
		nil,
		`/etc/hostname: mode: -rw-r--r--`,
		false,
	},
	{
		[]string{"--exec=/inspect -file-name /etc/hostname -read-file"},
		[]string{"--hostname custom_hostname_setting"},
		`<<<custom_hostname_setting`,
		false,
	},
	{
		[]string{"--exec=/inspect -file-name /etc/hostname -read-file"},
		nil,
		`<<<rkt-`,
		false,
	},
}

func TestPrepareAppCheckEtcHostname(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	for i, ti := range etcHostnameTests {

		aciFileName := patchTestACI(fmt.Sprintf("rkt-inspect-hostname-%d.aci", i), ti.aciBuildArgs...)
		defer os.Remove(aciFileName)

		rktCmd := fmt.Sprintf("%s --insecure-options=image run %s %s", ctx.Cmd(), aciFileName, strings.Join(ti.runArgs, " "))
		runRktAndCheckOutput(t, rktCmd, ti.expectedOutput, ti.expectErr)
	}
}
