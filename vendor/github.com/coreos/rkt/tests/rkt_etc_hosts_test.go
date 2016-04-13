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
var etcHostsTests = []struct {
	rktCmd      string
	expectRegEx string
}{
	{
		`/bin/sh -c "export FILE=/etc/hosts; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true ^ETC_HOSTS_CREATE^"`,
		`127\.0\.0\.1.*`,
	},
	{
		`/bin/sh -c "export FILE=/etc/hosts; ^RKT_BIN^ --debug --insecure-options=image run --mds-register=false --inherit-env=true --volume=etc,kind=host,source=^TMPETC^ ^ETC_HOSTS_EXISTS^"`,
		`<<<preexisting etc>>>`,
	},
}

func TestPrepareAppEnsureEtcHosts(t *testing.T) {
	etcHostsCreateImage := patchTestACI("rkt-inspect-etc-hosts-create.aci", "--exec=/inspect --read-file")
	defer os.Remove(etcHostsCreateImage)
	etcHostsExistsImage := patchTestACI("rkt-inspect-etc-hosts-exists.aci", "--exec=/inspect --read-file", "--mounts=etc,path=/etc,readOnly=false")
	defer os.Remove(etcHostsExistsImage)
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	tmpdir := createTempDirOrPanic("rkt-tests.")
	defer os.RemoveAll(tmpdir)
	tmpetc := createTempDirOrPanic("rkt-tests-etc.")
	defer os.RemoveAll(tmpetc)

	tmpfile := filepath.Join(tmpetc, "hosts")
	if err := ioutil.WriteFile(tmpfile, []byte(`<<<preexisting etc>>>`), 0600); err != nil {
		t.Fatalf("Cannot create etc/hosts file: %v", err)
	}

	for i, tt := range etcHostsTests {
		cmd := strings.Replace(tt.rktCmd, "^TMPDIR^", tmpdir, -1)
		cmd = strings.Replace(cmd, "^RKT_BIN^", ctx.Cmd(), -1)
		cmd = strings.Replace(cmd, "^ETC_HOSTS_CREATE^", etcHostsCreateImage, -1)
		cmd = strings.Replace(cmd, "^ETC_HOSTS_EXISTS^", etcHostsExistsImage, -1)
		cmd = strings.Replace(cmd, "^TMPETC^", tmpetc, -1)

		t.Logf("Running test #%v: %v", i, cmd)

		child, err := gexpect.Spawn(cmd)
		if err != nil {
			t.Fatalf("Cannot exec rkt #%v: %v", i, err)
		}

		_, _, err = expectRegexTimeoutWithOutput(child, tt.expectRegEx, time.Minute)
		if err != nil {
			t.Fatalf("Expected %q but not found #%v: %v", tt.expectRegEx, i, err)
		}

		err = child.Wait()
		if err != nil {
			t.Fatalf("rkt didn't terminate correctly: %v", err)
		}
	}
}
