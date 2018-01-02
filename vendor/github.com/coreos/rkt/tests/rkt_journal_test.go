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
	"os"
	"path/filepath"
	"strings"
	"testing"

	sd_util "github.com/coreos/go-systemd/util"
	"github.com/coreos/rkt/tests/testutils"
)

const journalDir = "/var/log/journal"

func TestJournalLink(t *testing.T) {
	if !sd_util.IsRunningSystemd() {
		t.Skip("Systemd is not running on the host.")
	}

	if _, err := os.Stat(journalDir); os.IsNotExist(err) {
		t.Skip("Persistent journaling disabled.")
	}

	image := getInspectImagePath()

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	rktCmd := fmt.Sprintf("%s prepare --insecure-options=image %s", ctx.Cmd(), image)
	uuid := runRktAndGetUUID(t, rktCmd)

	rktCmd = fmt.Sprintf("%s run-prepared %s", ctx.Cmd(), uuid)
	spawnAndWaitOrFail(t, rktCmd, 0)

	machineID := strings.Replace(uuid, "-", "", -1)
	journalPath := filepath.Join("/var/log/journal", machineID)

	link, err := os.Readlink(journalPath)
	if err != nil {
		t.Fatalf("failed to read journal link %q", journalPath)
	}

	podJournal := filepath.Join(ctx.DataDir(), "pods/run", uuid, "stage1/rootfs/var/log/journal/", machineID)
	if link != podJournal {
		t.Fatalf("unexpected target of journal link: %q. Expected %q", link, podJournal)
	}
}
