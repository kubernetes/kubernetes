// Copyright 2014,2015 Red Hat, Inc
// Copyright 2014,2015 Docker, Inc
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

// +build linux

package selinux_test

import (
	"os"
	"testing"

	"github.com/coreos/rkt/pkg/selinux"
)

func testSetfilecon(t *testing.T) {
	if selinux.SelinuxEnabled() {
		tmp := "selinux_test"
		out, _ := os.OpenFile(tmp, os.O_WRONLY, 0)
		out.Close()
		err := selinux.Setfilecon(tmp, "system_u:object_r:bin_t:s0")
		if err != nil {
			t.Log("Setfilecon failed")
			t.Fatal(err)
		}
		os.Remove(tmp)
	}
}

func TestSELinux(t *testing.T) {
	var (
		err            error
		plabel, flabel string
	)

	if selinux.SelinuxEnabled() {
		t.Log("Enabled")
		plabel, flabel = selinux.GetLxcContexts()
		if plabel == "" {
			t.Log("No lxc contexts, skipping tests")
			return
		}
		t.Log(plabel)
		t.Log(flabel)
		selinux.FreeLxcContexts(plabel)
		plabel, flabel = selinux.GetLxcContexts()
		t.Log(plabel)
		t.Log(flabel)
		selinux.FreeLxcContexts(plabel)
		t.Log("getenforce ", selinux.SelinuxGetEnforce())
		t.Log("getenforcemode ", selinux.SelinuxGetEnforceMode())
		pid := os.Getpid()
		t.Logf("PID:%d MCS:%s\n", pid, selinux.IntToMcs(pid, 1023))
		err = selinux.Setfscreatecon("unconfined_u:unconfined_r:unconfined_t:s0")
		if err == nil {
			t.Log(selinux.Getfscreatecon())
		} else {
			t.Log("setfscreatecon failed", err)
			t.Fatal(err)
		}
		err = selinux.Setfscreatecon("")
		if err == nil {
			t.Log(selinux.Getfscreatecon())
		} else {
			t.Log("setfscreatecon failed", err)
			t.Fatal(err)
		}
		t.Log(selinux.Getpidcon(1))
	} else {
		t.Log("Disabled")
	}
}
