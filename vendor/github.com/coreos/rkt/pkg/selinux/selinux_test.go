// Copyright 2016 The rkt Authors
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
	"fmt"
	"io/ioutil"
	"os"
	"testing"

	"github.com/coreos/rkt/pkg/selinux"
)

func TestSetfilecon(t *testing.T) {
	if !selinux.SelinuxEnabled() {
		t.Skip("SELinux not enabled")
	}

	f, err := ioutil.TempFile("", "rkt-selinux-test")
	if err != nil {
		panic(fmt.Sprintf("unable to create tempfile: %v", err))
	}
	f.Close()
	defer os.Remove(f.Name())
	err = selinux.Setfilecon(f.Name(), "system_u:object_r:bin_t:s0")
	if err != nil {
		t.Log("Setfilecon failed")
		t.Fatal(err)
	}
}

func TestSELinux(t *testing.T) {
	if !selinux.SelinuxEnabled() {
		t.Skip("SELinux not enabled")
	}

	dir, err := ioutil.TempDir("", "rkt-selinux-test")
	if err != nil {
		panic(fmt.Sprintf("unable to create temp dir: %v", err))
	}
	defer os.RemoveAll(dir)
	if err := selinux.SetMCSDir(dir); err != nil {
		t.Errorf("error setting MCS directory: %v", err)
	}

	var plabel, flabel string

	plabel, flabel, err = selinux.GetLxcContexts()
	if err != nil {
		t.Fatal(err)
	}
	if plabel == "" {
		t.Skip("No LXC contexts, skipping tests")
	}
	t.Log(plabel)
	t.Log(flabel)
	selinux.FreeLxcContexts(plabel)
	plabel, flabel, err = selinux.GetLxcContexts()
	if err != nil {
		t.Fatal(err)
	}
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
}
