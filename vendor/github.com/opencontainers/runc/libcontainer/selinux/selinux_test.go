// +build linux,selinux

package selinux_test

import (
	"os"
	"testing"

	"github.com/opencontainers/runc/libcontainer/selinux"
)

func TestSetfilecon(t *testing.T) {
	if selinux.SelinuxEnabled() {
		tmp := "selinux_test"
		out, _ := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE, 0)
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
		t.Log(plabel)
		t.Log(flabel)
		selinux.FreeLxcContexts(plabel)
		plabel, flabel = selinux.GetLxcContexts()
		t.Log(plabel)
		t.Log(flabel)
		selinux.FreeLxcContexts(plabel)
		t.Log("getenforce ", selinux.SelinuxGetEnforce())
		mode := selinux.SelinuxGetEnforceMode()
		t.Log("getenforcemode ", mode)

		defer selinux.SelinuxSetEnforce(mode)
		if err := selinux.SelinuxSetEnforce(selinux.Enforcing); err != nil {
			t.Fatalf("enforcing selinux failed: %v", err)
		}
		if err := selinux.SelinuxSetEnforce(selinux.Permissive); err != nil {
			t.Fatalf("setting selinux mode to permissive failed: %v", err)
		}
		selinux.SelinuxSetEnforce(mode)

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
