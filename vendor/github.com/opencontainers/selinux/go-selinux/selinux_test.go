// +build linux,selinux

package selinux

import (
	"os"
	"testing"
)

func TestSetFileLabel(t *testing.T) {
	if GetEnabled() {
		tmp := "selinux_test"
		con := "system_u:object_r:bin_t:s0"
		out, _ := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE, 0)
		out.Close()
		err := SetFileLabel(tmp, con)
		if err != nil {
			t.Log("Setfilecon failed")
			t.Fatal(err)
		}
		filelabel, err := FileLabel(tmp)
		if err != nil {
			t.Log("FileLabel failed")
			t.Fatal(err)
		}
		if con != filelabel {
			t.Fatal("FileLabel failed, returned %s expected %s", filelabel, con)
		}

		os.Remove(tmp)
	}
}

func TestSELinux(t *testing.T) {
	var (
		err            error
		plabel, flabel string
	)

	if GetEnabled() {
		t.Log("Enabled")
		plabel, flabel = ContainerLabels()
		t.Log(plabel)
		t.Log(flabel)
		ReleaseLabel(plabel)
		plabel, flabel = ContainerLabels()
		t.Log(plabel)
		t.Log(flabel)
		ReleaseLabel(plabel)
		t.Log("Enforcing Mode", EnforceMode())
		mode := DefaultEnforceMode()
		t.Log("Default Enforce Mode ", mode)

		defer SetEnforceMode(mode)
		if err := SetEnforceMode(Enforcing); err != nil {
			t.Fatalf("enforcing selinux failed: %v", err)
		}
		if err := SetEnforceMode(Permissive); err != nil {
			t.Fatalf("setting selinux mode to permissive failed: %v", err)
		}
		SetEnforceMode(mode)

		pid := os.Getpid()
		t.Logf("PID:%d MCS:%s\n", pid, intToMcs(pid, 1023))
		err = SetFSCreateLabel("unconfined_u:unconfined_r:unconfined_t:s0")
		if err == nil {
			t.Log(FSCreateLabel())
		} else {
			t.Log("SetFSCreateLabel failed", err)
			t.Fatal(err)
		}
		err = SetFSCreateLabel("")
		if err == nil {
			t.Log(FSCreateLabel())
		} else {
			t.Log("SetFSCreateLabel failed", err)
			t.Fatal(err)
		}
		t.Log(PidLabel(1))
	}
}
