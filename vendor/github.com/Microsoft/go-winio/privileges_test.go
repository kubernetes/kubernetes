package winio

import "testing"

func TestRunWithUnavailablePrivilege(t *testing.T) {
	err := RunWithPrivilege("SeCreateTokenPrivilege", func() error { return nil })
	if _, ok := err.(*PrivilegeError); err == nil || !ok {
		t.Fatal("expected PrivilegeError")
	}
}

func TestRunWithPrivileges(t *testing.T) {
	err := RunWithPrivilege("SeShutdownPrivilege", func() error { return nil })
	if err != nil {
		t.Fatal(err)
	}
}
