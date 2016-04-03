// +build linux

package lxc

import (
	"testing"
)

func TestParseRunningInfo(t *testing.T) {
	raw := `
    state: RUNNING
    pid:    50`

	info, err := parseLxcInfo(raw)
	if err != nil {
		t.Fatal(err)
	}
	if !info.Running {
		t.Fatal("info should return a running state")
	}
	if info.Pid != 50 {
		t.Fatalf("info should have pid 50 got %d", info.Pid)
	}
}

func TestEmptyInfo(t *testing.T) {
	_, err := parseLxcInfo("")
	if err == nil {
		t.Fatal("error should not be nil")
	}
}

func TestBadInfo(t *testing.T) {
	_, err := parseLxcInfo("state")
	if err != nil {
		t.Fatal(err)
	}
}
