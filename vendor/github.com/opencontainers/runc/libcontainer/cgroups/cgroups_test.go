// +build linux

package cgroups

import (
	"testing"
)

func TestParseCgroups(t *testing.T) {
	cgroups, err := ParseCgroupFile("/proc/self/cgroup")
	if err != nil {
		t.Fatal(err)
	}

	if _, ok := cgroups["cpu"]; !ok {
		t.Fail()
	}
}
