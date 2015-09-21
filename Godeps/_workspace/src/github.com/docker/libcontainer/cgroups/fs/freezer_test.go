package fs

import (
	"testing"

	"github.com/docker/libcontainer/configs"
)

func TestFreezerSetState(t *testing.T) {
	helper := NewCgroupTestUtil("freezer", t)
	defer helper.cleanup()

	helper.writeFileContents(map[string]string{
		"freezer.state": string(configs.Frozen),
	})

	helper.CgroupData.c.Freezer = configs.Thawed
	freezer := &FreezerGroup{}
	if err := freezer.Set(helper.CgroupPath, helper.CgroupData.c); err != nil {
		t.Fatal(err)
	}

	value, err := getCgroupParamString(helper.CgroupPath, "freezer.state")
	if err != nil {
		t.Fatalf("Failed to parse freezer.state - %s", err)
	}
	if value != string(configs.Thawed) {
		t.Fatal("Got the wrong value, set freezer.state failed.")
	}
}

func TestFreezerSetInvalidState(t *testing.T) {
	helper := NewCgroupTestUtil("freezer", t)
	defer helper.cleanup()

	const (
		invalidArg configs.FreezerState = "Invalid"
	)

	helper.CgroupData.c.Freezer = invalidArg
	freezer := &FreezerGroup{}
	if err := freezer.Set(helper.CgroupPath, helper.CgroupData.c); err == nil {
		t.Fatal("Failed to return invalid argument error")
	}
}
