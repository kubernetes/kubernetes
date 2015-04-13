package fs

import (
	"testing"
)

func TestCpusetSetCpus(t *testing.T) {
	helper := NewCgroupTestUtil("cpuset", t)
	defer helper.cleanup()

	const (
		cpusBefore = "0"
		cpusAfter  = "1-3"
	)

	helper.writeFileContents(map[string]string{
		"cpuset.cpus": cpusBefore,
	})

	helper.CgroupData.c.CpusetCpus = cpusAfter
	cpuset := &CpusetGroup{}
	if err := cpuset.Set(helper.CgroupPath, helper.CgroupData.c); err != nil {
		t.Fatal(err)
	}

	value, err := getCgroupParamString(helper.CgroupPath, "cpuset.cpus")
	if err != nil {
		t.Fatalf("Failed to parse cpuset.cpus - %s", err)
	}

	if value != cpusAfter {
		t.Fatal("Got the wrong value, set cpuset.cpus failed.")
	}
}

func TestCpusetSetMems(t *testing.T) {
	helper := NewCgroupTestUtil("cpuset", t)
	defer helper.cleanup()

	const (
		memsBefore = "0"
		memsAfter  = "1"
	)

	helper.writeFileContents(map[string]string{
		"cpuset.mems": memsBefore,
	})

	helper.CgroupData.c.CpusetMems = memsAfter
	cpuset := &CpusetGroup{}
	if err := cpuset.Set(helper.CgroupPath, helper.CgroupData.c); err != nil {
		t.Fatal(err)
	}

	value, err := getCgroupParamString(helper.CgroupPath, "cpuset.mems")
	if err != nil {
		t.Fatalf("Failed to parse cpuset.mems - %s", err)
	}

	if value != memsAfter {
		t.Fatal("Got the wrong value, set cpuset.mems failed.")
	}
}
