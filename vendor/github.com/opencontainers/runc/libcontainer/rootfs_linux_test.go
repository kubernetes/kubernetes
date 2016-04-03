// +build linux

package libcontainer

import "testing"

func TestCheckMountDestOnProc(t *testing.T) {
	dest := "/rootfs/proc/"
	err := checkMountDestination("/rootfs", dest)
	if err == nil {
		t.Fatal("destination inside proc should return an error")
	}
}

func TestCheckMountDestInSys(t *testing.T) {
	dest := "/rootfs//sys/fs/cgroup"
	err := checkMountDestination("/rootfs", dest)
	if err != nil {
		t.Fatal("destination inside /sys should not return an error")
	}
}

func TestCheckMountDestFalsePositive(t *testing.T) {
	dest := "/rootfs/sysfiles/fs/cgroup"
	err := checkMountDestination("/rootfs", dest)
	if err != nil {
		t.Fatal(err)
	}
}

func TestCheckMountRoot(t *testing.T) {
	dest := "/rootfs"
	err := checkMountDestination("/rootfs", dest)
	if err == nil {
		t.Fatal(err)
	}
}
