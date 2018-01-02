// +build linux

package zfs

import (
	"testing"

	"github.com/docker/docker/daemon/graphdriver/graphtest"
)

// This avoids creating a new driver for each test if all tests are run
// Make sure to put new tests between TestZfsSetup and TestZfsTeardown
func TestZfsSetup(t *testing.T) {
	graphtest.GetDriver(t, "zfs")
}

func TestZfsCreateEmpty(t *testing.T) {
	graphtest.DriverTestCreateEmpty(t, "zfs")
}

func TestZfsCreateBase(t *testing.T) {
	graphtest.DriverTestCreateBase(t, "zfs")
}

func TestZfsCreateSnap(t *testing.T) {
	graphtest.DriverTestCreateSnap(t, "zfs")
}

func TestZfsSetQuota(t *testing.T) {
	graphtest.DriverTestSetQuota(t, "zfs")
}

func TestZfsTeardown(t *testing.T) {
	graphtest.PutDriver(t)
}
