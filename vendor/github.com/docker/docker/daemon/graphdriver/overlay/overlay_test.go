// +build linux

package overlay

import (
	"github.com/docker/docker/daemon/graphdriver/graphtest"
	"testing"
)

// This avoids creating a new driver for each test if all tests are run
// Make sure to put new tests between TestOverlaySetup and TestOverlayTeardown
func TestOverlaySetup(t *testing.T) {
	graphtest.GetDriver(t, "overlay")
}

func TestOverlayCreateEmpty(t *testing.T) {
	graphtest.DriverTestCreateEmpty(t, "overlay")
}

func TestOverlayCreateBase(t *testing.T) {
	graphtest.DriverTestCreateBase(t, "overlay")
}

func TestOverlayCreateSnap(t *testing.T) {
	graphtest.DriverTestCreateSnap(t, "overlay")
}

func TestOverlayTeardown(t *testing.T) {
	graphtest.PutDriver(t)
}
