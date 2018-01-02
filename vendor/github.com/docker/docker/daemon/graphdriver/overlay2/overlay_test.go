// +build linux

package overlay2

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/daemon/graphdriver/graphtest"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/reexec"
	"golang.org/x/sys/unix"
)

func init() {
	// Do not sure chroot to speed run time and allow archive
	// errors or hangs to be debugged directly from the test process.
	untar = archive.UntarUncompressed
	graphdriver.ApplyUncompressedLayer = archive.ApplyUncompressedLayer

	reexec.Init()
}

func cdMountFrom(dir, device, target, mType, label string) error {
	wd, err := os.Getwd()
	if err != nil {
		return err
	}
	os.Chdir(dir)
	defer os.Chdir(wd)

	return unix.Mount(device, target, mType, 0, label)
}

func skipIfNaive(t *testing.T) {
	td, err := ioutil.TempDir("", "naive-check-")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(td)

	if useNaiveDiff(td) {
		t.Skipf("Cannot run test with naive diff")
	}
}

// This avoids creating a new driver for each test if all tests are run
// Make sure to put new tests between TestOverlaySetup and TestOverlayTeardown
func TestOverlaySetup(t *testing.T) {
	graphtest.GetDriver(t, driverName)
}

func TestOverlayCreateEmpty(t *testing.T) {
	graphtest.DriverTestCreateEmpty(t, driverName)
}

func TestOverlayCreateBase(t *testing.T) {
	graphtest.DriverTestCreateBase(t, driverName)
}

func TestOverlayCreateSnap(t *testing.T) {
	graphtest.DriverTestCreateSnap(t, driverName)
}

func TestOverlay128LayerRead(t *testing.T) {
	graphtest.DriverTestDeepLayerRead(t, 128, driverName)
}

func TestOverlayDiffApply10Files(t *testing.T) {
	skipIfNaive(t)
	graphtest.DriverTestDiffApply(t, 10, driverName)
}

func TestOverlayChanges(t *testing.T) {
	skipIfNaive(t)
	graphtest.DriverTestChanges(t, driverName)
}

func TestOverlayTeardown(t *testing.T) {
	graphtest.PutDriver(t)
}

// Benchmarks should always setup new driver

func BenchmarkExists(b *testing.B) {
	graphtest.DriverBenchExists(b, driverName)
}

func BenchmarkGetEmpty(b *testing.B) {
	graphtest.DriverBenchGetEmpty(b, driverName)
}

func BenchmarkDiffBase(b *testing.B) {
	graphtest.DriverBenchDiffBase(b, driverName)
}

func BenchmarkDiffSmallUpper(b *testing.B) {
	graphtest.DriverBenchDiffN(b, 10, 10, driverName)
}

func BenchmarkDiff10KFileUpper(b *testing.B) {
	graphtest.DriverBenchDiffN(b, 10, 10000, driverName)
}

func BenchmarkDiff10KFilesBottom(b *testing.B) {
	graphtest.DriverBenchDiffN(b, 10000, 10, driverName)
}

func BenchmarkDiffApply100(b *testing.B) {
	graphtest.DriverBenchDiffApplyN(b, 100, driverName)
}

func BenchmarkDiff20Layers(b *testing.B) {
	graphtest.DriverBenchDeepLayerDiff(b, 20, driverName)
}

func BenchmarkRead20Layers(b *testing.B) {
	graphtest.DriverBenchDeepLayerRead(b, 20, driverName)
}
