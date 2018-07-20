// +build linux freebsd

package graphtest

import (
	"bytes"
	"io/ioutil"
	"math/rand"
	"os"
	"path"
	"reflect"
	"testing"
	"unsafe"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/daemon/graphdriver/quota"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/go-units"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/sys/unix"
)

var (
	drv *Driver
)

// Driver conforms to graphdriver.Driver interface and
// contains information such as root and reference count of the number of clients using it.
// This helps in testing drivers added into the framework.
type Driver struct {
	graphdriver.Driver
	root     string
	refCount int
}

func newDriver(t testing.TB, name string, options []string) *Driver {
	root, err := ioutil.TempDir("", "docker-graphtest-")
	require.NoError(t, err)

	require.NoError(t, os.MkdirAll(root, 0755))
	d, err := graphdriver.GetDriver(name, nil, graphdriver.Options{DriverOptions: options, Root: root})
	if err != nil {
		t.Logf("graphdriver: %v\n", err)
		if graphdriver.IsDriverNotSupported(err) {
			t.Skipf("Driver %s not supported", name)
		}
		t.Fatal(err)
	}
	return &Driver{d, root, 1}
}

func cleanup(t testing.TB, d *Driver) {
	if err := drv.Cleanup(); err != nil {
		t.Fatal(err)
	}
	os.RemoveAll(d.root)
}

// GetDriver create a new driver with given name or return an existing driver with the name updating the reference count.
func GetDriver(t testing.TB, name string, options ...string) graphdriver.Driver {
	if drv == nil {
		drv = newDriver(t, name, options)
	} else {
		drv.refCount++
	}
	return drv
}

// PutDriver removes the driver if it is no longer used and updates the reference count.
func PutDriver(t testing.TB) {
	if drv == nil {
		t.Skip("No driver to put!")
	}
	drv.refCount--
	if drv.refCount == 0 {
		cleanup(t, drv)
		drv = nil
	}
}

// DriverTestCreateEmpty creates a new image and verifies it is empty and the right metadata
func DriverTestCreateEmpty(t testing.TB, drivername string, driverOptions ...string) {
	driver := GetDriver(t, drivername, driverOptions...)
	defer PutDriver(t)

	err := driver.Create("empty", "", nil)
	require.NoError(t, err)

	defer func() {
		require.NoError(t, driver.Remove("empty"))
	}()

	if !driver.Exists("empty") {
		t.Fatal("Newly created image doesn't exist")
	}

	dir, err := driver.Get("empty", "")
	require.NoError(t, err)

	verifyFile(t, dir.Path(), 0755|os.ModeDir, 0, 0)

	// Verify that the directory is empty
	fis, err := readDir(dir, dir.Path())
	require.NoError(t, err)
	assert.Len(t, fis, 0)

	driver.Put("empty")
}

// DriverTestCreateBase create a base driver and verify.
func DriverTestCreateBase(t testing.TB, drivername string, driverOptions ...string) {
	driver := GetDriver(t, drivername, driverOptions...)
	defer PutDriver(t)

	createBase(t, driver, "Base")
	defer func() {
		require.NoError(t, driver.Remove("Base"))
	}()
	verifyBase(t, driver, "Base")
}

// DriverTestCreateSnap Create a driver and snap and verify.
func DriverTestCreateSnap(t testing.TB, drivername string, driverOptions ...string) {
	driver := GetDriver(t, drivername, driverOptions...)
	defer PutDriver(t)

	createBase(t, driver, "Base")
	defer func() {
		require.NoError(t, driver.Remove("Base"))
	}()

	err := driver.Create("Snap", "Base", nil)
	require.NoError(t, err)
	defer func() {
		require.NoError(t, driver.Remove("Snap"))
	}()

	verifyBase(t, driver, "Snap")
}

// DriverTestDeepLayerRead reads a file from a lower layer under a given number of layers
func DriverTestDeepLayerRead(t testing.TB, layerCount int, drivername string, driverOptions ...string) {
	driver := GetDriver(t, drivername, driverOptions...)
	defer PutDriver(t)

	base := stringid.GenerateRandomID()
	if err := driver.Create(base, "", nil); err != nil {
		t.Fatal(err)
	}

	content := []byte("test content")
	if err := addFile(driver, base, "testfile.txt", content); err != nil {
		t.Fatal(err)
	}

	topLayer, err := addManyLayers(driver, base, layerCount)
	if err != nil {
		t.Fatal(err)
	}

	err = checkManyLayers(driver, topLayer, layerCount)
	if err != nil {
		t.Fatal(err)
	}

	if err := checkFile(driver, topLayer, "testfile.txt", content); err != nil {
		t.Fatal(err)
	}
}

// DriverTestDiffApply tests diffing and applying produces the same layer
func DriverTestDiffApply(t testing.TB, fileCount int, drivername string, driverOptions ...string) {
	driver := GetDriver(t, drivername, driverOptions...)
	defer PutDriver(t)
	base := stringid.GenerateRandomID()
	upper := stringid.GenerateRandomID()
	deleteFile := "file-remove.txt"
	deleteFileContent := []byte("This file should get removed in upper!")
	deleteDir := "var/lib"

	if err := driver.Create(base, "", nil); err != nil {
		t.Fatal(err)
	}

	if err := addManyFiles(driver, base, fileCount, 3); err != nil {
		t.Fatal(err)
	}

	if err := addFile(driver, base, deleteFile, deleteFileContent); err != nil {
		t.Fatal(err)
	}

	if err := addDirectory(driver, base, deleteDir); err != nil {
		t.Fatal(err)
	}

	if err := driver.Create(upper, base, nil); err != nil {
		t.Fatal(err)
	}

	if err := addManyFiles(driver, upper, fileCount, 6); err != nil {
		t.Fatal(err)
	}

	if err := removeAll(driver, upper, deleteFile, deleteDir); err != nil {
		t.Fatal(err)
	}

	diffSize, err := driver.DiffSize(upper, "")
	if err != nil {
		t.Fatal(err)
	}

	diff := stringid.GenerateRandomID()
	if err := driver.Create(diff, base, nil); err != nil {
		t.Fatal(err)
	}

	if err := checkManyFiles(driver, diff, fileCount, 3); err != nil {
		t.Fatal(err)
	}

	if err := checkFile(driver, diff, deleteFile, deleteFileContent); err != nil {
		t.Fatal(err)
	}

	arch, err := driver.Diff(upper, base)
	if err != nil {
		t.Fatal(err)
	}

	buf := bytes.NewBuffer(nil)
	if _, err := buf.ReadFrom(arch); err != nil {
		t.Fatal(err)
	}
	if err := arch.Close(); err != nil {
		t.Fatal(err)
	}

	applyDiffSize, err := driver.ApplyDiff(diff, base, bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	if applyDiffSize != diffSize {
		t.Fatalf("Apply diff size different, got %d, expected %d", applyDiffSize, diffSize)
	}

	if err := checkManyFiles(driver, diff, fileCount, 6); err != nil {
		t.Fatal(err)
	}

	if err := checkFileRemoved(driver, diff, deleteFile); err != nil {
		t.Fatal(err)
	}

	if err := checkFileRemoved(driver, diff, deleteDir); err != nil {
		t.Fatal(err)
	}
}

// DriverTestChanges tests computed changes on a layer matches changes made
func DriverTestChanges(t testing.TB, drivername string, driverOptions ...string) {
	driver := GetDriver(t, drivername, driverOptions...)
	defer PutDriver(t)
	base := stringid.GenerateRandomID()
	upper := stringid.GenerateRandomID()
	if err := driver.Create(base, "", nil); err != nil {
		t.Fatal(err)
	}

	if err := addManyFiles(driver, base, 20, 3); err != nil {
		t.Fatal(err)
	}

	if err := driver.Create(upper, base, nil); err != nil {
		t.Fatal(err)
	}

	expectedChanges, err := changeManyFiles(driver, upper, 20, 6)
	if err != nil {
		t.Fatal(err)
	}

	changes, err := driver.Changes(upper, base)
	if err != nil {
		t.Fatal(err)
	}

	if err = checkChanges(expectedChanges, changes); err != nil {
		t.Fatal(err)
	}
}

func writeRandomFile(path string, size uint64) error {
	buf := make([]int64, size/8)

	r := rand.NewSource(0)
	for i := range buf {
		buf[i] = r.Int63()
	}

	// Cast to []byte
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&buf))
	header.Len *= 8
	header.Cap *= 8
	data := *(*[]byte)(unsafe.Pointer(&header))

	return ioutil.WriteFile(path, data, 0700)
}

// DriverTestSetQuota Create a driver and test setting quota.
func DriverTestSetQuota(t *testing.T, drivername string, required bool) {
	driver := GetDriver(t, drivername)
	defer PutDriver(t)

	createBase(t, driver, "Base")
	createOpts := &graphdriver.CreateOpts{}
	createOpts.StorageOpt = make(map[string]string, 1)
	createOpts.StorageOpt["size"] = "50M"
	layerName := drivername + "Test"
	if err := driver.CreateReadWrite(layerName, "Base", createOpts); err == quota.ErrQuotaNotSupported && !required {
		t.Skipf("Quota not supported on underlying filesystem: %v", err)
	} else if err != nil {
		t.Fatal(err)
	}

	mountPath, err := driver.Get(layerName, "")
	if err != nil {
		t.Fatal(err)
	}

	quota := uint64(50 * units.MiB)

	// Try to write a file smaller than quota, and ensure it works
	err = writeRandomFile(path.Join(mountPath.Path(), "smallfile"), quota/2)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(path.Join(mountPath.Path(), "smallfile"))

	// Try to write a file bigger than quota. We've already filled up half the quota, so hitting the limit should be easy
	err = writeRandomFile(path.Join(mountPath.Path(), "bigfile"), quota)
	if err == nil {
		t.Fatalf("expected write to fail(), instead had success")
	}
	if pathError, ok := err.(*os.PathError); ok && pathError.Err != unix.EDQUOT && pathError.Err != unix.ENOSPC {
		os.Remove(path.Join(mountPath.Path(), "bigfile"))
		t.Fatalf("expect write() to fail with %v or %v, got %v", unix.EDQUOT, unix.ENOSPC, pathError.Err)
	}
}
