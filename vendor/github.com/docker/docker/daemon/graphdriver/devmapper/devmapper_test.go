// +build linux

package devmapper

import (
	"fmt"
	"os"
	"syscall"
	"testing"
	"time"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/daemon/graphdriver/graphtest"
)

func init() {
	// Reduce the size of the base fs and loopback for the tests
	defaultDataLoopbackSize = 300 * 1024 * 1024
	defaultMetaDataLoopbackSize = 200 * 1024 * 1024
	defaultBaseFsSize = 300 * 1024 * 1024
	defaultUdevSyncOverride = true
	if err := initLoopbacks(); err != nil {
		panic(err)
	}
}

// initLoopbacks ensures that the loopback devices are properly created within
// the system running the device mapper tests.
func initLoopbacks() error {
	statT, err := getBaseLoopStats()
	if err != nil {
		return err
	}
	// create at least 8 loopback files, ya, that is a good number
	for i := 0; i < 8; i++ {
		loopPath := fmt.Sprintf("/dev/loop%d", i)
		// only create new loopback files if they don't exist
		if _, err := os.Stat(loopPath); err != nil {
			if mkerr := syscall.Mknod(loopPath,
				uint32(statT.Mode|syscall.S_IFBLK), int((7<<8)|(i&0xff)|((i&0xfff00)<<12))); mkerr != nil {
				return mkerr
			}
			os.Chown(loopPath, int(statT.Uid), int(statT.Gid))
		}
	}
	return nil
}

// getBaseLoopStats inspects /dev/loop0 to collect uid,gid, and mode for the
// loop0 device on the system.  If it does not exist we assume 0,0,0660 for the
// stat data
func getBaseLoopStats() (*syscall.Stat_t, error) {
	loop0, err := os.Stat("/dev/loop0")
	if err != nil {
		if os.IsNotExist(err) {
			return &syscall.Stat_t{
				Uid:  0,
				Gid:  0,
				Mode: 0660,
			}, nil
		}
		return nil, err
	}
	return loop0.Sys().(*syscall.Stat_t), nil
}

// This avoids creating a new driver for each test if all tests are run
// Make sure to put new tests between TestDevmapperSetup and TestDevmapperTeardown
func TestDevmapperSetup(t *testing.T) {
	graphtest.GetDriver(t, "devicemapper")
}

func TestDevmapperCreateEmpty(t *testing.T) {
	graphtest.DriverTestCreateEmpty(t, "devicemapper")
}

func TestDevmapperCreateBase(t *testing.T) {
	graphtest.DriverTestCreateBase(t, "devicemapper")
}

func TestDevmapperCreateSnap(t *testing.T) {
	graphtest.DriverTestCreateSnap(t, "devicemapper")
}

func TestDevmapperTeardown(t *testing.T) {
	graphtest.PutDriver(t)
}

func TestDevmapperReduceLoopBackSize(t *testing.T) {
	tenMB := int64(10 * 1024 * 1024)
	testChangeLoopBackSize(t, -tenMB, defaultDataLoopbackSize, defaultMetaDataLoopbackSize)
}

func TestDevmapperIncreaseLoopBackSize(t *testing.T) {
	tenMB := int64(10 * 1024 * 1024)
	testChangeLoopBackSize(t, tenMB, defaultDataLoopbackSize+tenMB, defaultMetaDataLoopbackSize+tenMB)
}

func testChangeLoopBackSize(t *testing.T, delta, expectDataSize, expectMetaDataSize int64) {
	driver := graphtest.GetDriver(t, "devicemapper").(*graphtest.Driver).Driver.(*graphdriver.NaiveDiffDriver).ProtoDriver.(*Driver)
	defer graphtest.PutDriver(t)
	// make sure data or metadata loopback size are the default size
	if s := driver.DeviceSet.Status(); s.Data.Total != uint64(defaultDataLoopbackSize) || s.Metadata.Total != uint64(defaultMetaDataLoopbackSize) {
		t.Fatal("data or metadata loop back size is incorrect")
	}
	if err := driver.Cleanup(); err != nil {
		t.Fatal(err)
	}
	//Reload
	d, err := Init(driver.home, []string{
		fmt.Sprintf("dm.loopdatasize=%d", defaultDataLoopbackSize+delta),
		fmt.Sprintf("dm.loopmetadatasize=%d", defaultMetaDataLoopbackSize+delta),
	}, nil, nil)
	if err != nil {
		t.Fatalf("error creating devicemapper driver: %v", err)
	}
	driver = d.(*graphdriver.NaiveDiffDriver).ProtoDriver.(*Driver)
	if s := driver.DeviceSet.Status(); s.Data.Total != uint64(expectDataSize) || s.Metadata.Total != uint64(expectMetaDataSize) {
		t.Fatal("data or metadata loop back size is incorrect")
	}
	if err := driver.Cleanup(); err != nil {
		t.Fatal(err)
	}
}

// Make sure devices.Lock() has been release upon return from cleanupDeletedDevices() function
func TestDevmapperLockReleasedDeviceDeletion(t *testing.T) {
	driver := graphtest.GetDriver(t, "devicemapper").(*graphtest.Driver).Driver.(*graphdriver.NaiveDiffDriver).ProtoDriver.(*Driver)
	defer graphtest.PutDriver(t)

	// Call cleanupDeletedDevices() and after the call take and release
	// DeviceSet Lock. If lock has not been released, this will hang.
	driver.DeviceSet.cleanupDeletedDevices()

	doneChan := make(chan bool)

	go func() {
		driver.DeviceSet.Lock()
		defer driver.DeviceSet.Unlock()
		doneChan <- true
	}()

	select {
	case <-time.After(time.Second * 5):
		// Timer expired. That means lock was not released upon
		// function return and we are deadlocked. Release lock
		// here so that cleanup could succeed and fail the test.
		driver.DeviceSet.Unlock()
		t.Fatal("Could not acquire devices lock after call to cleanupDeletedDevices()")
	case <-doneChan:
	}
}
