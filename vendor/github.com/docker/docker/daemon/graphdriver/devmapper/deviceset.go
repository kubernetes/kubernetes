// +build linux

package devmapper

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/dockerversion"
	"github.com/docker/docker/pkg/devicemapper"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/loopback"
	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/pkg/parsers"
	units "github.com/docker/go-units"
	"github.com/opencontainers/selinux/go-selinux/label"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

var (
	defaultDataLoopbackSize      int64  = 100 * 1024 * 1024 * 1024
	defaultMetaDataLoopbackSize  int64  = 2 * 1024 * 1024 * 1024
	defaultBaseFsSize            uint64 = 10 * 1024 * 1024 * 1024
	defaultThinpBlockSize        uint32 = 128 // 64K = 128 512b sectors
	defaultUdevSyncOverride             = false
	maxDeviceID                         = 0xffffff // 24 bit, pool limit
	deviceIDMapSz                       = (maxDeviceID + 1) / 8
	driverDeferredRemovalSupport        = false
	enableDeferredRemoval               = false
	enableDeferredDeletion              = false
	userBaseSize                        = false
	defaultMinFreeSpacePercent   uint32 = 10
	lvmSetupConfigForce          bool
)

const deviceSetMetaFile string = "deviceset-metadata"
const transactionMetaFile string = "transaction-metadata"

type transaction struct {
	OpenTransactionID uint64 `json:"open_transaction_id"`
	DeviceIDHash      string `json:"device_hash"`
	DeviceID          int    `json:"device_id"`
}

type devInfo struct {
	Hash          string `json:"-"`
	DeviceID      int    `json:"device_id"`
	Size          uint64 `json:"size"`
	TransactionID uint64 `json:"transaction_id"`
	Initialized   bool   `json:"initialized"`
	Deleted       bool   `json:"deleted"`
	devices       *DeviceSet

	// The global DeviceSet lock guarantees that we serialize all
	// the calls to libdevmapper (which is not threadsafe), but we
	// sometimes release that lock while sleeping. In that case
	// this per-device lock is still held, protecting against
	// other accesses to the device that we're doing the wait on.
	//
	// WARNING: In order to avoid AB-BA deadlocks when releasing
	// the global lock while holding the per-device locks all
	// device locks must be acquired *before* the device lock, and
	// multiple device locks should be acquired parent before child.
	lock sync.Mutex
}

type metaData struct {
	Devices map[string]*devInfo `json:"Devices"`
}

// DeviceSet holds information about list of devices
type DeviceSet struct {
	metaData      `json:"-"`
	sync.Mutex    `json:"-"` // Protects all fields of DeviceSet and serializes calls into libdevmapper
	root          string
	devicePrefix  string
	TransactionID uint64 `json:"-"`
	NextDeviceID  int    `json:"next_device_id"`
	deviceIDMap   []byte

	// Options
	dataLoopbackSize      int64
	metaDataLoopbackSize  int64
	baseFsSize            uint64
	filesystem            string
	mountOptions          string
	mkfsArgs              []string
	dataDevice            string // block or loop dev
	dataLoopFile          string // loopback file, if used
	metadataDevice        string // block or loop dev
	metadataLoopFile      string // loopback file, if used
	doBlkDiscard          bool
	thinpBlockSize        uint32
	thinPoolDevice        string
	transaction           `json:"-"`
	overrideUdevSyncCheck bool
	deferredRemove        bool   // use deferred removal
	deferredDelete        bool   // use deferred deletion
	BaseDeviceUUID        string // save UUID of base device
	BaseDeviceFilesystem  string // save filesystem of base device
	nrDeletedDevices      uint   // number of deleted devices
	deletionWorkerTicker  *time.Ticker
	uidMaps               []idtools.IDMap
	gidMaps               []idtools.IDMap
	minFreeSpacePercent   uint32 //min free space percentage in thinpool
	xfsNospaceRetries     string // max retries when xfs receives ENOSPC
	lvmSetupConfig        directLVMConfig
}

// DiskUsage contains information about disk usage and is used when reporting Status of a device.
type DiskUsage struct {
	// Used bytes on the disk.
	Used uint64
	// Total bytes on the disk.
	Total uint64
	// Available bytes on the disk.
	Available uint64
}

// Status returns the information about the device.
type Status struct {
	// PoolName is the name of the data pool.
	PoolName string
	// DataFile is the actual block device for data.
	DataFile string
	// DataLoopback loopback file, if used.
	DataLoopback string
	// MetadataFile is the actual block device for metadata.
	MetadataFile string
	// MetadataLoopback is the loopback file, if used.
	MetadataLoopback string
	// Data is the disk used for data.
	Data DiskUsage
	// Metadata is the disk used for meta data.
	Metadata DiskUsage
	// BaseDeviceSize is base size of container and image
	BaseDeviceSize uint64
	// BaseDeviceFS is backing filesystem.
	BaseDeviceFS string
	// SectorSize size of the vector.
	SectorSize uint64
	// UdevSyncSupported is true if sync is supported.
	UdevSyncSupported bool
	// DeferredRemoveEnabled is true then the device is not unmounted.
	DeferredRemoveEnabled bool
	// True if deferred deletion is enabled. This is different from
	// deferred removal. "removal" means that device mapper device is
	// deactivated. Thin device is still in thin pool and can be activated
	// again. But "deletion" means that thin device will be deleted from
	// thin pool and it can't be activated again.
	DeferredDeleteEnabled      bool
	DeferredDeletedDeviceCount uint
	MinFreeSpace               uint64
}

// Structure used to export image/container metadata in docker inspect.
type deviceMetadata struct {
	deviceID   int
	deviceSize uint64 // size in bytes
	deviceName string // Device name as used during activation
}

// DevStatus returns information about device mounted containing its id, size and sector information.
type DevStatus struct {
	// DeviceID is the id of the device.
	DeviceID int
	// Size is the size of the filesystem.
	Size uint64
	// TransactionID is a unique integer per device set used to identify an operation on the file system, this number is incremental.
	TransactionID uint64
	// SizeInSectors indicates the size of the sectors allocated.
	SizeInSectors uint64
	// MappedSectors indicates number of mapped sectors.
	MappedSectors uint64
	// HighestMappedSector is the pointer to the highest mapped sector.
	HighestMappedSector uint64
}

func getDevName(name string) string {
	return "/dev/mapper/" + name
}

func (info *devInfo) Name() string {
	hash := info.Hash
	if hash == "" {
		hash = "base"
	}
	return fmt.Sprintf("%s-%s", info.devices.devicePrefix, hash)
}

func (info *devInfo) DevName() string {
	return getDevName(info.Name())
}

func (devices *DeviceSet) loopbackDir() string {
	return path.Join(devices.root, "devicemapper")
}

func (devices *DeviceSet) metadataDir() string {
	return path.Join(devices.root, "metadata")
}

func (devices *DeviceSet) metadataFile(info *devInfo) string {
	file := info.Hash
	if file == "" {
		file = "base"
	}
	return path.Join(devices.metadataDir(), file)
}

func (devices *DeviceSet) transactionMetaFile() string {
	return path.Join(devices.metadataDir(), transactionMetaFile)
}

func (devices *DeviceSet) deviceSetMetaFile() string {
	return path.Join(devices.metadataDir(), deviceSetMetaFile)
}

func (devices *DeviceSet) oldMetadataFile() string {
	return path.Join(devices.loopbackDir(), "json")
}

func (devices *DeviceSet) getPoolName() string {
	if devices.thinPoolDevice == "" {
		return devices.devicePrefix + "-pool"
	}
	return devices.thinPoolDevice
}

func (devices *DeviceSet) getPoolDevName() string {
	return getDevName(devices.getPoolName())
}

func (devices *DeviceSet) hasImage(name string) bool {
	dirname := devices.loopbackDir()
	filename := path.Join(dirname, name)

	_, err := os.Stat(filename)
	return err == nil
}

// ensureImage creates a sparse file of <size> bytes at the path
// <root>/devicemapper/<name>.
// If the file already exists and new size is larger than its current size, it grows to the new size.
// Either way it returns the full path.
func (devices *DeviceSet) ensureImage(name string, size int64) (string, error) {
	dirname := devices.loopbackDir()
	filename := path.Join(dirname, name)

	uid, gid, err := idtools.GetRootUIDGID(devices.uidMaps, devices.gidMaps)
	if err != nil {
		return "", err
	}
	if err := idtools.MkdirAllAs(dirname, 0700, uid, gid); err != nil && !os.IsExist(err) {
		return "", err
	}

	if fi, err := os.Stat(filename); err != nil {
		if !os.IsNotExist(err) {
			return "", err
		}
		logrus.Debugf("devmapper: Creating loopback file %s for device-manage use", filename)
		file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0600)
		if err != nil {
			return "", err
		}
		defer file.Close()

		if err := file.Truncate(size); err != nil {
			return "", err
		}
	} else {
		if fi.Size() < size {
			file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0600)
			if err != nil {
				return "", err
			}
			defer file.Close()
			if err := file.Truncate(size); err != nil {
				return "", fmt.Errorf("devmapper: Unable to grow loopback file %s: %v", filename, err)
			}
		} else if fi.Size() > size {
			logrus.Warnf("devmapper: Can't shrink loopback file %s", filename)
		}
	}
	return filename, nil
}

func (devices *DeviceSet) allocateTransactionID() uint64 {
	devices.OpenTransactionID = devices.TransactionID + 1
	return devices.OpenTransactionID
}

func (devices *DeviceSet) updatePoolTransactionID() error {
	if err := devicemapper.SetTransactionID(devices.getPoolDevName(), devices.TransactionID, devices.OpenTransactionID); err != nil {
		return fmt.Errorf("devmapper: Error setting devmapper transaction ID: %s", err)
	}
	devices.TransactionID = devices.OpenTransactionID
	return nil
}

func (devices *DeviceSet) removeMetadata(info *devInfo) error {
	if err := os.RemoveAll(devices.metadataFile(info)); err != nil {
		return fmt.Errorf("devmapper: Error removing metadata file %s: %s", devices.metadataFile(info), err)
	}
	return nil
}

// Given json data and file path, write it to disk
func (devices *DeviceSet) writeMetaFile(jsonData []byte, filePath string) error {
	tmpFile, err := ioutil.TempFile(devices.metadataDir(), ".tmp")
	if err != nil {
		return fmt.Errorf("devmapper: Error creating metadata file: %s", err)
	}

	n, err := tmpFile.Write(jsonData)
	if err != nil {
		return fmt.Errorf("devmapper: Error writing metadata to %s: %s", tmpFile.Name(), err)
	}
	if n < len(jsonData) {
		return io.ErrShortWrite
	}
	if err := tmpFile.Sync(); err != nil {
		return fmt.Errorf("devmapper: Error syncing metadata file %s: %s", tmpFile.Name(), err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("devmapper: Error closing metadata file %s: %s", tmpFile.Name(), err)
	}
	if err := os.Rename(tmpFile.Name(), filePath); err != nil {
		return fmt.Errorf("devmapper: Error committing metadata file %s: %s", tmpFile.Name(), err)
	}

	return nil
}

func (devices *DeviceSet) saveMetadata(info *devInfo) error {
	jsonData, err := json.Marshal(info)
	if err != nil {
		return fmt.Errorf("devmapper: Error encoding metadata to json: %s", err)
	}
	if err := devices.writeMetaFile(jsonData, devices.metadataFile(info)); err != nil {
		return err
	}
	return nil
}

func (devices *DeviceSet) markDeviceIDUsed(deviceID int) {
	var mask byte
	i := deviceID % 8
	mask = 1 << uint(i)
	devices.deviceIDMap[deviceID/8] = devices.deviceIDMap[deviceID/8] | mask
}

func (devices *DeviceSet) markDeviceIDFree(deviceID int) {
	var mask byte
	i := deviceID % 8
	mask = ^(1 << uint(i))
	devices.deviceIDMap[deviceID/8] = devices.deviceIDMap[deviceID/8] & mask
}

func (devices *DeviceSet) isDeviceIDFree(deviceID int) bool {
	var mask byte
	i := deviceID % 8
	mask = (1 << uint(i))
	return (devices.deviceIDMap[deviceID/8] & mask) == 0
}

// Should be called with devices.Lock() held.
func (devices *DeviceSet) lookupDevice(hash string) (*devInfo, error) {
	info := devices.Devices[hash]
	if info == nil {
		info = devices.loadMetadata(hash)
		if info == nil {
			return nil, fmt.Errorf("devmapper: Unknown device %s", hash)
		}

		devices.Devices[hash] = info
	}
	return info, nil
}

func (devices *DeviceSet) lookupDeviceWithLock(hash string) (*devInfo, error) {
	devices.Lock()
	defer devices.Unlock()
	info, err := devices.lookupDevice(hash)
	return info, err
}

// This function relies on that device hash map has been loaded in advance.
// Should be called with devices.Lock() held.
func (devices *DeviceSet) constructDeviceIDMap() {
	logrus.Debug("devmapper: constructDeviceIDMap()")
	defer logrus.Debug("devmapper: constructDeviceIDMap() END")

	for _, info := range devices.Devices {
		devices.markDeviceIDUsed(info.DeviceID)
		logrus.Debugf("devmapper: Added deviceId=%d to DeviceIdMap", info.DeviceID)
	}
}

func (devices *DeviceSet) deviceFileWalkFunction(path string, finfo os.FileInfo) error {

	// Skip some of the meta files which are not device files.
	if strings.HasSuffix(finfo.Name(), ".migrated") {
		logrus.Debugf("devmapper: Skipping file %s", path)
		return nil
	}

	if strings.HasPrefix(finfo.Name(), ".") {
		logrus.Debugf("devmapper: Skipping file %s", path)
		return nil
	}

	if finfo.Name() == deviceSetMetaFile {
		logrus.Debugf("devmapper: Skipping file %s", path)
		return nil
	}

	if finfo.Name() == transactionMetaFile {
		logrus.Debugf("devmapper: Skipping file %s", path)
		return nil
	}

	logrus.Debugf("devmapper: Loading data for file %s", path)

	hash := finfo.Name()
	if hash == "base" {
		hash = ""
	}

	// Include deleted devices also as cleanup delete device logic
	// will go through it and see if there are any deleted devices.
	if _, err := devices.lookupDevice(hash); err != nil {
		return fmt.Errorf("devmapper: Error looking up device %s:%v", hash, err)
	}

	return nil
}

func (devices *DeviceSet) loadDeviceFilesOnStart() error {
	logrus.Debug("devmapper: loadDeviceFilesOnStart()")
	defer logrus.Debug("devmapper: loadDeviceFilesOnStart() END")

	var scan = func(path string, info os.FileInfo, err error) error {
		if err != nil {
			logrus.Debugf("devmapper: Can't walk the file %s", path)
			return nil
		}

		// Skip any directories
		if info.IsDir() {
			return nil
		}

		return devices.deviceFileWalkFunction(path, info)
	}

	return filepath.Walk(devices.metadataDir(), scan)
}

// Should be called with devices.Lock() held.
func (devices *DeviceSet) unregisterDevice(hash string) error {
	logrus.Debugf("devmapper: unregisterDevice(%v)", hash)
	info := &devInfo{
		Hash: hash,
	}

	delete(devices.Devices, hash)

	if err := devices.removeMetadata(info); err != nil {
		logrus.Debugf("devmapper: Error removing metadata: %s", err)
		return err
	}

	return nil
}

// Should be called with devices.Lock() held.
func (devices *DeviceSet) registerDevice(id int, hash string, size uint64, transactionID uint64) (*devInfo, error) {
	logrus.Debugf("devmapper: registerDevice(%v, %v)", id, hash)
	info := &devInfo{
		Hash:          hash,
		DeviceID:      id,
		Size:          size,
		TransactionID: transactionID,
		Initialized:   false,
		devices:       devices,
	}

	devices.Devices[hash] = info

	if err := devices.saveMetadata(info); err != nil {
		// Try to remove unused device
		delete(devices.Devices, hash)
		return nil, err
	}

	return info, nil
}

func (devices *DeviceSet) activateDeviceIfNeeded(info *devInfo, ignoreDeleted bool) error {
	logrus.Debugf("devmapper: activateDeviceIfNeeded(%v)", info.Hash)

	if info.Deleted && !ignoreDeleted {
		return fmt.Errorf("devmapper: Can't activate device %v as it is marked for deletion", info.Hash)
	}

	// Make sure deferred removal on device is canceled, if one was
	// scheduled.
	if err := devices.cancelDeferredRemovalIfNeeded(info); err != nil {
		return fmt.Errorf("devmapper: Device Deferred Removal Cancellation Failed: %s", err)
	}

	if devinfo, _ := devicemapper.GetInfo(info.Name()); devinfo != nil && devinfo.Exists != 0 {
		return nil
	}

	return devicemapper.ActivateDevice(devices.getPoolDevName(), info.Name(), info.DeviceID, info.Size)
}

// Return true only if kernel supports xfs and mkfs.xfs is available
func xfsSupported() bool {
	// Make sure mkfs.xfs is available
	if _, err := exec.LookPath("mkfs.xfs"); err != nil {
		return false
	}

	// Check if kernel supports xfs filesystem or not.
	exec.Command("modprobe", "xfs").Run()

	f, err := os.Open("/proc/filesystems")
	if err != nil {
		logrus.Warnf("devmapper: Could not check if xfs is supported: %v", err)
		return false
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	for s.Scan() {
		if strings.HasSuffix(s.Text(), "\txfs") {
			return true
		}
	}

	if err := s.Err(); err != nil {
		logrus.Warnf("devmapper: Could not check if xfs is supported: %v", err)
	}
	return false
}

func determineDefaultFS() string {
	if xfsSupported() {
		return "xfs"
	}

	logrus.Warn("devmapper: XFS is not supported in your system. Either the kernel doesn't support it or mkfs.xfs is not in your PATH. Defaulting to ext4 filesystem")
	return "ext4"
}

func (devices *DeviceSet) createFilesystem(info *devInfo) (err error) {
	devname := info.DevName()

	args := []string{}
	args = append(args, devices.mkfsArgs...)

	args = append(args, devname)

	if devices.filesystem == "" {
		devices.filesystem = determineDefaultFS()
	}
	if err := devices.saveBaseDeviceFilesystem(devices.filesystem); err != nil {
		return err
	}

	logrus.Infof("devmapper: Creating filesystem %s on device %s", devices.filesystem, info.Name())
	defer func() {
		if err != nil {
			logrus.Infof("devmapper: Error while creating filesystem %s on device %s: %v", devices.filesystem, info.Name(), err)
		} else {
			logrus.Infof("devmapper: Successfully created filesystem %s on device %s", devices.filesystem, info.Name())
		}
	}()

	switch devices.filesystem {
	case "xfs":
		err = exec.Command("mkfs.xfs", args...).Run()
	case "ext4":
		err = exec.Command("mkfs.ext4", append([]string{"-E", "nodiscard,lazy_itable_init=0,lazy_journal_init=0"}, args...)...).Run()
		if err != nil {
			err = exec.Command("mkfs.ext4", append([]string{"-E", "nodiscard,lazy_itable_init=0"}, args...)...).Run()
		}
		if err != nil {
			return err
		}
		err = exec.Command("tune2fs", append([]string{"-c", "-1", "-i", "0"}, devname)...).Run()
	default:
		err = fmt.Errorf("devmapper: Unsupported filesystem type %s", devices.filesystem)
	}
	return
}

func (devices *DeviceSet) migrateOldMetaData() error {
	// Migrate old metadata file
	jsonData, err := ioutil.ReadFile(devices.oldMetadataFile())
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	if jsonData != nil {
		m := metaData{Devices: make(map[string]*devInfo)}

		if err := json.Unmarshal(jsonData, &m); err != nil {
			return err
		}

		for hash, info := range m.Devices {
			info.Hash = hash
			devices.saveMetadata(info)
		}
		if err := os.Rename(devices.oldMetadataFile(), devices.oldMetadataFile()+".migrated"); err != nil {
			return err
		}

	}

	return nil
}

// Cleanup deleted devices. It assumes that all the devices have been
// loaded in the hash table.
func (devices *DeviceSet) cleanupDeletedDevices() error {
	devices.Lock()

	// If there are no deleted devices, there is nothing to do.
	if devices.nrDeletedDevices == 0 {
		devices.Unlock()
		return nil
	}

	var deletedDevices []*devInfo

	for _, info := range devices.Devices {
		if !info.Deleted {
			continue
		}
		logrus.Debugf("devmapper: Found deleted device %s.", info.Hash)
		deletedDevices = append(deletedDevices, info)
	}

	// Delete the deleted devices. DeleteDevice() first takes the info lock
	// and then devices.Lock(). So drop it to avoid deadlock.
	devices.Unlock()

	for _, info := range deletedDevices {
		// This will again try deferred deletion.
		if err := devices.DeleteDevice(info.Hash, false); err != nil {
			logrus.Warnf("devmapper: Deletion of device %s, device_id=%v failed:%v", info.Hash, info.DeviceID, err)
		}
	}

	return nil
}

func (devices *DeviceSet) countDeletedDevices() {
	for _, info := range devices.Devices {
		if !info.Deleted {
			continue
		}
		devices.nrDeletedDevices++
	}
}

func (devices *DeviceSet) startDeviceDeletionWorker() {
	// Deferred deletion is not enabled. Don't do anything.
	if !devices.deferredDelete {
		return
	}

	logrus.Debug("devmapper: Worker to cleanup deleted devices started")
	for range devices.deletionWorkerTicker.C {
		devices.cleanupDeletedDevices()
	}
}

func (devices *DeviceSet) initMetaData() error {
	devices.Lock()
	defer devices.Unlock()

	if err := devices.migrateOldMetaData(); err != nil {
		return err
	}

	_, transactionID, _, _, _, _, err := devices.poolStatus()
	if err != nil {
		return err
	}

	devices.TransactionID = transactionID

	if err := devices.loadDeviceFilesOnStart(); err != nil {
		return fmt.Errorf("devmapper: Failed to load device files:%v", err)
	}

	devices.constructDeviceIDMap()
	devices.countDeletedDevices()

	if err := devices.processPendingTransaction(); err != nil {
		return err
	}

	// Start a goroutine to cleanup Deleted Devices
	go devices.startDeviceDeletionWorker()
	return nil
}

func (devices *DeviceSet) incNextDeviceID() {
	// IDs are 24bit, so wrap around
	devices.NextDeviceID = (devices.NextDeviceID + 1) & maxDeviceID
}

func (devices *DeviceSet) getNextFreeDeviceID() (int, error) {
	devices.incNextDeviceID()
	for i := 0; i <= maxDeviceID; i++ {
		if devices.isDeviceIDFree(devices.NextDeviceID) {
			devices.markDeviceIDUsed(devices.NextDeviceID)
			return devices.NextDeviceID, nil
		}
		devices.incNextDeviceID()
	}

	return 0, fmt.Errorf("devmapper: Unable to find a free device ID")
}

func (devices *DeviceSet) poolHasFreeSpace() error {
	if devices.minFreeSpacePercent == 0 {
		return nil
	}

	_, _, dataUsed, dataTotal, metadataUsed, metadataTotal, err := devices.poolStatus()
	if err != nil {
		return err
	}

	minFreeData := (dataTotal * uint64(devices.minFreeSpacePercent)) / 100
	if minFreeData < 1 {
		minFreeData = 1
	}
	dataFree := dataTotal - dataUsed
	if dataFree < minFreeData {
		return fmt.Errorf("devmapper: Thin Pool has %v free data blocks which is less than minimum required %v free data blocks. Create more free space in thin pool or use dm.min_free_space option to change behavior", (dataTotal - dataUsed), minFreeData)
	}

	minFreeMetadata := (metadataTotal * uint64(devices.minFreeSpacePercent)) / 100
	if minFreeMetadata < 1 {
		minFreeMetadata = 1
	}

	metadataFree := metadataTotal - metadataUsed
	if metadataFree < minFreeMetadata {
		return fmt.Errorf("devmapper: Thin Pool has %v free metadata blocks which is less than minimum required %v free metadata blocks. Create more free metadata space in thin pool or use dm.min_free_space option to change behavior", (metadataTotal - metadataUsed), minFreeMetadata)
	}

	return nil
}

func (devices *DeviceSet) createRegisterDevice(hash string) (*devInfo, error) {
	devices.Lock()
	defer devices.Unlock()

	deviceID, err := devices.getNextFreeDeviceID()
	if err != nil {
		return nil, err
	}

	if err := devices.openTransaction(hash, deviceID); err != nil {
		logrus.Debugf("devmapper: Error opening transaction hash = %s deviceID = %d", hash, deviceID)
		devices.markDeviceIDFree(deviceID)
		return nil, err
	}

	for {
		if err := devicemapper.CreateDevice(devices.getPoolDevName(), deviceID); err != nil {
			if devicemapper.DeviceIDExists(err) {
				// Device ID already exists. This should not
				// happen. Now we have a mechanism to find
				// a free device ID. So something is not right.
				// Give a warning and continue.
				logrus.Errorf("devmapper: Device ID %d exists in pool but it is supposed to be unused", deviceID)
				deviceID, err = devices.getNextFreeDeviceID()
				if err != nil {
					return nil, err
				}
				// Save new device id into transaction
				devices.refreshTransaction(deviceID)
				continue
			}
			logrus.Debugf("devmapper: Error creating device: %s", err)
			devices.markDeviceIDFree(deviceID)
			return nil, err
		}
		break
	}

	logrus.Debugf("devmapper: Registering device (id %v) with FS size %v", deviceID, devices.baseFsSize)
	info, err := devices.registerDevice(deviceID, hash, devices.baseFsSize, devices.OpenTransactionID)
	if err != nil {
		_ = devicemapper.DeleteDevice(devices.getPoolDevName(), deviceID)
		devices.markDeviceIDFree(deviceID)
		return nil, err
	}

	if err := devices.closeTransaction(); err != nil {
		devices.unregisterDevice(hash)
		devicemapper.DeleteDevice(devices.getPoolDevName(), deviceID)
		devices.markDeviceIDFree(deviceID)
		return nil, err
	}
	return info, nil
}

func (devices *DeviceSet) takeSnapshot(hash string, baseInfo *devInfo, size uint64) error {
	var (
		devinfo *devicemapper.Info
		err     error
	)

	if err = devices.poolHasFreeSpace(); err != nil {
		return err
	}

	if devices.deferredRemove {
		devinfo, err = devicemapper.GetInfoWithDeferred(baseInfo.Name())
		if err != nil {
			return err
		}
		if devinfo != nil && devinfo.DeferredRemove != 0 {
			err = devices.cancelDeferredRemoval(baseInfo)
			if err != nil {
				// If Error is ErrEnxio. Device is probably already gone. Continue.
				if err != devicemapper.ErrEnxio {
					return err
				}
				devinfo = nil
			} else {
				defer devices.deactivateDevice(baseInfo)
			}
		}
	} else {
		devinfo, err = devicemapper.GetInfo(baseInfo.Name())
		if err != nil {
			return err
		}
	}

	doSuspend := devinfo != nil && devinfo.Exists != 0

	if doSuspend {
		if err = devicemapper.SuspendDevice(baseInfo.Name()); err != nil {
			return err
		}
		defer devicemapper.ResumeDevice(baseInfo.Name())
	}

	if err = devices.createRegisterSnapDevice(hash, baseInfo, size); err != nil {
		return err
	}

	return nil
}

func (devices *DeviceSet) createRegisterSnapDevice(hash string, baseInfo *devInfo, size uint64) error {
	deviceID, err := devices.getNextFreeDeviceID()
	if err != nil {
		return err
	}

	if err := devices.openTransaction(hash, deviceID); err != nil {
		logrus.Debugf("devmapper: Error opening transaction hash = %s deviceID = %d", hash, deviceID)
		devices.markDeviceIDFree(deviceID)
		return err
	}

	for {
		if err := devicemapper.CreateSnapDeviceRaw(devices.getPoolDevName(), deviceID, baseInfo.DeviceID); err != nil {
			if devicemapper.DeviceIDExists(err) {
				// Device ID already exists. This should not
				// happen. Now we have a mechanism to find
				// a free device ID. So something is not right.
				// Give a warning and continue.
				logrus.Errorf("devmapper: Device ID %d exists in pool but it is supposed to be unused", deviceID)
				deviceID, err = devices.getNextFreeDeviceID()
				if err != nil {
					return err
				}
				// Save new device id into transaction
				devices.refreshTransaction(deviceID)
				continue
			}
			logrus.Debugf("devmapper: Error creating snap device: %s", err)
			devices.markDeviceIDFree(deviceID)
			return err
		}
		break
	}

	if _, err := devices.registerDevice(deviceID, hash, size, devices.OpenTransactionID); err != nil {
		devicemapper.DeleteDevice(devices.getPoolDevName(), deviceID)
		devices.markDeviceIDFree(deviceID)
		logrus.Debugf("devmapper: Error registering device: %s", err)
		return err
	}

	if err := devices.closeTransaction(); err != nil {
		devices.unregisterDevice(hash)
		devicemapper.DeleteDevice(devices.getPoolDevName(), deviceID)
		devices.markDeviceIDFree(deviceID)
		return err
	}
	return nil
}

func (devices *DeviceSet) loadMetadata(hash string) *devInfo {
	info := &devInfo{Hash: hash, devices: devices}

	jsonData, err := ioutil.ReadFile(devices.metadataFile(info))
	if err != nil {
		logrus.Debugf("devmapper: Failed to read %s with err: %v", devices.metadataFile(info), err)
		return nil
	}

	if err := json.Unmarshal(jsonData, &info); err != nil {
		logrus.Debugf("devmapper: Failed to unmarshal devInfo from %s with err: %v", devices.metadataFile(info), err)
		return nil
	}

	if info.DeviceID > maxDeviceID {
		logrus.Errorf("devmapper: Ignoring Invalid DeviceId=%d", info.DeviceID)
		return nil
	}

	return info
}

func getDeviceUUID(device string) (string, error) {
	out, err := exec.Command("blkid", "-s", "UUID", "-o", "value", device).Output()
	if err != nil {
		return "", fmt.Errorf("devmapper: Failed to find uuid for device %s:%v", device, err)
	}

	uuid := strings.TrimSuffix(string(out), "\n")
	uuid = strings.TrimSpace(uuid)
	logrus.Debugf("devmapper: UUID for device: %s is:%s", device, uuid)
	return uuid, nil
}

func (devices *DeviceSet) getBaseDeviceSize() uint64 {
	info, _ := devices.lookupDevice("")
	if info == nil {
		return 0
	}
	return info.Size
}

func (devices *DeviceSet) getBaseDeviceFS() string {
	return devices.BaseDeviceFilesystem
}

func (devices *DeviceSet) verifyBaseDeviceUUIDFS(baseInfo *devInfo) error {
	devices.Lock()
	defer devices.Unlock()

	if err := devices.activateDeviceIfNeeded(baseInfo, false); err != nil {
		return err
	}
	defer devices.deactivateDevice(baseInfo)

	uuid, err := getDeviceUUID(baseInfo.DevName())
	if err != nil {
		return err
	}

	if devices.BaseDeviceUUID != uuid {
		return fmt.Errorf("devmapper: Current Base Device UUID:%s does not match with stored UUID:%s. Possibly using a different thin pool than last invocation", uuid, devices.BaseDeviceUUID)
	}

	if devices.BaseDeviceFilesystem == "" {
		fsType, err := ProbeFsType(baseInfo.DevName())
		if err != nil {
			return err
		}
		if err := devices.saveBaseDeviceFilesystem(fsType); err != nil {
			return err
		}
	}

	// If user specified a filesystem using dm.fs option and current
	// file system of base image is not same, warn user that dm.fs
	// will be ignored.
	if devices.BaseDeviceFilesystem != devices.filesystem {
		logrus.Warnf("devmapper: Base device already exists and has filesystem %s on it. User specified filesystem %s will be ignored.", devices.BaseDeviceFilesystem, devices.filesystem)
		devices.filesystem = devices.BaseDeviceFilesystem
	}
	return nil
}

func (devices *DeviceSet) saveBaseDeviceFilesystem(fs string) error {
	devices.BaseDeviceFilesystem = fs
	return devices.saveDeviceSetMetaData()
}

func (devices *DeviceSet) saveBaseDeviceUUID(baseInfo *devInfo) error {
	devices.Lock()
	defer devices.Unlock()

	if err := devices.activateDeviceIfNeeded(baseInfo, false); err != nil {
		return err
	}
	defer devices.deactivateDevice(baseInfo)

	uuid, err := getDeviceUUID(baseInfo.DevName())
	if err != nil {
		return err
	}

	devices.BaseDeviceUUID = uuid
	return devices.saveDeviceSetMetaData()
}

func (devices *DeviceSet) createBaseImage() error {
	logrus.Debug("devmapper: Initializing base device-mapper thin volume")

	// Create initial device
	info, err := devices.createRegisterDevice("")
	if err != nil {
		return err
	}

	logrus.Debug("devmapper: Creating filesystem on base device-mapper thin volume")

	if err := devices.activateDeviceIfNeeded(info, false); err != nil {
		return err
	}

	if err := devices.createFilesystem(info); err != nil {
		return err
	}

	info.Initialized = true
	if err := devices.saveMetadata(info); err != nil {
		info.Initialized = false
		return err
	}

	if err := devices.saveBaseDeviceUUID(info); err != nil {
		return fmt.Errorf("devmapper: Could not query and save base device UUID:%v", err)
	}

	return nil
}

// Returns if thin pool device exists or not. If device exists, also makes
// sure it is a thin pool device and not some other type of device.
func (devices *DeviceSet) thinPoolExists(thinPoolDevice string) (bool, error) {
	logrus.Debugf("devmapper: Checking for existence of the pool %s", thinPoolDevice)

	info, err := devicemapper.GetInfo(thinPoolDevice)
	if err != nil {
		return false, fmt.Errorf("devmapper: GetInfo() on device %s failed: %v", thinPoolDevice, err)
	}

	// Device does not exist.
	if info.Exists == 0 {
		return false, nil
	}

	_, _, deviceType, _, err := devicemapper.GetStatus(thinPoolDevice)
	if err != nil {
		return false, fmt.Errorf("devmapper: GetStatus() on device %s failed: %v", thinPoolDevice, err)
	}

	if deviceType != "thin-pool" {
		return false, fmt.Errorf("devmapper: Device %s is not a thin pool", thinPoolDevice)
	}

	return true, nil
}

func (devices *DeviceSet) checkThinPool() error {
	_, transactionID, dataUsed, _, _, _, err := devices.poolStatus()
	if err != nil {
		return err
	}
	if dataUsed != 0 {
		return fmt.Errorf("devmapper: Unable to take ownership of thin-pool (%s) that already has used data blocks",
			devices.thinPoolDevice)
	}
	if transactionID != 0 {
		return fmt.Errorf("devmapper: Unable to take ownership of thin-pool (%s) with non-zero transaction ID",
			devices.thinPoolDevice)
	}
	return nil
}

// Base image is initialized properly. Either save UUID for first time (for
// upgrade case or verify UUID.
func (devices *DeviceSet) setupVerifyBaseImageUUIDFS(baseInfo *devInfo) error {
	// If BaseDeviceUUID is nil (upgrade case), save it and return success.
	if devices.BaseDeviceUUID == "" {
		if err := devices.saveBaseDeviceUUID(baseInfo); err != nil {
			return fmt.Errorf("devmapper: Could not query and save base device UUID:%v", err)
		}
		return nil
	}

	if err := devices.verifyBaseDeviceUUIDFS(baseInfo); err != nil {
		return fmt.Errorf("devmapper: Base Device UUID and Filesystem verification failed: %v", err)
	}

	return nil
}

func (devices *DeviceSet) checkGrowBaseDeviceFS(info *devInfo) error {

	if !userBaseSize {
		return nil
	}

	if devices.baseFsSize < devices.getBaseDeviceSize() {
		return fmt.Errorf("devmapper: Base device size cannot be smaller than %s", units.HumanSize(float64(devices.getBaseDeviceSize())))
	}

	if devices.baseFsSize == devices.getBaseDeviceSize() {
		return nil
	}

	info.lock.Lock()
	defer info.lock.Unlock()

	devices.Lock()
	defer devices.Unlock()

	info.Size = devices.baseFsSize

	if err := devices.saveMetadata(info); err != nil {
		// Try to remove unused device
		delete(devices.Devices, info.Hash)
		return err
	}

	return devices.growFS(info)
}

func (devices *DeviceSet) growFS(info *devInfo) error {
	if err := devices.activateDeviceIfNeeded(info, false); err != nil {
		return fmt.Errorf("Error activating devmapper device: %s", err)
	}

	defer devices.deactivateDevice(info)

	fsMountPoint := "/run/docker/mnt"
	if _, err := os.Stat(fsMountPoint); os.IsNotExist(err) {
		if err := os.MkdirAll(fsMountPoint, 0700); err != nil {
			return err
		}
		defer os.RemoveAll(fsMountPoint)
	}

	options := ""
	if devices.BaseDeviceFilesystem == "xfs" {
		// XFS needs nouuid or it can't mount filesystems with the same fs
		options = joinMountOptions(options, "nouuid")
	}
	options = joinMountOptions(options, devices.mountOptions)

	if err := mount.Mount(info.DevName(), fsMountPoint, devices.BaseDeviceFilesystem, options); err != nil {
		return fmt.Errorf("Error mounting '%s' on '%s': %s", info.DevName(), fsMountPoint, err)
	}

	defer unix.Unmount(fsMountPoint, unix.MNT_DETACH)

	switch devices.BaseDeviceFilesystem {
	case "ext4":
		if out, err := exec.Command("resize2fs", info.DevName()).CombinedOutput(); err != nil {
			return fmt.Errorf("Failed to grow rootfs:%v:%s", err, string(out))
		}
	case "xfs":
		if out, err := exec.Command("xfs_growfs", info.DevName()).CombinedOutput(); err != nil {
			return fmt.Errorf("Failed to grow rootfs:%v:%s", err, string(out))
		}
	default:
		return fmt.Errorf("Unsupported filesystem type %s", devices.BaseDeviceFilesystem)
	}
	return nil
}

func (devices *DeviceSet) setupBaseImage() error {
	oldInfo, _ := devices.lookupDeviceWithLock("")

	// base image already exists. If it is initialized properly, do UUID
	// verification and return. Otherwise remove image and set it up
	// fresh.

	if oldInfo != nil {
		if oldInfo.Initialized && !oldInfo.Deleted {
			if err := devices.setupVerifyBaseImageUUIDFS(oldInfo); err != nil {
				return err
			}

			if err := devices.checkGrowBaseDeviceFS(oldInfo); err != nil {
				return err
			}

			return nil
		}

		logrus.Debug("devmapper: Removing uninitialized base image")
		// If previous base device is in deferred delete state,
		// that needs to be cleaned up first. So don't try
		// deferred deletion.
		if err := devices.DeleteDevice("", true); err != nil {
			return err
		}
	}

	// If we are setting up base image for the first time, make sure
	// thin pool is empty.
	if devices.thinPoolDevice != "" && oldInfo == nil {
		if err := devices.checkThinPool(); err != nil {
			return err
		}
	}

	// Create new base image device
	if err := devices.createBaseImage(); err != nil {
		return err
	}

	return nil
}

func setCloseOnExec(name string) {
	if fileInfos, _ := ioutil.ReadDir("/proc/self/fd"); fileInfos != nil {
		for _, i := range fileInfos {
			link, _ := os.Readlink(filepath.Join("/proc/self/fd", i.Name()))
			if link == name {
				fd, err := strconv.Atoi(i.Name())
				if err == nil {
					unix.CloseOnExec(fd)
				}
			}
		}
	}
}

func major(device uint64) uint64 {
	return (device >> 8) & 0xfff
}

func minor(device uint64) uint64 {
	return (device & 0xff) | ((device >> 12) & 0xfff00)
}

// ResizePool increases the size of the pool.
func (devices *DeviceSet) ResizePool(size int64) error {
	dirname := devices.loopbackDir()
	datafilename := path.Join(dirname, "data")
	if len(devices.dataDevice) > 0 {
		datafilename = devices.dataDevice
	}
	metadatafilename := path.Join(dirname, "metadata")
	if len(devices.metadataDevice) > 0 {
		metadatafilename = devices.metadataDevice
	}

	datafile, err := os.OpenFile(datafilename, os.O_RDWR, 0)
	if datafile == nil {
		return err
	}
	defer datafile.Close()

	fi, err := datafile.Stat()
	if fi == nil {
		return err
	}

	if fi.Size() > size {
		return fmt.Errorf("devmapper: Can't shrink file")
	}

	dataloopback := loopback.FindLoopDeviceFor(datafile)
	if dataloopback == nil {
		return fmt.Errorf("devmapper: Unable to find loopback mount for: %s", datafilename)
	}
	defer dataloopback.Close()

	metadatafile, err := os.OpenFile(metadatafilename, os.O_RDWR, 0)
	if metadatafile == nil {
		return err
	}
	defer metadatafile.Close()

	metadataloopback := loopback.FindLoopDeviceFor(metadatafile)
	if metadataloopback == nil {
		return fmt.Errorf("devmapper: Unable to find loopback mount for: %s", metadatafilename)
	}
	defer metadataloopback.Close()

	// Grow loopback file
	if err := datafile.Truncate(size); err != nil {
		return fmt.Errorf("devmapper: Unable to grow loopback file: %s", err)
	}

	// Reload size for loopback device
	if err := loopback.SetCapacity(dataloopback); err != nil {
		return fmt.Errorf("Unable to update loopback capacity: %s", err)
	}

	// Suspend the pool
	if err := devicemapper.SuspendDevice(devices.getPoolName()); err != nil {
		return fmt.Errorf("devmapper: Unable to suspend pool: %s", err)
	}

	// Reload with the new block sizes
	if err := devicemapper.ReloadPool(devices.getPoolName(), dataloopback, metadataloopback, devices.thinpBlockSize); err != nil {
		return fmt.Errorf("devmapper: Unable to reload pool: %s", err)
	}

	// Resume the pool
	if err := devicemapper.ResumeDevice(devices.getPoolName()); err != nil {
		return fmt.Errorf("devmapper: Unable to resume pool: %s", err)
	}

	return nil
}

func (devices *DeviceSet) loadTransactionMetaData() error {
	jsonData, err := ioutil.ReadFile(devices.transactionMetaFile())
	if err != nil {
		// There is no active transaction. This will be the case
		// during upgrade.
		if os.IsNotExist(err) {
			devices.OpenTransactionID = devices.TransactionID
			return nil
		}
		return err
	}

	json.Unmarshal(jsonData, &devices.transaction)
	return nil
}

func (devices *DeviceSet) saveTransactionMetaData() error {
	jsonData, err := json.Marshal(&devices.transaction)
	if err != nil {
		return fmt.Errorf("devmapper: Error encoding metadata to json: %s", err)
	}

	return devices.writeMetaFile(jsonData, devices.transactionMetaFile())
}

func (devices *DeviceSet) removeTransactionMetaData() error {
	return os.RemoveAll(devices.transactionMetaFile())
}

func (devices *DeviceSet) rollbackTransaction() error {
	logrus.Debugf("devmapper: Rolling back open transaction: TransactionID=%d hash=%s device_id=%d", devices.OpenTransactionID, devices.DeviceIDHash, devices.DeviceID)

	// A device id might have already been deleted before transaction
	// closed. In that case this call will fail. Just leave a message
	// in case of failure.
	if err := devicemapper.DeleteDevice(devices.getPoolDevName(), devices.DeviceID); err != nil {
		logrus.Errorf("devmapper: Unable to delete device: %s", err)
	}

	dinfo := &devInfo{Hash: devices.DeviceIDHash}
	if err := devices.removeMetadata(dinfo); err != nil {
		logrus.Errorf("devmapper: Unable to remove metadata: %s", err)
	} else {
		devices.markDeviceIDFree(devices.DeviceID)
	}

	if err := devices.removeTransactionMetaData(); err != nil {
		logrus.Errorf("devmapper: Unable to remove transaction meta file %s: %s", devices.transactionMetaFile(), err)
	}

	return nil
}

func (devices *DeviceSet) processPendingTransaction() error {
	if err := devices.loadTransactionMetaData(); err != nil {
		return err
	}

	// If there was open transaction but pool transaction ID is same
	// as open transaction ID, nothing to roll back.
	if devices.TransactionID == devices.OpenTransactionID {
		return nil
	}

	// If open transaction ID is less than pool transaction ID, something
	// is wrong. Bail out.
	if devices.OpenTransactionID < devices.TransactionID {
		logrus.Errorf("devmapper: Open Transaction id %d is less than pool transaction id %d", devices.OpenTransactionID, devices.TransactionID)
		return nil
	}

	// Pool transaction ID is not same as open transaction. There is
	// a transaction which was not completed.
	if err := devices.rollbackTransaction(); err != nil {
		return fmt.Errorf("devmapper: Rolling back open transaction failed: %s", err)
	}

	devices.OpenTransactionID = devices.TransactionID
	return nil
}

func (devices *DeviceSet) loadDeviceSetMetaData() error {
	jsonData, err := ioutil.ReadFile(devices.deviceSetMetaFile())
	if err != nil {
		// For backward compatibility return success if file does
		// not exist.
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	return json.Unmarshal(jsonData, devices)
}

func (devices *DeviceSet) saveDeviceSetMetaData() error {
	jsonData, err := json.Marshal(devices)
	if err != nil {
		return fmt.Errorf("devmapper: Error encoding metadata to json: %s", err)
	}

	return devices.writeMetaFile(jsonData, devices.deviceSetMetaFile())
}

func (devices *DeviceSet) openTransaction(hash string, DeviceID int) error {
	devices.allocateTransactionID()
	devices.DeviceIDHash = hash
	devices.DeviceID = DeviceID
	if err := devices.saveTransactionMetaData(); err != nil {
		return fmt.Errorf("devmapper: Error saving transaction metadata: %s", err)
	}
	return nil
}

func (devices *DeviceSet) refreshTransaction(DeviceID int) error {
	devices.DeviceID = DeviceID
	if err := devices.saveTransactionMetaData(); err != nil {
		return fmt.Errorf("devmapper: Error saving transaction metadata: %s", err)
	}
	return nil
}

func (devices *DeviceSet) closeTransaction() error {
	if err := devices.updatePoolTransactionID(); err != nil {
		logrus.Debug("devmapper: Failed to close Transaction")
		return err
	}
	return nil
}

func determineDriverCapabilities(version string) error {
	/*
	 * Driver version 4.27.0 and greater support deferred activation
	 * feature.
	 */

	logrus.Debugf("devicemapper: driver version is %s", version)

	versionSplit := strings.Split(version, ".")
	major, err := strconv.Atoi(versionSplit[0])
	if err != nil {
		return graphdriver.ErrNotSupported
	}

	if major > 4 {
		driverDeferredRemovalSupport = true
		return nil
	}

	if major < 4 {
		return nil
	}

	minor, err := strconv.Atoi(versionSplit[1])
	if err != nil {
		return graphdriver.ErrNotSupported
	}

	/*
	 * If major is 4 and minor is 27, then there is no need to
	 * check for patch level as it can not be less than 0.
	 */
	if minor >= 27 {
		driverDeferredRemovalSupport = true
		return nil
	}

	return nil
}

// Determine the major and minor number of loopback device
func getDeviceMajorMinor(file *os.File) (uint64, uint64, error) {
	var stat unix.Stat_t
	err := unix.Stat(file.Name(), &stat)
	if err != nil {
		return 0, 0, err
	}

	dev := stat.Rdev
	majorNum := major(dev)
	minorNum := minor(dev)

	logrus.Debugf("devmapper: Major:Minor for device: %s is:%v:%v", file.Name(), majorNum, minorNum)
	return majorNum, minorNum, nil
}

// Given a file which is backing file of a loop back device, find the
// loopback device name and its major/minor number.
func getLoopFileDeviceMajMin(filename string) (string, uint64, uint64, error) {
	file, err := os.Open(filename)
	if err != nil {
		logrus.Debugf("devmapper: Failed to open file %s", filename)
		return "", 0, 0, err
	}

	defer file.Close()
	loopbackDevice := loopback.FindLoopDeviceFor(file)
	if loopbackDevice == nil {
		return "", 0, 0, fmt.Errorf("devmapper: Unable to find loopback mount for: %s", filename)
	}
	defer loopbackDevice.Close()

	Major, Minor, err := getDeviceMajorMinor(loopbackDevice)
	if err != nil {
		return "", 0, 0, err
	}
	return loopbackDevice.Name(), Major, Minor, nil
}

// Get the major/minor numbers of thin pool data and metadata devices
func (devices *DeviceSet) getThinPoolDataMetaMajMin() (uint64, uint64, uint64, uint64, error) {
	var params, poolDataMajMin, poolMetadataMajMin string

	_, _, _, params, err := devicemapper.GetTable(devices.getPoolName())
	if err != nil {
		return 0, 0, 0, 0, err
	}

	if _, err = fmt.Sscanf(params, "%s %s", &poolMetadataMajMin, &poolDataMajMin); err != nil {
		return 0, 0, 0, 0, err
	}

	logrus.Debugf("devmapper: poolDataMajMin=%s poolMetaMajMin=%s\n", poolDataMajMin, poolMetadataMajMin)

	poolDataMajMinorSplit := strings.Split(poolDataMajMin, ":")
	poolDataMajor, err := strconv.ParseUint(poolDataMajMinorSplit[0], 10, 32)
	if err != nil {
		return 0, 0, 0, 0, err
	}

	poolDataMinor, err := strconv.ParseUint(poolDataMajMinorSplit[1], 10, 32)
	if err != nil {
		return 0, 0, 0, 0, err
	}

	poolMetadataMajMinorSplit := strings.Split(poolMetadataMajMin, ":")
	poolMetadataMajor, err := strconv.ParseUint(poolMetadataMajMinorSplit[0], 10, 32)
	if err != nil {
		return 0, 0, 0, 0, err
	}

	poolMetadataMinor, err := strconv.ParseUint(poolMetadataMajMinorSplit[1], 10, 32)
	if err != nil {
		return 0, 0, 0, 0, err
	}

	return poolDataMajor, poolDataMinor, poolMetadataMajor, poolMetadataMinor, nil
}

func (devices *DeviceSet) loadThinPoolLoopBackInfo() error {
	poolDataMajor, poolDataMinor, poolMetadataMajor, poolMetadataMinor, err := devices.getThinPoolDataMetaMajMin()
	if err != nil {
		return err
	}

	dirname := devices.loopbackDir()

	// data device has not been passed in. So there should be a data file
	// which is being mounted as loop device.
	if devices.dataDevice == "" {
		datafilename := path.Join(dirname, "data")
		dataLoopDevice, dataMajor, dataMinor, err := getLoopFileDeviceMajMin(datafilename)
		if err != nil {
			return err
		}

		// Compare the two
		if poolDataMajor == dataMajor && poolDataMinor == dataMinor {
			devices.dataDevice = dataLoopDevice
			devices.dataLoopFile = datafilename
		}

	}

	// metadata device has not been passed in. So there should be a
	// metadata file which is being mounted as loop device.
	if devices.metadataDevice == "" {
		metadatafilename := path.Join(dirname, "metadata")
		metadataLoopDevice, metadataMajor, metadataMinor, err := getLoopFileDeviceMajMin(metadatafilename)
		if err != nil {
			return err
		}
		if poolMetadataMajor == metadataMajor && poolMetadataMinor == metadataMinor {
			devices.metadataDevice = metadataLoopDevice
			devices.metadataLoopFile = metadatafilename
		}
	}

	return nil
}

func (devices *DeviceSet) enableDeferredRemovalDeletion() error {

	// If user asked for deferred removal then check both libdm library
	// and kernel driver support deferred removal otherwise error out.
	if enableDeferredRemoval {
		if !driverDeferredRemovalSupport {
			return fmt.Errorf("devmapper: Deferred removal can not be enabled as kernel does not support it")
		}
		if !devicemapper.LibraryDeferredRemovalSupport {
			return fmt.Errorf("devmapper: Deferred removal can not be enabled as libdm does not support it")
		}
		logrus.Debug("devmapper: Deferred removal support enabled.")
		devices.deferredRemove = true
	}

	if enableDeferredDeletion {
		if !devices.deferredRemove {
			return fmt.Errorf("devmapper: Deferred deletion can not be enabled as deferred removal is not enabled. Enable deferred removal using --storage-opt dm.use_deferred_removal=true parameter")
		}
		logrus.Debug("devmapper: Deferred deletion support enabled.")
		devices.deferredDelete = true
	}
	return nil
}

func (devices *DeviceSet) initDevmapper(doInit bool) (retErr error) {
	if err := devices.enableDeferredRemovalDeletion(); err != nil {
		return err
	}

	// https://github.com/docker/docker/issues/4036
	if supported := devicemapper.UdevSetSyncSupport(true); !supported {
		if dockerversion.IAmStatic == "true" {
			logrus.Error("devmapper: Udev sync is not supported. This will lead to data loss and unexpected behavior. Install a dynamic binary to use devicemapper or select a different storage driver. For more information, see https://docs.docker.com/engine/reference/commandline/dockerd/#storage-driver-options")
		} else {
			logrus.Error("devmapper: Udev sync is not supported. This will lead to data loss and unexpected behavior. Install a more recent version of libdevmapper or select a different storage driver. For more information, see https://docs.docker.com/engine/reference/commandline/dockerd/#storage-driver-options")
		}

		if !devices.overrideUdevSyncCheck {
			return graphdriver.ErrNotSupported
		}
	}

	//create the root dir of the devmapper driver ownership to match this
	//daemon's remapped root uid/gid so containers can start properly
	uid, gid, err := idtools.GetRootUIDGID(devices.uidMaps, devices.gidMaps)
	if err != nil {
		return err
	}
	if err := idtools.MkdirAs(devices.root, 0700, uid, gid); err != nil && !os.IsExist(err) {
		return err
	}
	if err := os.MkdirAll(devices.metadataDir(), 0700); err != nil && !os.IsExist(err) {
		return err
	}

	prevSetupConfig, err := readLVMConfig(devices.root)
	if err != nil {
		return err
	}

	if !reflect.DeepEqual(devices.lvmSetupConfig, directLVMConfig{}) {
		if devices.thinPoolDevice != "" {
			return errors.New("cannot setup direct-lvm when `dm.thinpooldev` is also specified")
		}

		if !reflect.DeepEqual(prevSetupConfig, devices.lvmSetupConfig) {
			if !reflect.DeepEqual(prevSetupConfig, directLVMConfig{}) {
				return errors.New("changing direct-lvm config is not supported")
			}
			logrus.WithField("storage-driver", "devicemapper").WithField("direct-lvm-config", devices.lvmSetupConfig).Debugf("Setting up direct lvm mode")
			if err := verifyBlockDevice(devices.lvmSetupConfig.Device, lvmSetupConfigForce); err != nil {
				return err
			}
			if err := setupDirectLVM(devices.lvmSetupConfig); err != nil {
				return err
			}
			if err := writeLVMConfig(devices.root, devices.lvmSetupConfig); err != nil {
				return err
			}
		}
		devices.thinPoolDevice = "docker-thinpool"
		logrus.WithField("storage-driver", "devicemapper").Debugf("Setting dm.thinpooldev to %q", devices.thinPoolDevice)
	}

	// Set the device prefix from the device id and inode of the docker root dir
	var st unix.Stat_t
	if err := unix.Stat(devices.root, &st); err != nil {
		return fmt.Errorf("devmapper: Error looking up dir %s: %s", devices.root, err)
	}
	// "reg-" stands for "regular file".
	// In the future we might use "dev-" for "device file", etc.
	// docker-maj,min[-inode] stands for:
	//	- Managed by docker
	//	- The target of this device is at major <maj> and minor <min>
	//	- If <inode> is defined, use that file inside the device as a loopback image. Otherwise use the device itself.
	devices.devicePrefix = fmt.Sprintf("docker-%d:%d-%d", major(st.Dev), minor(st.Dev), st.Ino)
	logrus.Debugf("devmapper: Generated prefix: %s", devices.devicePrefix)

	// Check for the existence of the thin-pool device
	poolExists, err := devices.thinPoolExists(devices.getPoolName())
	if err != nil {
		return err
	}

	// It seems libdevmapper opens this without O_CLOEXEC, and go exec will not close files
	// that are not Close-on-exec,
	// so we add this badhack to make sure it closes itself
	setCloseOnExec("/dev/mapper/control")

	// Make sure the sparse images exist in <root>/devicemapper/data and
	// <root>/devicemapper/metadata

	createdLoopback := false

	// If the pool doesn't exist, create it
	if !poolExists && devices.thinPoolDevice == "" {
		logrus.Debug("devmapper: Pool doesn't exist. Creating it.")

		var (
			dataFile     *os.File
			metadataFile *os.File
		)

		if devices.dataDevice == "" {
			// Make sure the sparse images exist in <root>/devicemapper/data

			hasData := devices.hasImage("data")

			if !doInit && !hasData {
				return errors.New("loopback data file not found")
			}

			if !hasData {
				createdLoopback = true
			}

			data, err := devices.ensureImage("data", devices.dataLoopbackSize)
			if err != nil {
				logrus.Debugf("devmapper: Error device ensureImage (data): %s", err)
				return err
			}

			dataFile, err = loopback.AttachLoopDevice(data)
			if err != nil {
				return err
			}
			devices.dataLoopFile = data
			devices.dataDevice = dataFile.Name()
		} else {
			dataFile, err = os.OpenFile(devices.dataDevice, os.O_RDWR, 0600)
			if err != nil {
				return err
			}
		}
		defer dataFile.Close()

		if devices.metadataDevice == "" {
			// Make sure the sparse images exist in <root>/devicemapper/metadata

			hasMetadata := devices.hasImage("metadata")

			if !doInit && !hasMetadata {
				return errors.New("loopback metadata file not found")
			}

			if !hasMetadata {
				createdLoopback = true
			}

			metadata, err := devices.ensureImage("metadata", devices.metaDataLoopbackSize)
			if err != nil {
				logrus.Debugf("devmapper: Error device ensureImage (metadata): %s", err)
				return err
			}

			metadataFile, err = loopback.AttachLoopDevice(metadata)
			if err != nil {
				return err
			}
			devices.metadataLoopFile = metadata
			devices.metadataDevice = metadataFile.Name()
		} else {
			metadataFile, err = os.OpenFile(devices.metadataDevice, os.O_RDWR, 0600)
			if err != nil {
				return err
			}
		}
		defer metadataFile.Close()

		if err := devicemapper.CreatePool(devices.getPoolName(), dataFile, metadataFile, devices.thinpBlockSize); err != nil {
			return err
		}
		defer func() {
			if retErr != nil {
				err = devices.deactivatePool()
				if err != nil {
					logrus.Warnf("devmapper: Failed to deactivatePool: %v", err)
				}
			}
		}()
	}

	// Pool already exists and caller did not pass us a pool. That means
	// we probably created pool earlier and could not remove it as some
	// containers were still using it. Detect some of the properties of
	// pool, like is it using loop devices.
	if poolExists && devices.thinPoolDevice == "" {
		if err := devices.loadThinPoolLoopBackInfo(); err != nil {
			logrus.Debugf("devmapper: Failed to load thin pool loopback device information:%v", err)
			return err
		}
	}

	// If we didn't just create the data or metadata image, we need to
	// load the transaction id and migrate old metadata
	if !createdLoopback {
		if err := devices.initMetaData(); err != nil {
			return err
		}
	}

	if devices.thinPoolDevice == "" {
		if devices.metadataLoopFile != "" || devices.dataLoopFile != "" {
			logrus.Warn("devmapper: Usage of loopback devices is strongly discouraged for production use. Please use `--storage-opt dm.thinpooldev` or use `man docker` to refer to dm.thinpooldev section.")
		}
	}

	// Right now this loads only NextDeviceID. If there is more metadata
	// down the line, we might have to move it earlier.
	if err := devices.loadDeviceSetMetaData(); err != nil {
		return err
	}

	// Setup the base image
	if doInit {
		if err := devices.setupBaseImage(); err != nil {
			logrus.Debugf("devmapper: Error device setupBaseImage: %s", err)
			return err
		}
	}

	return nil
}

// AddDevice adds a device and registers in the hash.
func (devices *DeviceSet) AddDevice(hash, baseHash string, storageOpt map[string]string) error {
	logrus.Debugf("devmapper: AddDevice START(hash=%s basehash=%s)", hash, baseHash)
	defer logrus.Debugf("devmapper: AddDevice END(hash=%s basehash=%s)", hash, baseHash)

	// If a deleted device exists, return error.
	baseInfo, err := devices.lookupDeviceWithLock(baseHash)
	if err != nil {
		return err
	}

	if baseInfo.Deleted {
		return fmt.Errorf("devmapper: Base device %v has been marked for deferred deletion", baseInfo.Hash)
	}

	baseInfo.lock.Lock()
	defer baseInfo.lock.Unlock()

	devices.Lock()
	defer devices.Unlock()

	// Also include deleted devices in case hash of new device is
	// same as one of the deleted devices.
	if info, _ := devices.lookupDevice(hash); info != nil {
		return fmt.Errorf("devmapper: device %s already exists. Deleted=%v", hash, info.Deleted)
	}

	size, err := devices.parseStorageOpt(storageOpt)
	if err != nil {
		return err
	}

	if size == 0 {
		size = baseInfo.Size
	}

	if size < baseInfo.Size {
		return fmt.Errorf("devmapper: Container size cannot be smaller than %s", units.HumanSize(float64(baseInfo.Size)))
	}

	if err := devices.takeSnapshot(hash, baseInfo, size); err != nil {
		return err
	}

	// Grow the container rootfs.
	if size > baseInfo.Size {
		info, err := devices.lookupDevice(hash)
		if err != nil {
			return err
		}

		if err := devices.growFS(info); err != nil {
			return err
		}
	}

	return nil
}

func (devices *DeviceSet) parseStorageOpt(storageOpt map[string]string) (uint64, error) {

	// Read size to change the block device size per container.
	for key, val := range storageOpt {
		key := strings.ToLower(key)
		switch key {
		case "size":
			size, err := units.RAMInBytes(val)
			if err != nil {
				return 0, err
			}
			return uint64(size), nil
		default:
			return 0, fmt.Errorf("Unknown option %s", key)
		}
	}

	return 0, nil
}

func (devices *DeviceSet) markForDeferredDeletion(info *devInfo) error {
	// If device is already in deleted state, there is nothing to be done.
	if info.Deleted {
		return nil
	}

	logrus.Debugf("devmapper: Marking device %s for deferred deletion.", info.Hash)

	info.Deleted = true

	// save device metadata to reflect deleted state.
	if err := devices.saveMetadata(info); err != nil {
		info.Deleted = false
		return err
	}

	devices.nrDeletedDevices++
	return nil
}

// Should be called with devices.Lock() held.
func (devices *DeviceSet) deleteTransaction(info *devInfo, syncDelete bool) error {
	if err := devices.openTransaction(info.Hash, info.DeviceID); err != nil {
		logrus.Debugf("devmapper: Error opening transaction hash = %s deviceId = %d", "", info.DeviceID)
		return err
	}

	defer devices.closeTransaction()

	err := devicemapper.DeleteDevice(devices.getPoolDevName(), info.DeviceID)
	if err != nil {
		// If syncDelete is true, we want to return error. If deferred
		// deletion is not enabled, we return an error. If error is
		// something other then EBUSY, return an error.
		if syncDelete || !devices.deferredDelete || err != devicemapper.ErrBusy {
			logrus.Debugf("devmapper: Error deleting device: %s", err)
			return err
		}
	}

	if err == nil {
		if err := devices.unregisterDevice(info.Hash); err != nil {
			return err
		}
		// If device was already in deferred delete state that means
		// deletion was being tried again later. Reduce the deleted
		// device count.
		if info.Deleted {
			devices.nrDeletedDevices--
		}
		devices.markDeviceIDFree(info.DeviceID)
	} else {
		if err := devices.markForDeferredDeletion(info); err != nil {
			return err
		}
	}

	return nil
}

// Issue discard only if device open count is zero.
func (devices *DeviceSet) issueDiscard(info *devInfo) error {
	logrus.Debugf("devmapper: issueDiscard START(device: %s).", info.Hash)
	defer logrus.Debugf("devmapper: issueDiscard END(device: %s).", info.Hash)
	// This is a workaround for the kernel not discarding block so
	// on the thin pool when we remove a thinp device, so we do it
	// manually.
	// Even if device is deferred deleted, activate it and issue
	// discards.
	if err := devices.activateDeviceIfNeeded(info, true); err != nil {
		return err
	}

	devinfo, err := devicemapper.GetInfo(info.Name())
	if err != nil {
		return err
	}

	if devinfo.OpenCount != 0 {
		logrus.Debugf("devmapper: Device: %s is in use. OpenCount=%d. Not issuing discards.", info.Hash, devinfo.OpenCount)
		return nil
	}

	if err := devicemapper.BlockDeviceDiscard(info.DevName()); err != nil {
		logrus.Debugf("devmapper: Error discarding block on device: %s (ignoring)", err)
	}
	return nil
}

// Should be called with devices.Lock() held.
func (devices *DeviceSet) deleteDevice(info *devInfo, syncDelete bool) error {
	if devices.doBlkDiscard {
		devices.issueDiscard(info)
	}

	// Try to deactivate device in case it is active.
	// If deferred removal is enabled and deferred deletion is disabled
	// then make sure device is removed synchronously. There have been
	// some cases of device being busy for short duration and we would
	// rather busy wait for device removal to take care of these cases.
	deferredRemove := devices.deferredRemove
	if !devices.deferredDelete {
		deferredRemove = false
	}

	if err := devices.deactivateDeviceMode(info, deferredRemove); err != nil {
		logrus.Debugf("devmapper: Error deactivating device: %s", err)
		return err
	}

	if err := devices.deleteTransaction(info, syncDelete); err != nil {
		return err
	}

	return nil
}

// DeleteDevice will return success if device has been marked for deferred
// removal. If one wants to override that and want DeleteDevice() to fail if
// device was busy and could not be deleted, set syncDelete=true.
func (devices *DeviceSet) DeleteDevice(hash string, syncDelete bool) error {
	logrus.Debugf("devmapper: DeleteDevice START(hash=%v syncDelete=%v)", hash, syncDelete)
	defer logrus.Debugf("devmapper: DeleteDevice END(hash=%v syncDelete=%v)", hash, syncDelete)
	info, err := devices.lookupDeviceWithLock(hash)
	if err != nil {
		return err
	}

	info.lock.Lock()
	defer info.lock.Unlock()

	devices.Lock()
	defer devices.Unlock()

	return devices.deleteDevice(info, syncDelete)
}

func (devices *DeviceSet) deactivatePool() error {
	logrus.Debug("devmapper: deactivatePool() START")
	defer logrus.Debug("devmapper: deactivatePool() END")
	devname := devices.getPoolDevName()

	devinfo, err := devicemapper.GetInfo(devname)
	if err != nil {
		return err
	}

	if devinfo.Exists == 0 {
		return nil
	}
	if err := devicemapper.RemoveDevice(devname); err != nil {
		return err
	}

	if d, err := devicemapper.GetDeps(devname); err == nil {
		logrus.Warnf("devmapper: device %s still has %d active dependents", devname, d.Count)
	}

	return nil
}

func (devices *DeviceSet) deactivateDevice(info *devInfo) error {
	return devices.deactivateDeviceMode(info, devices.deferredRemove)
}

func (devices *DeviceSet) deactivateDeviceMode(info *devInfo, deferredRemove bool) error {
	var err error
	logrus.Debugf("devmapper: deactivateDevice START(%s)", info.Hash)
	defer logrus.Debugf("devmapper: deactivateDevice END(%s)", info.Hash)

	devinfo, err := devicemapper.GetInfo(info.Name())
	if err != nil {
		return err
	}

	if devinfo.Exists == 0 {
		return nil
	}

	if deferredRemove {
		err = devicemapper.RemoveDeviceDeferred(info.Name())
	} else {
		err = devices.removeDevice(info.Name())
	}

	// This function's semantics is such that it does not return an
	// error if device does not exist. So if device went away by
	// the time we actually tried to remove it, do not return error.
	if err != devicemapper.ErrEnxio {
		return err
	}
	return nil
}

// Issues the underlying dm remove operation.
func (devices *DeviceSet) removeDevice(devname string) error {
	var err error

	logrus.Debugf("devmapper: removeDevice START(%s)", devname)
	defer logrus.Debugf("devmapper: removeDevice END(%s)", devname)

	for i := 0; i < 200; i++ {
		err = devicemapper.RemoveDevice(devname)
		if err == nil {
			break
		}
		if err != devicemapper.ErrBusy {
			return err
		}

		// If we see EBUSY it may be a transient error,
		// sleep a bit a retry a few times.
		devices.Unlock()
		time.Sleep(100 * time.Millisecond)
		devices.Lock()
	}

	return err
}

func (devices *DeviceSet) cancelDeferredRemovalIfNeeded(info *devInfo) error {
	if !devices.deferredRemove {
		return nil
	}

	logrus.Debugf("devmapper: cancelDeferredRemovalIfNeeded START(%s)", info.Name())
	defer logrus.Debugf("devmapper: cancelDeferredRemovalIfNeeded END(%s)", info.Name())

	devinfo, err := devicemapper.GetInfoWithDeferred(info.Name())
	if err != nil {
		return err
	}

	if devinfo != nil && devinfo.DeferredRemove == 0 {
		return nil
	}

	// Cancel deferred remove
	if err := devices.cancelDeferredRemoval(info); err != nil {
		// If Error is ErrEnxio. Device is probably already gone. Continue.
		if err != devicemapper.ErrEnxio {
			return err
		}
	}
	return nil
}

func (devices *DeviceSet) cancelDeferredRemoval(info *devInfo) error {
	logrus.Debugf("devmapper: cancelDeferredRemoval START(%s)", info.Name())
	defer logrus.Debugf("devmapper: cancelDeferredRemoval END(%s)", info.Name())

	var err error

	// Cancel deferred remove
	for i := 0; i < 100; i++ {
		err = devicemapper.CancelDeferredRemove(info.Name())
		if err != nil {
			if err == devicemapper.ErrBusy {
				// If we see EBUSY it may be a transient error,
				// sleep a bit a retry a few times.
				devices.Unlock()
				time.Sleep(100 * time.Millisecond)
				devices.Lock()
				continue
			}
		}
		break
	}
	return err
}

// Shutdown shuts down the device by unmounting the root.
func (devices *DeviceSet) Shutdown(home string) error {
	logrus.Debugf("devmapper: [deviceset %s] Shutdown()", devices.devicePrefix)
	logrus.Debugf("devmapper: Shutting down DeviceSet: %s", devices.root)
	defer logrus.Debugf("devmapper: [deviceset %s] Shutdown() END", devices.devicePrefix)

	// Stop deletion worker. This should start delivering new events to
	// ticker channel. That means no new instance of cleanupDeletedDevice()
	// will run after this call. If one instance is already running at
	// the time of the call, it must be holding devices.Lock() and
	// we will block on this lock till cleanup function exits.
	devices.deletionWorkerTicker.Stop()

	devices.Lock()
	// Save DeviceSet Metadata first. Docker kills all threads if they
	// don't finish in certain time. It is possible that Shutdown()
	// routine does not finish in time as we loop trying to deactivate
	// some devices while these are busy. In that case shutdown() routine
	// will be killed and we will not get a chance to save deviceset
	// metadata. Hence save this early before trying to deactivate devices.
	devices.saveDeviceSetMetaData()

	// ignore the error since it's just a best effort to not try to unmount something that's mounted
	mounts, _ := mount.GetMounts()
	mounted := make(map[string]bool, len(mounts))
	for _, mnt := range mounts {
		mounted[mnt.Mountpoint] = true
	}

	if err := filepath.Walk(path.Join(home, "mnt"), func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			return nil
		}

		if mounted[p] {
			// We use MNT_DETACH here in case it is still busy in some running
			// container. This means it'll go away from the global scope directly,
			// and the device will be released when that container dies.
			if err := unix.Unmount(p, unix.MNT_DETACH); err != nil {
				logrus.Debugf("devmapper: Shutdown unmounting %s, error: %s", p, err)
			}
		}

		if devInfo, err := devices.lookupDevice(path.Base(p)); err != nil {
			logrus.Debugf("devmapper: Shutdown lookup device %s, error: %s", path.Base(p), err)
		} else {
			if err := devices.deactivateDevice(devInfo); err != nil {
				logrus.Debugf("devmapper: Shutdown deactivate %s , error: %s", devInfo.Hash, err)
			}
		}

		return nil
	}); err != nil && !os.IsNotExist(err) {
		devices.Unlock()
		return err
	}

	devices.Unlock()

	info, _ := devices.lookupDeviceWithLock("")
	if info != nil {
		info.lock.Lock()
		devices.Lock()
		if err := devices.deactivateDevice(info); err != nil {
			logrus.Debugf("devmapper: Shutdown deactivate base , error: %s", err)
		}
		devices.Unlock()
		info.lock.Unlock()
	}

	devices.Lock()
	if devices.thinPoolDevice == "" {
		if err := devices.deactivatePool(); err != nil {
			logrus.Debugf("devmapper: Shutdown deactivate pool , error: %s", err)
		}
	}
	devices.Unlock()

	return nil
}

// Recent XFS changes allow changing behavior of filesystem in case of errors.
// When thin pool gets full and XFS gets ENOSPC error, currently it tries
// IO infinitely and sometimes it can block the container process
// and process can't be killWith 0 value, XFS will not retry upon error
// and instead will shutdown filesystem.

func (devices *DeviceSet) xfsSetNospaceRetries(info *devInfo) error {
	dmDevicePath, err := os.Readlink(info.DevName())
	if err != nil {
		return fmt.Errorf("devmapper: readlink failed for device %v:%v", info.DevName(), err)
	}

	dmDeviceName := path.Base(dmDevicePath)
	filePath := "/sys/fs/xfs/" + dmDeviceName + "/error/metadata/ENOSPC/max_retries"
	maxRetriesFile, err := os.OpenFile(filePath, os.O_WRONLY, 0)
	if err != nil {
		return fmt.Errorf("devmapper: user specified daemon option dm.xfs_nospace_max_retries but it does not seem to be supported on this system :%v", err)
	}
	defer maxRetriesFile.Close()

	// Set max retries to 0
	_, err = maxRetriesFile.WriteString(devices.xfsNospaceRetries)
	if err != nil {
		return fmt.Errorf("devmapper: Failed to write string %v to file %v:%v", devices.xfsNospaceRetries, filePath, err)
	}
	return nil
}

// MountDevice mounts the device if not already mounted.
func (devices *DeviceSet) MountDevice(hash, path, mountLabel string) error {
	info, err := devices.lookupDeviceWithLock(hash)
	if err != nil {
		return err
	}

	if info.Deleted {
		return fmt.Errorf("devmapper: Can't mount device %v as it has been marked for deferred deletion", info.Hash)
	}

	info.lock.Lock()
	defer info.lock.Unlock()

	devices.Lock()
	defer devices.Unlock()

	if err := devices.activateDeviceIfNeeded(info, false); err != nil {
		return fmt.Errorf("devmapper: Error activating devmapper device for '%s': %s", hash, err)
	}

	fstype, err := ProbeFsType(info.DevName())
	if err != nil {
		return err
	}

	options := ""

	if fstype == "xfs" {
		// XFS needs nouuid or it can't mount filesystems with the same fs
		options = joinMountOptions(options, "nouuid")
	}

	options = joinMountOptions(options, devices.mountOptions)
	options = joinMountOptions(options, label.FormatMountLabel("", mountLabel))

	if err := mount.Mount(info.DevName(), path, fstype, options); err != nil {
		return fmt.Errorf("devmapper: Error mounting '%s' on '%s': %s", info.DevName(), path, err)
	}

	if fstype == "xfs" && devices.xfsNospaceRetries != "" {
		if err := devices.xfsSetNospaceRetries(info); err != nil {
			unix.Unmount(path, unix.MNT_DETACH)
			devices.deactivateDevice(info)
			return err
		}
	}

	return nil
}

// UnmountDevice unmounts the device and removes it from hash.
func (devices *DeviceSet) UnmountDevice(hash, mountPath string) error {
	logrus.Debugf("devmapper: UnmountDevice START(hash=%s)", hash)
	defer logrus.Debugf("devmapper: UnmountDevice END(hash=%s)", hash)

	info, err := devices.lookupDeviceWithLock(hash)
	if err != nil {
		return err
	}

	info.lock.Lock()
	defer info.lock.Unlock()

	devices.Lock()
	defer devices.Unlock()

	logrus.Debugf("devmapper: Unmount(%s)", mountPath)
	if err := unix.Unmount(mountPath, unix.MNT_DETACH); err != nil {
		return err
	}
	logrus.Debug("devmapper: Unmount done")

	return devices.deactivateDevice(info)
}

// HasDevice returns true if the device metadata exists.
func (devices *DeviceSet) HasDevice(hash string) bool {
	info, _ := devices.lookupDeviceWithLock(hash)
	return info != nil
}

// List returns a list of device ids.
func (devices *DeviceSet) List() []string {
	devices.Lock()
	defer devices.Unlock()

	ids := make([]string, len(devices.Devices))
	i := 0
	for k := range devices.Devices {
		ids[i] = k
		i++
	}
	return ids
}

func (devices *DeviceSet) deviceStatus(devName string) (sizeInSectors, mappedSectors, highestMappedSector uint64, err error) {
	var params string
	_, sizeInSectors, _, params, err = devicemapper.GetStatus(devName)
	if err != nil {
		return
	}
	if _, err = fmt.Sscanf(params, "%d %d", &mappedSectors, &highestMappedSector); err == nil {
		return
	}
	return
}

// GetDeviceStatus provides size, mapped sectors
func (devices *DeviceSet) GetDeviceStatus(hash string) (*DevStatus, error) {
	info, err := devices.lookupDeviceWithLock(hash)
	if err != nil {
		return nil, err
	}

	info.lock.Lock()
	defer info.lock.Unlock()

	devices.Lock()
	defer devices.Unlock()

	status := &DevStatus{
		DeviceID:      info.DeviceID,
		Size:          info.Size,
		TransactionID: info.TransactionID,
	}

	if err := devices.activateDeviceIfNeeded(info, false); err != nil {
		return nil, fmt.Errorf("devmapper: Error activating devmapper device for '%s': %s", hash, err)
	}

	sizeInSectors, mappedSectors, highestMappedSector, err := devices.deviceStatus(info.DevName())

	if err != nil {
		return nil, err
	}

	status.SizeInSectors = sizeInSectors
	status.MappedSectors = mappedSectors
	status.HighestMappedSector = highestMappedSector

	return status, nil
}

func (devices *DeviceSet) poolStatus() (totalSizeInSectors, transactionID, dataUsed, dataTotal, metadataUsed, metadataTotal uint64, err error) {
	var params string
	if _, totalSizeInSectors, _, params, err = devicemapper.GetStatus(devices.getPoolName()); err == nil {
		_, err = fmt.Sscanf(params, "%d %d/%d %d/%d", &transactionID, &metadataUsed, &metadataTotal, &dataUsed, &dataTotal)
	}
	return
}

// DataDevicePath returns the path to the data storage for this deviceset,
// regardless of loopback or block device
func (devices *DeviceSet) DataDevicePath() string {
	return devices.dataDevice
}

// MetadataDevicePath returns the path to the metadata storage for this deviceset,
// regardless of loopback or block device
func (devices *DeviceSet) MetadataDevicePath() string {
	return devices.metadataDevice
}

func (devices *DeviceSet) getUnderlyingAvailableSpace(loopFile string) (uint64, error) {
	buf := new(unix.Statfs_t)
	if err := unix.Statfs(loopFile, buf); err != nil {
		logrus.Warnf("devmapper: Couldn't stat loopfile filesystem %v: %v", loopFile, err)
		return 0, err
	}
	return buf.Bfree * uint64(buf.Bsize), nil
}

func (devices *DeviceSet) isRealFile(loopFile string) (bool, error) {
	if loopFile != "" {
		fi, err := os.Stat(loopFile)
		if err != nil {
			logrus.Warnf("devmapper: Couldn't stat loopfile %v: %v", loopFile, err)
			return false, err
		}
		return fi.Mode().IsRegular(), nil
	}
	return false, nil
}

// Status returns the current status of this deviceset
func (devices *DeviceSet) Status() *Status {
	devices.Lock()
	defer devices.Unlock()

	status := &Status{}

	status.PoolName = devices.getPoolName()
	status.DataFile = devices.DataDevicePath()
	status.DataLoopback = devices.dataLoopFile
	status.MetadataFile = devices.MetadataDevicePath()
	status.MetadataLoopback = devices.metadataLoopFile
	status.UdevSyncSupported = devicemapper.UdevSyncSupported()
	status.DeferredRemoveEnabled = devices.deferredRemove
	status.DeferredDeleteEnabled = devices.deferredDelete
	status.DeferredDeletedDeviceCount = devices.nrDeletedDevices
	status.BaseDeviceSize = devices.getBaseDeviceSize()
	status.BaseDeviceFS = devices.getBaseDeviceFS()

	totalSizeInSectors, _, dataUsed, dataTotal, metadataUsed, metadataTotal, err := devices.poolStatus()
	if err == nil {
		// Convert from blocks to bytes
		blockSizeInSectors := totalSizeInSectors / dataTotal

		status.Data.Used = dataUsed * blockSizeInSectors * 512
		status.Data.Total = dataTotal * blockSizeInSectors * 512
		status.Data.Available = status.Data.Total - status.Data.Used

		// metadata blocks are always 4k
		status.Metadata.Used = metadataUsed * 4096
		status.Metadata.Total = metadataTotal * 4096
		status.Metadata.Available = status.Metadata.Total - status.Metadata.Used

		status.SectorSize = blockSizeInSectors * 512

		if check, _ := devices.isRealFile(devices.dataLoopFile); check {
			actualSpace, err := devices.getUnderlyingAvailableSpace(devices.dataLoopFile)
			if err == nil && actualSpace < status.Data.Available {
				status.Data.Available = actualSpace
			}
		}

		if check, _ := devices.isRealFile(devices.metadataLoopFile); check {
			actualSpace, err := devices.getUnderlyingAvailableSpace(devices.metadataLoopFile)
			if err == nil && actualSpace < status.Metadata.Available {
				status.Metadata.Available = actualSpace
			}
		}

		minFreeData := (dataTotal * uint64(devices.minFreeSpacePercent)) / 100
		status.MinFreeSpace = minFreeData * blockSizeInSectors * 512
	}

	return status
}

// Status returns the current status of this deviceset
func (devices *DeviceSet) exportDeviceMetadata(hash string) (*deviceMetadata, error) {
	info, err := devices.lookupDeviceWithLock(hash)
	if err != nil {
		return nil, err
	}

	info.lock.Lock()
	defer info.lock.Unlock()

	metadata := &deviceMetadata{info.DeviceID, info.Size, info.Name()}
	return metadata, nil
}

// NewDeviceSet creates the device set based on the options provided.
func NewDeviceSet(root string, doInit bool, options []string, uidMaps, gidMaps []idtools.IDMap) (*DeviceSet, error) {
	devicemapper.SetDevDir("/dev")

	devices := &DeviceSet{
		root:                  root,
		metaData:              metaData{Devices: make(map[string]*devInfo)},
		dataLoopbackSize:      defaultDataLoopbackSize,
		metaDataLoopbackSize:  defaultMetaDataLoopbackSize,
		baseFsSize:            defaultBaseFsSize,
		overrideUdevSyncCheck: defaultUdevSyncOverride,
		doBlkDiscard:          true,
		thinpBlockSize:        defaultThinpBlockSize,
		deviceIDMap:           make([]byte, deviceIDMapSz),
		deletionWorkerTicker:  time.NewTicker(time.Second * 30),
		uidMaps:               uidMaps,
		gidMaps:               gidMaps,
		minFreeSpacePercent:   defaultMinFreeSpacePercent,
	}

	version, err := devicemapper.GetDriverVersion()
	if err != nil {
		// Can't even get driver version, assume not supported
		return nil, graphdriver.ErrNotSupported
	}

	if err := determineDriverCapabilities(version); err != nil {
		return nil, graphdriver.ErrNotSupported
	}

	if driverDeferredRemovalSupport && devicemapper.LibraryDeferredRemovalSupport {
		// enable deferred stuff by default
		enableDeferredDeletion = true
		enableDeferredRemoval = true
	}

	foundBlkDiscard := false
	var lvmSetupConfig directLVMConfig
	for _, option := range options {
		key, val, err := parsers.ParseKeyValueOpt(option)
		if err != nil {
			return nil, err
		}
		key = strings.ToLower(key)
		switch key {
		case "dm.basesize":
			size, err := units.RAMInBytes(val)
			if err != nil {
				return nil, err
			}
			userBaseSize = true
			devices.baseFsSize = uint64(size)
		case "dm.loopdatasize":
			size, err := units.RAMInBytes(val)
			if err != nil {
				return nil, err
			}
			devices.dataLoopbackSize = size
		case "dm.loopmetadatasize":
			size, err := units.RAMInBytes(val)
			if err != nil {
				return nil, err
			}
			devices.metaDataLoopbackSize = size
		case "dm.fs":
			if val != "ext4" && val != "xfs" {
				return nil, fmt.Errorf("devmapper: Unsupported filesystem %s\n", val)
			}
			devices.filesystem = val
		case "dm.mkfsarg":
			devices.mkfsArgs = append(devices.mkfsArgs, val)
		case "dm.mountopt":
			devices.mountOptions = joinMountOptions(devices.mountOptions, val)
		case "dm.metadatadev":
			devices.metadataDevice = val
		case "dm.datadev":
			devices.dataDevice = val
		case "dm.thinpooldev":
			devices.thinPoolDevice = strings.TrimPrefix(val, "/dev/mapper/")
		case "dm.blkdiscard":
			foundBlkDiscard = true
			devices.doBlkDiscard, err = strconv.ParseBool(val)
			if err != nil {
				return nil, err
			}
		case "dm.blocksize":
			size, err := units.RAMInBytes(val)
			if err != nil {
				return nil, err
			}
			// convert to 512b sectors
			devices.thinpBlockSize = uint32(size) >> 9
		case "dm.override_udev_sync_check":
			devices.overrideUdevSyncCheck, err = strconv.ParseBool(val)
			if err != nil {
				return nil, err
			}

		case "dm.use_deferred_removal":
			enableDeferredRemoval, err = strconv.ParseBool(val)
			if err != nil {
				return nil, err
			}

		case "dm.use_deferred_deletion":
			enableDeferredDeletion, err = strconv.ParseBool(val)
			if err != nil {
				return nil, err
			}

		case "dm.min_free_space":
			if !strings.HasSuffix(val, "%") {
				return nil, fmt.Errorf("devmapper: Option dm.min_free_space requires %% suffix")
			}

			valstring := strings.TrimSuffix(val, "%")
			minFreeSpacePercent, err := strconv.ParseUint(valstring, 10, 32)
			if err != nil {
				return nil, err
			}

			if minFreeSpacePercent >= 100 {
				return nil, fmt.Errorf("devmapper: Invalid value %v for option dm.min_free_space", val)
			}

			devices.minFreeSpacePercent = uint32(minFreeSpacePercent)
		case "dm.xfs_nospace_max_retries":
			_, err := strconv.ParseUint(val, 10, 64)
			if err != nil {
				return nil, err
			}
			devices.xfsNospaceRetries = val
		case "dm.directlvm_device":
			lvmSetupConfig.Device = val
		case "dm.directlvm_device_force":
			lvmSetupConfigForce, err = strconv.ParseBool(val)
			if err != nil {
				return nil, err
			}
		case "dm.thinp_percent":
			per, err := strconv.ParseUint(strings.TrimSuffix(val, "%"), 10, 32)
			if err != nil {
				return nil, errors.Wrapf(err, "could not parse `dm.thinp_percent=%s`", val)
			}
			if per >= 100 {
				return nil, errors.New("dm.thinp_percent must be greater than 0 and less than 100")
			}
			lvmSetupConfig.ThinpPercent = per
		case "dm.thinp_metapercent":
			per, err := strconv.ParseUint(strings.TrimSuffix(val, "%"), 10, 32)
			if err != nil {
				return nil, errors.Wrapf(err, "could not parse `dm.thinp_metapercent=%s`", val)
			}
			if per >= 100 {
				return nil, errors.New("dm.thinp_metapercent must be greater than 0 and less than 100")
			}
			lvmSetupConfig.ThinpMetaPercent = per
		case "dm.thinp_autoextend_percent":
			per, err := strconv.ParseUint(strings.TrimSuffix(val, "%"), 10, 32)
			if err != nil {
				return nil, errors.Wrapf(err, "could not parse `dm.thinp_autoextend_percent=%s`", val)
			}
			if per > 100 {
				return nil, errors.New("dm.thinp_autoextend_percent must be greater than 0 and less than 100")
			}
			lvmSetupConfig.AutoExtendPercent = per
		case "dm.thinp_autoextend_threshold":
			per, err := strconv.ParseUint(strings.TrimSuffix(val, "%"), 10, 32)
			if err != nil {
				return nil, errors.Wrapf(err, "could not parse `dm.thinp_autoextend_threshold=%s`", val)
			}
			if per > 100 {
				return nil, errors.New("dm.thinp_autoextend_threshold must be greater than 0 and less than 100")
			}
			lvmSetupConfig.AutoExtendThreshold = per
		case "dm.libdm_log_level":
			level, err := strconv.ParseInt(val, 10, 32)
			if err != nil {
				return nil, errors.Wrapf(err, "could not parse `dm.libdm_log_level=%s`", val)
			}
			if level < devicemapper.LogLevelFatal || level > devicemapper.LogLevelDebug {
				return nil, errors.Errorf("dm.libdm_log_level must be in range [%d,%d]", devicemapper.LogLevelFatal, devicemapper.LogLevelDebug)
			}
			// Register a new logging callback with the specified level.
			devicemapper.LogInit(devicemapper.DefaultLogger{
				Level: int(level),
			})
		default:
			return nil, fmt.Errorf("devmapper: Unknown option %s\n", key)
		}
	}

	if err := validateLVMConfig(lvmSetupConfig); err != nil {
		return nil, err
	}

	devices.lvmSetupConfig = lvmSetupConfig

	// By default, don't do blk discard hack on raw devices, its rarely useful and is expensive
	if !foundBlkDiscard && (devices.dataDevice != "" || devices.thinPoolDevice != "") {
		devices.doBlkDiscard = false
	}

	if err := devices.initDevmapper(doInit); err != nil {
		return nil, err
	}

	return devices, nil
}
