// +build linux

package devmapper

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/devicemapper"
	"github.com/docker/docker/pkg/parsers"
	"github.com/docker/docker/pkg/units"
	"github.com/opencontainers/runc/libcontainer/label"
)

var (
	DefaultDataLoopbackSize     int64  = 100 * 1024 * 1024 * 1024
	DefaultMetaDataLoopbackSize int64  = 2 * 1024 * 1024 * 1024
	DefaultBaseFsSize           uint64 = 100 * 1024 * 1024 * 1024
	DefaultThinpBlockSize       uint32 = 128 // 64K = 128 512b sectors
	DefaultUdevSyncOverride     bool   = false
	MaxDeviceId                 int    = 0xffffff // 24 bit, pool limit
	DeviceIdMapSz               int    = (MaxDeviceId + 1) / 8
	// We retry device removal so many a times that even error messages
	// will fill up console during normal operation. So only log Fatal
	// messages by default.
	DMLogLevel                   int  = devicemapper.LogLevelFatal
	DriverDeferredRemovalSupport bool = false
	EnableDeferredRemoval        bool = false
)

const deviceSetMetaFile string = "deviceset-metadata"
const transactionMetaFile string = "transaction-metadata"

type Transaction struct {
	OpenTransactionId uint64 `json:"open_transaction_id"`
	DeviceIdHash      string `json:"device_hash"`
	DeviceId          int    `json:"device_id"`
}

type DevInfo struct {
	Hash          string `json:"-"`
	DeviceId      int    `json:"device_id"`
	Size          uint64 `json:"size"`
	TransactionId uint64 `json:"transaction_id"`
	Initialized   bool   `json:"initialized"`
	devices       *DeviceSet

	mountCount int
	mountPath  string

	// The global DeviceSet lock guarantees that we serialize all
	// the calls to libdevmapper (which is not threadsafe), but we
	// sometimes release that lock while sleeping. In that case
	// this per-device lock is still held, protecting against
	// other accesses to the device that we're doing the wait on.
	//
	// WARNING: In order to avoid AB-BA deadlocks when releasing
	// the global lock while holding the per-device locks all
	// device locks must be aquired *before* the device lock, and
	// multiple device locks should be aquired parent before child.
	lock sync.Mutex
}

type MetaData struct {
	Devices     map[string]*DevInfo `json:"Devices"`
	devicesLock sync.Mutex          // Protects all read/writes to Devices map
}

type DeviceSet struct {
	MetaData      `json:"-"`
	sync.Mutex    `json:"-"` // Protects Devices map and serializes calls into libdevmapper
	root          string
	devicePrefix  string
	TransactionId uint64 `json:"-"`
	NextDeviceId  int    `json:"next_device_id"`
	deviceIdMap   []byte

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
	Transaction           `json:"-"`
	overrideUdevSyncCheck bool
	deferredRemove        bool   // use deferred removal
	BaseDeviceUUID        string //save UUID of base device
}

type DiskUsage struct {
	Used      uint64
	Total     uint64
	Available uint64
}

type Status struct {
	PoolName              string
	DataFile              string // actual block device for data
	DataLoopback          string // loopback file, if used
	MetadataFile          string // actual block device for metadata
	MetadataLoopback      string // loopback file, if used
	Data                  DiskUsage
	Metadata              DiskUsage
	SectorSize            uint64
	UdevSyncSupported     bool
	DeferredRemoveEnabled bool
}

// Structure used to export image/container metadata in docker inspect.
type DeviceMetadata struct {
	deviceId   int
	deviceSize uint64 // size in bytes
	deviceName string // Device name as used during activation
}

type DevStatus struct {
	DeviceId            int
	Size                uint64
	TransactionId       uint64
	SizeInSectors       uint64
	MappedSectors       uint64
	HighestMappedSector uint64
}

func getDevName(name string) string {
	return "/dev/mapper/" + name
}

func (info *DevInfo) Name() string {
	hash := info.Hash
	if hash == "" {
		hash = "base"
	}
	return fmt.Sprintf("%s-%s", info.devices.devicePrefix, hash)
}

func (info *DevInfo) DevName() string {
	return getDevName(info.Name())
}

func (devices *DeviceSet) loopbackDir() string {
	return path.Join(devices.root, "devicemapper")
}

func (devices *DeviceSet) metadataDir() string {
	return path.Join(devices.root, "metadata")
}

func (devices *DeviceSet) metadataFile(info *DevInfo) string {
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
// If the file already exists, it does nothing.
// Either way it returns the full path.
func (devices *DeviceSet) ensureImage(name string, size int64) (string, error) {
	dirname := devices.loopbackDir()
	filename := path.Join(dirname, name)

	if err := os.MkdirAll(dirname, 0700); err != nil && !os.IsExist(err) {
		return "", err
	}

	if _, err := os.Stat(filename); err != nil {
		if !os.IsNotExist(err) {
			return "", err
		}
		logrus.Debugf("Creating loopback file %s for device-manage use", filename)
		file, err := os.OpenFile(filename, os.O_RDWR|os.O_CREATE, 0600)
		if err != nil {
			return "", err
		}
		defer file.Close()

		if err := file.Truncate(size); err != nil {
			return "", err
		}
	}
	return filename, nil
}

func (devices *DeviceSet) allocateTransactionId() uint64 {
	devices.OpenTransactionId = devices.TransactionId + 1
	return devices.OpenTransactionId
}

func (devices *DeviceSet) updatePoolTransactionId() error {
	if err := devicemapper.SetTransactionId(devices.getPoolDevName(), devices.TransactionId, devices.OpenTransactionId); err != nil {
		return fmt.Errorf("Error setting devmapper transaction ID: %s", err)
	}
	devices.TransactionId = devices.OpenTransactionId
	return nil
}

func (devices *DeviceSet) removeMetadata(info *DevInfo) error {
	if err := os.RemoveAll(devices.metadataFile(info)); err != nil {
		return fmt.Errorf("Error removing metadata file %s: %s", devices.metadataFile(info), err)
	}
	return nil
}

// Given json data and file path, write it to disk
func (devices *DeviceSet) writeMetaFile(jsonData []byte, filePath string) error {
	tmpFile, err := ioutil.TempFile(devices.metadataDir(), ".tmp")
	if err != nil {
		return fmt.Errorf("Error creating metadata file: %s", err)
	}

	n, err := tmpFile.Write(jsonData)
	if err != nil {
		return fmt.Errorf("Error writing metadata to %s: %s", tmpFile.Name(), err)
	}
	if n < len(jsonData) {
		return io.ErrShortWrite
	}
	if err := tmpFile.Sync(); err != nil {
		return fmt.Errorf("Error syncing metadata file %s: %s", tmpFile.Name(), err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("Error closing metadata file %s: %s", tmpFile.Name(), err)
	}
	if err := os.Rename(tmpFile.Name(), filePath); err != nil {
		return fmt.Errorf("Error committing metadata file %s: %s", tmpFile.Name(), err)
	}

	return nil
}

func (devices *DeviceSet) saveMetadata(info *DevInfo) error {
	jsonData, err := json.Marshal(info)
	if err != nil {
		return fmt.Errorf("Error encoding metadata to json: %s", err)
	}
	if err := devices.writeMetaFile(jsonData, devices.metadataFile(info)); err != nil {
		return err
	}
	return nil
}

func (devices *DeviceSet) markDeviceIdUsed(deviceId int) {
	var mask byte
	i := deviceId % 8
	mask = 1 << uint(i)
	devices.deviceIdMap[deviceId/8] = devices.deviceIdMap[deviceId/8] | mask
}

func (devices *DeviceSet) markDeviceIdFree(deviceId int) {
	var mask byte
	i := deviceId % 8
	mask = ^(1 << uint(i))
	devices.deviceIdMap[deviceId/8] = devices.deviceIdMap[deviceId/8] & mask
}

func (devices *DeviceSet) isDeviceIdFree(deviceId int) bool {
	var mask byte
	i := deviceId % 8
	mask = (1 << uint(i))
	if (devices.deviceIdMap[deviceId/8] & mask) != 0 {
		return false
	}
	return true
}

func (devices *DeviceSet) lookupDevice(hash string) (*DevInfo, error) {
	devices.devicesLock.Lock()
	defer devices.devicesLock.Unlock()
	info := devices.Devices[hash]
	if info == nil {
		info = devices.loadMetadata(hash)
		if info == nil {
			return nil, fmt.Errorf("Unknown device %s", hash)
		}

		devices.Devices[hash] = info
	}
	return info, nil
}

func (devices *DeviceSet) deviceFileWalkFunction(path string, finfo os.FileInfo) error {

	// Skip some of the meta files which are not device files.
	if strings.HasSuffix(finfo.Name(), ".migrated") {
		logrus.Debugf("Skipping file %s", path)
		return nil
	}

	if strings.HasPrefix(finfo.Name(), ".") {
		logrus.Debugf("Skipping file %s", path)
		return nil
	}

	if finfo.Name() == deviceSetMetaFile {
		logrus.Debugf("Skipping file %s", path)
		return nil
	}

	logrus.Debugf("Loading data for file %s", path)

	hash := finfo.Name()
	if hash == "base" {
		hash = ""
	}

	dinfo := devices.loadMetadata(hash)
	if dinfo == nil {
		return fmt.Errorf("Error loading device metadata file %s", hash)
	}

	if dinfo.DeviceId > MaxDeviceId {
		logrus.Errorf("Ignoring Invalid DeviceId=%d", dinfo.DeviceId)
		return nil
	}

	devices.Lock()
	devices.markDeviceIdUsed(dinfo.DeviceId)
	devices.Unlock()

	logrus.Debugf("Added deviceId=%d to DeviceIdMap", dinfo.DeviceId)
	return nil
}

func (devices *DeviceSet) constructDeviceIdMap() error {
	logrus.Debugf("[deviceset] constructDeviceIdMap()")
	defer logrus.Debugf("[deviceset] constructDeviceIdMap() END")

	var scan = func(path string, info os.FileInfo, err error) error {
		if err != nil {
			logrus.Debugf("Can't walk the file %s", path)
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

func (devices *DeviceSet) unregisterDevice(id int, hash string) error {
	logrus.Debugf("unregisterDevice(%v, %v)", id, hash)
	info := &DevInfo{
		Hash:     hash,
		DeviceId: id,
	}

	devices.devicesLock.Lock()
	delete(devices.Devices, hash)
	devices.devicesLock.Unlock()

	if err := devices.removeMetadata(info); err != nil {
		logrus.Debugf("Error removing metadata: %s", err)
		return err
	}

	return nil
}

func (devices *DeviceSet) registerDevice(id int, hash string, size uint64, transactionId uint64) (*DevInfo, error) {
	logrus.Debugf("registerDevice(%v, %v)", id, hash)
	info := &DevInfo{
		Hash:          hash,
		DeviceId:      id,
		Size:          size,
		TransactionId: transactionId,
		Initialized:   false,
		devices:       devices,
	}

	devices.devicesLock.Lock()
	devices.Devices[hash] = info
	devices.devicesLock.Unlock()

	if err := devices.saveMetadata(info); err != nil {
		// Try to remove unused device
		devices.devicesLock.Lock()
		delete(devices.Devices, hash)
		devices.devicesLock.Unlock()
		return nil, err
	}

	return info, nil
}

func (devices *DeviceSet) activateDeviceIfNeeded(info *DevInfo) error {
	logrus.Debugf("activateDeviceIfNeeded(%v)", info.Hash)

	// Make sure deferred removal on device is canceled, if one was
	// scheduled.
	if err := devices.cancelDeferredRemoval(info); err != nil {
		return fmt.Errorf("Deivce Deferred Removal Cancellation Failed: %s", err)
	}

	if devinfo, _ := devicemapper.GetInfo(info.Name()); devinfo != nil && devinfo.Exists != 0 {
		return nil
	}

	return devicemapper.ActivateDevice(devices.getPoolDevName(), info.Name(), info.DeviceId, info.Size)
}

func (devices *DeviceSet) createFilesystem(info *DevInfo) error {
	devname := info.DevName()

	args := []string{}
	for _, arg := range devices.mkfsArgs {
		args = append(args, arg)
	}

	args = append(args, devname)

	var err error
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
		err = fmt.Errorf("Unsupported filesystem type %s", devices.filesystem)
	}
	if err != nil {
		return err
	}

	return nil
}

func (devices *DeviceSet) migrateOldMetaData() error {
	// Migrate old metadata file
	jsonData, err := ioutil.ReadFile(devices.oldMetadataFile())
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	if jsonData != nil {
		m := MetaData{Devices: make(map[string]*DevInfo)}

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

func (devices *DeviceSet) initMetaData() error {
	if err := devices.migrateOldMetaData(); err != nil {
		return err
	}

	_, transactionId, _, _, _, _, err := devices.poolStatus()
	if err != nil {
		return err
	}

	devices.TransactionId = transactionId

	if err := devices.constructDeviceIdMap(); err != nil {
		return err
	}

	if err := devices.processPendingTransaction(); err != nil {
		return err
	}
	return nil
}

func (devices *DeviceSet) incNextDeviceId() {
	// Ids are 24bit, so wrap around
	devices.NextDeviceId = (devices.NextDeviceId + 1) & MaxDeviceId
}

func (devices *DeviceSet) getNextFreeDeviceId() (int, error) {
	devices.incNextDeviceId()
	for i := 0; i <= MaxDeviceId; i++ {
		if devices.isDeviceIdFree(devices.NextDeviceId) {
			devices.markDeviceIdUsed(devices.NextDeviceId)
			return devices.NextDeviceId, nil
		}
		devices.incNextDeviceId()
	}

	return 0, fmt.Errorf("Unable to find a free device Id")
}

func (devices *DeviceSet) createRegisterDevice(hash string) (*DevInfo, error) {
	deviceId, err := devices.getNextFreeDeviceId()
	if err != nil {
		return nil, err
	}

	if err := devices.openTransaction(hash, deviceId); err != nil {
		logrus.Debugf("Error opening transaction hash = %s deviceId = %d", hash, deviceId)
		devices.markDeviceIdFree(deviceId)
		return nil, err
	}

	for {
		if err := devicemapper.CreateDevice(devices.getPoolDevName(), deviceId); err != nil {
			if devicemapper.DeviceIdExists(err) {
				// Device Id already exists. This should not
				// happen. Now we have a mechianism to find
				// a free device Id. So something is not right.
				// Give a warning and continue.
				logrus.Errorf("Device Id %d exists in pool but it is supposed to be unused", deviceId)
				deviceId, err = devices.getNextFreeDeviceId()
				if err != nil {
					return nil, err
				}
				// Save new device id into transaction
				devices.refreshTransaction(deviceId)
				continue
			}
			logrus.Debugf("Error creating device: %s", err)
			devices.markDeviceIdFree(deviceId)
			return nil, err
		}
		break
	}

	logrus.Debugf("Registering device (id %v) with FS size %v", deviceId, devices.baseFsSize)
	info, err := devices.registerDevice(deviceId, hash, devices.baseFsSize, devices.OpenTransactionId)
	if err != nil {
		_ = devicemapper.DeleteDevice(devices.getPoolDevName(), deviceId)
		devices.markDeviceIdFree(deviceId)
		return nil, err
	}

	if err := devices.closeTransaction(); err != nil {
		devices.unregisterDevice(deviceId, hash)
		devicemapper.DeleteDevice(devices.getPoolDevName(), deviceId)
		devices.markDeviceIdFree(deviceId)
		return nil, err
	}
	return info, nil
}

func (devices *DeviceSet) createRegisterSnapDevice(hash string, baseInfo *DevInfo) error {
	deviceId, err := devices.getNextFreeDeviceId()
	if err != nil {
		return err
	}

	if err := devices.openTransaction(hash, deviceId); err != nil {
		logrus.Debugf("Error opening transaction hash = %s deviceId = %d", hash, deviceId)
		devices.markDeviceIdFree(deviceId)
		return err
	}

	for {
		if err := devicemapper.CreateSnapDevice(devices.getPoolDevName(), deviceId, baseInfo.Name(), baseInfo.DeviceId); err != nil {
			if devicemapper.DeviceIdExists(err) {
				// Device Id already exists. This should not
				// happen. Now we have a mechianism to find
				// a free device Id. So something is not right.
				// Give a warning and continue.
				logrus.Errorf("Device Id %d exists in pool but it is supposed to be unused", deviceId)
				deviceId, err = devices.getNextFreeDeviceId()
				if err != nil {
					return err
				}
				// Save new device id into transaction
				devices.refreshTransaction(deviceId)
				continue
			}
			logrus.Debugf("Error creating snap device: %s", err)
			devices.markDeviceIdFree(deviceId)
			return err
		}
		break
	}

	if _, err := devices.registerDevice(deviceId, hash, baseInfo.Size, devices.OpenTransactionId); err != nil {
		devicemapper.DeleteDevice(devices.getPoolDevName(), deviceId)
		devices.markDeviceIdFree(deviceId)
		logrus.Debugf("Error registering device: %s", err)
		return err
	}

	if err := devices.closeTransaction(); err != nil {
		devices.unregisterDevice(deviceId, hash)
		devicemapper.DeleteDevice(devices.getPoolDevName(), deviceId)
		devices.markDeviceIdFree(deviceId)
		return err
	}
	return nil
}

func (devices *DeviceSet) loadMetadata(hash string) *DevInfo {
	info := &DevInfo{Hash: hash, devices: devices}

	jsonData, err := ioutil.ReadFile(devices.metadataFile(info))
	if err != nil {
		return nil
	}

	if err := json.Unmarshal(jsonData, &info); err != nil {
		return nil
	}

	return info
}

func getDeviceUUID(device string) (string, error) {
	out, err := exec.Command("blkid", "-s", "UUID", "-o", "value", device).Output()
	if err != nil {
		logrus.Debugf("Failed to find uuid for device %s:%v", device, err)
		return "", err
	}

	uuid := strings.TrimSuffix(string(out), "\n")
	uuid = strings.TrimSpace(uuid)
	logrus.Debugf("UUID for device: %s is:%s", device, uuid)
	return uuid, nil
}

func (devices *DeviceSet) verifyBaseDeviceUUID(baseInfo *DevInfo) error {
	devices.Lock()
	defer devices.Unlock()

	if err := devices.activateDeviceIfNeeded(baseInfo); err != nil {
		return err
	}

	defer devices.deactivateDevice(baseInfo)

	uuid, err := getDeviceUUID(baseInfo.DevName())
	if err != nil {
		return err
	}

	if devices.BaseDeviceUUID != uuid {
		return fmt.Errorf("Current Base Device UUID:%s does not match with stored UUID:%s", uuid, devices.BaseDeviceUUID)
	}

	return nil
}

func (devices *DeviceSet) saveBaseDeviceUUID(baseInfo *DevInfo) error {
	devices.Lock()
	defer devices.Unlock()

	if err := devices.activateDeviceIfNeeded(baseInfo); err != nil {
		return err
	}

	defer devices.deactivateDevice(baseInfo)

	uuid, err := getDeviceUUID(baseInfo.DevName())
	if err != nil {
		return err
	}

	devices.BaseDeviceUUID = uuid
	devices.saveDeviceSetMetaData()
	return nil
}

func (devices *DeviceSet) setupBaseImage() error {
	oldInfo, _ := devices.lookupDevice("")
	if oldInfo != nil && oldInfo.Initialized {
		// If BaseDeviceUUID is nil (upgrade case), save it and
		// return success.
		if devices.BaseDeviceUUID == "" {
			if err := devices.saveBaseDeviceUUID(oldInfo); err != nil {
				return fmt.Errorf("Could not query and save base device UUID:%v", err)
			}
			return nil
		}

		if err := devices.verifyBaseDeviceUUID(oldInfo); err != nil {
			return fmt.Errorf("Base Device UUID verification failed. Possibly using a different thin pool then last invocation:%v", err)
		}
		return nil
	}

	if oldInfo != nil && !oldInfo.Initialized {
		logrus.Debugf("Removing uninitialized base image")
		if err := devices.DeleteDevice(""); err != nil {
			return err
		}
	}

	if devices.thinPoolDevice != "" && oldInfo == nil {
		_, transactionId, dataUsed, _, _, _, err := devices.poolStatus()
		if err != nil {
			return err
		}
		if dataUsed != 0 {
			return fmt.Errorf("Unable to take ownership of thin-pool (%s) that already has used data blocks",
				devices.thinPoolDevice)
		}
		if transactionId != 0 {
			return fmt.Errorf("Unable to take ownership of thin-pool (%s) with non-zero transaction Id",
				devices.thinPoolDevice)
		}
	}

	logrus.Debugf("Initializing base device-mapper thin volume")

	// Create initial device
	info, err := devices.createRegisterDevice("")
	if err != nil {
		return err
	}

	logrus.Debugf("Creating filesystem on base device-mapper thin volume")

	if err := devices.activateDeviceIfNeeded(info); err != nil {
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
		return fmt.Errorf("Could not query and save base device UUID:%v", err)
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
					syscall.CloseOnExec(fd)
				}
			}
		}
	}
}

func (devices *DeviceSet) DMLog(level int, file string, line int, dmError int, message string) {
	// By default libdm sends us all the messages including debug ones.
	// We need to filter out messages here and figure out which one
	// should be printed.
	if level > DMLogLevel {
		return
	}

	// FIXME(vbatts) push this back into ./pkg/devicemapper/
	if level <= devicemapper.LogLevelErr {
		logrus.Errorf("libdevmapper(%d): %s:%d (%d) %s", level, file, line, dmError, message)
	} else if level <= devicemapper.LogLevelInfo {
		logrus.Infof("libdevmapper(%d): %s:%d (%d) %s", level, file, line, dmError, message)
	} else {
		// FIXME(vbatts) push this back into ./pkg/devicemapper/
		logrus.Debugf("libdevmapper(%d): %s:%d (%d) %s", level, file, line, dmError, message)
	}
}

func major(device uint64) uint64 {
	return (device >> 8) & 0xfff
}

func minor(device uint64) uint64 {
	return (device & 0xff) | ((device >> 12) & 0xfff00)
}

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
		return fmt.Errorf("Can't shrink file")
	}

	dataloopback := devicemapper.FindLoopDeviceFor(datafile)
	if dataloopback == nil {
		return fmt.Errorf("Unable to find loopback mount for: %s", datafilename)
	}
	defer dataloopback.Close()

	metadatafile, err := os.OpenFile(metadatafilename, os.O_RDWR, 0)
	if metadatafile == nil {
		return err
	}
	defer metadatafile.Close()

	metadataloopback := devicemapper.FindLoopDeviceFor(metadatafile)
	if metadataloopback == nil {
		return fmt.Errorf("Unable to find loopback mount for: %s", metadatafilename)
	}
	defer metadataloopback.Close()

	// Grow loopback file
	if err := datafile.Truncate(size); err != nil {
		return fmt.Errorf("Unable to grow loopback file: %s", err)
	}

	// Reload size for loopback device
	if err := devicemapper.LoopbackSetCapacity(dataloopback); err != nil {
		return fmt.Errorf("Unable to update loopback capacity: %s", err)
	}

	// Suspend the pool
	if err := devicemapper.SuspendDevice(devices.getPoolName()); err != nil {
		return fmt.Errorf("Unable to suspend pool: %s", err)
	}

	// Reload with the new block sizes
	if err := devicemapper.ReloadPool(devices.getPoolName(), dataloopback, metadataloopback, devices.thinpBlockSize); err != nil {
		return fmt.Errorf("Unable to reload pool: %s", err)
	}

	// Resume the pool
	if err := devicemapper.ResumeDevice(devices.getPoolName()); err != nil {
		return fmt.Errorf("Unable to resume pool: %s", err)
	}

	return nil
}

func (devices *DeviceSet) loadTransactionMetaData() error {
	jsonData, err := ioutil.ReadFile(devices.transactionMetaFile())
	if err != nil {
		// There is no active transaction. This will be the case
		// during upgrade.
		if os.IsNotExist(err) {
			devices.OpenTransactionId = devices.TransactionId
			return nil
		}
		return err
	}

	json.Unmarshal(jsonData, &devices.Transaction)
	return nil
}

func (devices *DeviceSet) saveTransactionMetaData() error {
	jsonData, err := json.Marshal(&devices.Transaction)
	if err != nil {
		return fmt.Errorf("Error encoding metadata to json: %s", err)
	}

	return devices.writeMetaFile(jsonData, devices.transactionMetaFile())
}

func (devices *DeviceSet) removeTransactionMetaData() error {
	if err := os.RemoveAll(devices.transactionMetaFile()); err != nil {
		return err
	}
	return nil
}

func (devices *DeviceSet) rollbackTransaction() error {
	logrus.Debugf("Rolling back open transaction: TransactionId=%d hash=%s device_id=%d", devices.OpenTransactionId, devices.DeviceIdHash, devices.DeviceId)

	// A device id might have already been deleted before transaction
	// closed. In that case this call will fail. Just leave a message
	// in case of failure.
	if err := devicemapper.DeleteDevice(devices.getPoolDevName(), devices.DeviceId); err != nil {
		logrus.Errorf("Unable to delete device: %s", err)
	}

	dinfo := &DevInfo{Hash: devices.DeviceIdHash}
	if err := devices.removeMetadata(dinfo); err != nil {
		logrus.Errorf("Unable to remove metadata: %s", err)
	} else {
		devices.markDeviceIdFree(devices.DeviceId)
	}

	if err := devices.removeTransactionMetaData(); err != nil {
		logrus.Errorf("Unable to remove transaction meta file %s: %s", devices.transactionMetaFile(), err)
	}

	return nil
}

func (devices *DeviceSet) processPendingTransaction() error {
	if err := devices.loadTransactionMetaData(); err != nil {
		return err
	}

	// If there was open transaction but pool transaction Id is same
	// as open transaction Id, nothing to roll back.
	if devices.TransactionId == devices.OpenTransactionId {
		return nil
	}

	// If open transaction Id is less than pool transaction Id, something
	// is wrong. Bail out.
	if devices.OpenTransactionId < devices.TransactionId {
		logrus.Errorf("Open Transaction id %d is less than pool transaction id %d", devices.OpenTransactionId, devices.TransactionId)
		return nil
	}

	// Pool transaction Id is not same as open transaction. There is
	// a transaction which was not completed.
	if err := devices.rollbackTransaction(); err != nil {
		return fmt.Errorf("Rolling back open transaction failed: %s", err)
	}

	devices.OpenTransactionId = devices.TransactionId
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
		return fmt.Errorf("Error encoding metadata to json: %s", err)
	}

	return devices.writeMetaFile(jsonData, devices.deviceSetMetaFile())
}

func (devices *DeviceSet) openTransaction(hash string, DeviceId int) error {
	devices.allocateTransactionId()
	devices.DeviceIdHash = hash
	devices.DeviceId = DeviceId
	if err := devices.saveTransactionMetaData(); err != nil {
		return fmt.Errorf("Error saving transaction metadata: %s", err)
	}
	return nil
}

func (devices *DeviceSet) refreshTransaction(DeviceId int) error {
	devices.DeviceId = DeviceId
	if err := devices.saveTransactionMetaData(); err != nil {
		return fmt.Errorf("Error saving transaction metadata: %s", err)
	}
	return nil
}

func (devices *DeviceSet) closeTransaction() error {
	if err := devices.updatePoolTransactionId(); err != nil {
		logrus.Debugf("Failed to close Transaction")
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
		DriverDeferredRemovalSupport = true
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
		DriverDeferredRemovalSupport = true
		return nil
	}

	return nil
}

// Determine the major and minor number of loopback device
func getDeviceMajorMinor(file *os.File) (uint64, uint64, error) {
	stat, err := file.Stat()
	if err != nil {
		return 0, 0, err
	}

	dev := stat.Sys().(*syscall.Stat_t).Rdev
	majorNum := major(dev)
	minorNum := minor(dev)

	logrus.Debugf("[devmapper]: Major:Minor for device: %s is:%v:%v", file.Name(), majorNum, minorNum)
	return majorNum, minorNum, nil
}

// Given a file which is backing file of a loop back device, find the
// loopback device name and its major/minor number.
func getLoopFileDeviceMajMin(filename string) (string, uint64, uint64, error) {
	file, err := os.Open(filename)
	if err != nil {
		logrus.Debugf("[devmapper]: Failed to open file %s", filename)
		return "", 0, 0, err
	}

	defer file.Close()
	loopbackDevice := devicemapper.FindLoopDeviceFor(file)
	if loopbackDevice == nil {
		return "", 0, 0, fmt.Errorf("[devmapper]: Unable to find loopback mount for: %s", filename)
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

	logrus.Debugf("[devmapper]: poolDataMajMin=%s poolMetaMajMin=%s\n", poolDataMajMin, poolMetadataMajMin)

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

func (devices *DeviceSet) initDevmapper(doInit bool) error {
	// give ourselves to libdm as a log handler
	devicemapper.LogInit(devices)

	version, err := devicemapper.GetDriverVersion()
	if err != nil {
		// Can't even get driver version, assume not supported
		return graphdriver.ErrNotSupported
	}

	if err := determineDriverCapabilities(version); err != nil {
		return graphdriver.ErrNotSupported
	}

	// If user asked for deferred removal and both library and driver
	// supports deferred removal use it.
	if EnableDeferredRemoval && DriverDeferredRemovalSupport && devicemapper.LibraryDeferredRemovalSupport == true {
		logrus.Debugf("devmapper: Deferred removal support enabled.")
		devices.deferredRemove = true
	}

	// https://github.com/docker/docker/issues/4036
	if supported := devicemapper.UdevSetSyncSupport(true); !supported {
		logrus.Warn("Udev sync is not supported. This will lead to unexpected behavior, data loss and errors. For more information, see https://docs.docker.com/reference/commandline/cli/#daemon-storage-driver-option")
	}

	if err := os.MkdirAll(devices.metadataDir(), 0700); err != nil && !os.IsExist(err) {
		return err
	}

	// Set the device prefix from the device id and inode of the docker root dir

	st, err := os.Stat(devices.root)
	if err != nil {
		return fmt.Errorf("Error looking up dir %s: %s", devices.root, err)
	}
	sysSt := st.Sys().(*syscall.Stat_t)
	// "reg-" stands for "regular file".
	// In the future we might use "dev-" for "device file", etc.
	// docker-maj,min[-inode] stands for:
	//	- Managed by docker
	//	- The target of this device is at major <maj> and minor <min>
	//	- If <inode> is defined, use that file inside the device as a loopback image. Otherwise use the device itself.
	devices.devicePrefix = fmt.Sprintf("docker-%d:%d-%d", major(sysSt.Dev), minor(sysSt.Dev), sysSt.Ino)
	logrus.Debugf("Generated prefix: %s", devices.devicePrefix)

	// Check for the existence of the thin-pool device
	logrus.Debugf("Checking for existence of the pool '%s'", devices.getPoolName())
	info, err := devicemapper.GetInfo(devices.getPoolName())
	if info == nil {
		logrus.Debugf("Error device devicemapper.GetInfo: %s", err)
		return err
	}

	// It seems libdevmapper opens this without O_CLOEXEC, and go exec will not close files
	// that are not Close-on-exec, and lxc-start will die if it inherits any unexpected files,
	// so we add this badhack to make sure it closes itself
	setCloseOnExec("/dev/mapper/control")

	// Make sure the sparse images exist in <root>/devicemapper/data and
	// <root>/devicemapper/metadata

	createdLoopback := false

	// If the pool doesn't exist, create it
	if info.Exists == 0 && devices.thinPoolDevice == "" {
		logrus.Debugf("Pool doesn't exist. Creating it.")

		var (
			dataFile     *os.File
			metadataFile *os.File
		)

		if devices.dataDevice == "" {
			// Make sure the sparse images exist in <root>/devicemapper/data

			hasData := devices.hasImage("data")

			if !doInit && !hasData {
				return errors.New("Loopback data file not found")
			}

			if !hasData {
				createdLoopback = true
			}

			data, err := devices.ensureImage("data", devices.dataLoopbackSize)
			if err != nil {
				logrus.Debugf("Error device ensureImage (data): %s", err)
				return err
			}

			dataFile, err = devicemapper.AttachLoopDevice(data)
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
				return errors.New("Loopback metadata file not found")
			}

			if !hasMetadata {
				createdLoopback = true
			}

			metadata, err := devices.ensureImage("metadata", devices.metaDataLoopbackSize)
			if err != nil {
				logrus.Debugf("Error device ensureImage (metadata): %s", err)
				return err
			}

			metadataFile, err = devicemapper.AttachLoopDevice(metadata)
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
	}

	// Pool already exists and caller did not pass us a pool. That means
	// we probably created pool earlier and could not remove it as some
	// containers were still using it. Detect some of the properties of
	// pool, like is it using loop devices.
	if info.Exists != 0 && devices.thinPoolDevice == "" {
		if err := devices.loadThinPoolLoopBackInfo(); err != nil {
			logrus.Debugf("Failed to load thin pool loopback device information:%v", err)
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

	// Right now this loads only NextDeviceId. If there is more metadata
	// down the line, we might have to move it earlier.
	if err := devices.loadDeviceSetMetaData(); err != nil {
		return err
	}

	// Setup the base image
	if doInit {
		if err := devices.setupBaseImage(); err != nil {
			logrus.Debugf("Error device setupBaseImage: %s", err)
			return err
		}
	}

	return nil
}

func (devices *DeviceSet) AddDevice(hash, baseHash string) error {
	logrus.Debugf("[deviceset] AddDevice(hash=%s basehash=%s)", hash, baseHash)
	defer logrus.Debugf("[deviceset] AddDevice(hash=%s basehash=%s) END", hash, baseHash)

	baseInfo, err := devices.lookupDevice(baseHash)
	if err != nil {
		return err
	}

	baseInfo.lock.Lock()
	defer baseInfo.lock.Unlock()

	devices.Lock()
	defer devices.Unlock()

	if info, _ := devices.lookupDevice(hash); info != nil {
		return fmt.Errorf("device %s already exists", hash)
	}

	if err := devices.createRegisterSnapDevice(hash, baseInfo); err != nil {
		return err
	}

	return nil
}

func (devices *DeviceSet) deleteDevice(info *DevInfo) error {
	if devices.doBlkDiscard {
		// This is a workaround for the kernel not discarding block so
		// on the thin pool when we remove a thinp device, so we do it
		// manually
		if err := devices.activateDeviceIfNeeded(info); err == nil {
			if err := devicemapper.BlockDeviceDiscard(info.DevName()); err != nil {
				logrus.Debugf("Error discarding block on device: %s (ignoring)", err)
			}
		}
	}

	devinfo, _ := devicemapper.GetInfo(info.Name())
	if devinfo != nil && devinfo.Exists != 0 {
		if err := devices.removeDevice(info.Name()); err != nil {
			logrus.Debugf("Error removing device: %s", err)
			return err
		}
	}

	if err := devices.openTransaction(info.Hash, info.DeviceId); err != nil {
		logrus.Debugf("Error opening transaction hash = %s deviceId = %d", "", info.DeviceId)
		return err
	}

	if err := devicemapper.DeleteDevice(devices.getPoolDevName(), info.DeviceId); err != nil {
		logrus.Debugf("Error deleting device: %s", err)
		return err
	}

	if err := devices.unregisterDevice(info.DeviceId, info.Hash); err != nil {
		return err
	}

	if err := devices.closeTransaction(); err != nil {
		return err
	}

	devices.markDeviceIdFree(info.DeviceId)

	return nil
}

func (devices *DeviceSet) DeleteDevice(hash string) error {
	info, err := devices.lookupDevice(hash)
	if err != nil {
		return err
	}

	info.lock.Lock()
	defer info.lock.Unlock()

	devices.Lock()
	defer devices.Unlock()

	return devices.deleteDevice(info)
}

func (devices *DeviceSet) deactivatePool() error {
	logrus.Debugf("[devmapper] deactivatePool()")
	defer logrus.Debugf("[devmapper] deactivatePool END")
	devname := devices.getPoolDevName()

	devinfo, err := devicemapper.GetInfo(devname)
	if err != nil {
		return err
	}
	if d, err := devicemapper.GetDeps(devname); err == nil {
		// Access to more Debug output
		logrus.Debugf("[devmapper] devicemapper.GetDeps() %s: %#v", devname, d)
	}
	if devinfo.Exists != 0 {
		return devicemapper.RemoveDevice(devname)
	}

	return nil
}

func (devices *DeviceSet) deactivateDevice(info *DevInfo) error {
	logrus.Debugf("[devmapper] deactivateDevice(%s)", info.Hash)
	defer logrus.Debugf("[devmapper] deactivateDevice END(%s)", info.Hash)

	devinfo, err := devicemapper.GetInfo(info.Name())
	if err != nil {
		return err
	}

	if devinfo.Exists == 0 {
		return nil
	}

	if devices.deferredRemove {
		if err := devicemapper.RemoveDeviceDeferred(info.Name()); err != nil {
			return err
		}
	} else {
		if err := devices.removeDevice(info.Name()); err != nil {
			return err
		}
	}
	return nil
}

// Issues the underlying dm remove operation.
func (devices *DeviceSet) removeDevice(devname string) error {
	var err error

	logrus.Debugf("[devmapper] removeDevice START(%s)", devname)
	defer logrus.Debugf("[devmapper] removeDevice END(%s)", devname)

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

func (devices *DeviceSet) cancelDeferredRemoval(info *DevInfo) error {
	if !devices.deferredRemove {
		return nil
	}

	logrus.Debugf("[devmapper] cancelDeferredRemoval START(%s)", info.Name())
	defer logrus.Debugf("[devmapper] cancelDeferredRemoval END(%s)", info.Name())

	devinfo, err := devicemapper.GetInfoWithDeferred(info.Name())

	if devinfo != nil && devinfo.DeferredRemove == 0 {
		return nil
	}

	// Cancel deferred remove
	for i := 0; i < 100; i++ {
		err = devicemapper.CancelDeferredRemove(info.Name())
		if err == nil {
			break
		}

		if err == devicemapper.ErrEnxio {
			// Device is probably already gone. Return success.
			return nil
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

func (devices *DeviceSet) Shutdown() error {
	logrus.Debugf("[deviceset %s] Shutdown()", devices.devicePrefix)
	logrus.Debugf("[devmapper] Shutting down DeviceSet: %s", devices.root)
	defer logrus.Debugf("[deviceset %s] Shutdown() END", devices.devicePrefix)

	var devs []*DevInfo

	devices.devicesLock.Lock()
	for _, info := range devices.Devices {
		devs = append(devs, info)
	}
	devices.devicesLock.Unlock()

	for _, info := range devs {
		info.lock.Lock()
		if info.mountCount > 0 {
			// We use MNT_DETACH here in case it is still busy in some running
			// container. This means it'll go away from the global scope directly,
			// and the device will be released when that container dies.
			if err := syscall.Unmount(info.mountPath, syscall.MNT_DETACH); err != nil {
				logrus.Debugf("Shutdown unmounting %s, error: %s", info.mountPath, err)
			}

			devices.Lock()
			if err := devices.deactivateDevice(info); err != nil {
				logrus.Debugf("Shutdown deactivate %s , error: %s", info.Hash, err)
			}
			devices.Unlock()
		}
		info.lock.Unlock()
	}

	info, _ := devices.lookupDevice("")
	if info != nil {
		info.lock.Lock()
		devices.Lock()
		if err := devices.deactivateDevice(info); err != nil {
			logrus.Debugf("Shutdown deactivate base , error: %s", err)
		}
		devices.Unlock()
		info.lock.Unlock()
	}

	devices.Lock()
	if devices.thinPoolDevice == "" {
		if err := devices.deactivatePool(); err != nil {
			logrus.Debugf("Shutdown deactivate pool , error: %s", err)
		}
	}

	devices.saveDeviceSetMetaData()
	devices.Unlock()

	return nil
}

func (devices *DeviceSet) MountDevice(hash, path, mountLabel string) error {
	info, err := devices.lookupDevice(hash)
	if err != nil {
		return err
	}

	info.lock.Lock()
	defer info.lock.Unlock()

	devices.Lock()
	defer devices.Unlock()

	if info.mountCount > 0 {
		if path != info.mountPath {
			return fmt.Errorf("Trying to mount devmapper device in multiple places (%s, %s)", info.mountPath, path)
		}

		info.mountCount++
		return nil
	}

	if err := devices.activateDeviceIfNeeded(info); err != nil {
		return fmt.Errorf("Error activating devmapper device for '%s': %s", hash, err)
	}

	var flags uintptr = syscall.MS_MGC_VAL

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

	if err := syscall.Mount(info.DevName(), path, fstype, flags, options); err != nil {
		return fmt.Errorf("Error mounting '%s' on '%s': %s", info.DevName(), path, err)
	}

	info.mountCount = 1
	info.mountPath = path

	return nil
}

func (devices *DeviceSet) UnmountDevice(hash string) error {
	logrus.Debugf("[devmapper] UnmountDevice(hash=%s)", hash)
	defer logrus.Debugf("[devmapper] UnmountDevice(hash=%s) END", hash)

	info, err := devices.lookupDevice(hash)
	if err != nil {
		return err
	}

	info.lock.Lock()
	defer info.lock.Unlock()

	devices.Lock()
	defer devices.Unlock()

	if info.mountCount == 0 {
		return fmt.Errorf("UnmountDevice: device not-mounted id %s", hash)
	}

	info.mountCount--
	if info.mountCount > 0 {
		return nil
	}

	logrus.Debugf("[devmapper] Unmount(%s)", info.mountPath)
	if err := syscall.Unmount(info.mountPath, syscall.MNT_DETACH); err != nil {
		return err
	}
	logrus.Debugf("[devmapper] Unmount done")

	if err := devices.deactivateDevice(info); err != nil {
		return err
	}

	info.mountPath = ""

	return nil
}

func (devices *DeviceSet) HasDevice(hash string) bool {
	devices.Lock()
	defer devices.Unlock()

	info, _ := devices.lookupDevice(hash)
	return info != nil
}

func (devices *DeviceSet) HasActivatedDevice(hash string) bool {
	info, _ := devices.lookupDevice(hash)
	if info == nil {
		return false
	}

	info.lock.Lock()
	defer info.lock.Unlock()

	devices.Lock()
	defer devices.Unlock()

	devinfo, _ := devicemapper.GetInfo(info.Name())
	return devinfo != nil && devinfo.Exists != 0
}

func (devices *DeviceSet) List() []string {
	devices.Lock()
	defer devices.Unlock()

	devices.devicesLock.Lock()
	ids := make([]string, len(devices.Devices))
	i := 0
	for k := range devices.Devices {
		ids[i] = k
		i++
	}
	devices.devicesLock.Unlock()

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

func (devices *DeviceSet) GetDeviceStatus(hash string) (*DevStatus, error) {
	info, err := devices.lookupDevice(hash)
	if err != nil {
		return nil, err
	}

	info.lock.Lock()
	defer info.lock.Unlock()

	devices.Lock()
	defer devices.Unlock()

	status := &DevStatus{
		DeviceId:      info.DeviceId,
		Size:          info.Size,
		TransactionId: info.TransactionId,
	}

	if err := devices.activateDeviceIfNeeded(info); err != nil {
		return nil, fmt.Errorf("Error activating devmapper device for '%s': %s", hash, err)
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

func (devices *DeviceSet) poolStatus() (totalSizeInSectors, transactionId, dataUsed, dataTotal, metadataUsed, metadataTotal uint64, err error) {
	var params string
	if _, totalSizeInSectors, _, params, err = devicemapper.GetStatus(devices.getPoolName()); err == nil {
		_, err = fmt.Sscanf(params, "%d %d/%d %d/%d", &transactionId, &metadataUsed, &metadataTotal, &dataUsed, &dataTotal)
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
	buf := new(syscall.Statfs_t)
	if err := syscall.Statfs(loopFile, buf); err != nil {
		logrus.Warnf("Couldn't stat loopfile filesystem %v: %v", loopFile, err)
		return 0, err
	}
	return buf.Bfree * uint64(buf.Bsize), nil
}

func (devices *DeviceSet) isRealFile(loopFile string) (bool, error) {
	if loopFile != "" {
		fi, err := os.Stat(loopFile)
		if err != nil {
			logrus.Warnf("Couldn't stat loopfile %v: %v", loopFile, err)
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
	}

	return status
}

// Status returns the current status of this deviceset
func (devices *DeviceSet) ExportDeviceMetadata(hash string) (*DeviceMetadata, error) {
	info, err := devices.lookupDevice(hash)
	if err != nil {
		return nil, err
	}

	info.lock.Lock()
	defer info.lock.Unlock()

	metadata := &DeviceMetadata{info.DeviceId, info.Size, info.Name()}
	return metadata, nil
}

func NewDeviceSet(root string, doInit bool, options []string) (*DeviceSet, error) {
	devicemapper.SetDevDir("/dev")

	devices := &DeviceSet{
		root:                  root,
		MetaData:              MetaData{Devices: make(map[string]*DevInfo)},
		dataLoopbackSize:      DefaultDataLoopbackSize,
		metaDataLoopbackSize:  DefaultMetaDataLoopbackSize,
		baseFsSize:            DefaultBaseFsSize,
		overrideUdevSyncCheck: DefaultUdevSyncOverride,
		filesystem:            "ext4",
		doBlkDiscard:          true,
		thinpBlockSize:        DefaultThinpBlockSize,
		deviceIdMap:           make([]byte, DeviceIdMapSz),
	}

	foundBlkDiscard := false
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
				return nil, fmt.Errorf("Unsupported filesystem %s\n", val)
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
			EnableDeferredRemoval, err = strconv.ParseBool(val)
			if err != nil {
				return nil, err
			}

		default:
			return nil, fmt.Errorf("Unknown option %s\n", key)
		}
	}

	// By default, don't do blk discard hack on raw devices, its rarely useful and is expensive
	if !foundBlkDiscard && (devices.dataDevice != "" || devices.thinPoolDevice != "") {
		devices.doBlkDiscard = false
	}

	if err := devices.initDevmapper(doInit); err != nil {
		return nil, err
	}

	return devices, nil
}
