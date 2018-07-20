// +build linux,cgo

package devicemapper

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"unsafe"

	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

// Same as DM_DEVICE_* enum values from libdevmapper.h
// nolint: deadcode
const (
	deviceCreate TaskType = iota
	deviceReload
	deviceRemove
	deviceRemoveAll
	deviceSuspend
	deviceResume
	deviceInfo
	deviceDeps
	deviceRename
	deviceVersion
	deviceStatus
	deviceTable
	deviceWaitevent
	deviceList
	deviceClear
	deviceMknodes
	deviceListVersions
	deviceTargetMsg
	deviceSetGeometry
)

const (
	addNodeOnResume AddNodeType = iota
	addNodeOnCreate
)

// List of errors returned when using devicemapper.
var (
	ErrTaskRun              = errors.New("dm_task_run failed")
	ErrTaskSetName          = errors.New("dm_task_set_name failed")
	ErrTaskSetMessage       = errors.New("dm_task_set_message failed")
	ErrTaskSetAddNode       = errors.New("dm_task_set_add_node failed")
	ErrTaskSetRo            = errors.New("dm_task_set_ro failed")
	ErrTaskAddTarget        = errors.New("dm_task_add_target failed")
	ErrTaskSetSector        = errors.New("dm_task_set_sector failed")
	ErrTaskGetDeps          = errors.New("dm_task_get_deps failed")
	ErrTaskGetInfo          = errors.New("dm_task_get_info failed")
	ErrTaskGetDriverVersion = errors.New("dm_task_get_driver_version failed")
	ErrTaskDeferredRemove   = errors.New("dm_task_deferred_remove failed")
	ErrTaskSetCookie        = errors.New("dm_task_set_cookie failed")
	ErrNilCookie            = errors.New("cookie ptr can't be nil")
	ErrGetBlockSize         = errors.New("Can't get block size")
	ErrUdevWait             = errors.New("wait on udev cookie failed")
	ErrSetDevDir            = errors.New("dm_set_dev_dir failed")
	ErrGetLibraryVersion    = errors.New("dm_get_library_version failed")
	ErrCreateRemoveTask     = errors.New("Can't create task of type deviceRemove")
	ErrRunRemoveDevice      = errors.New("running RemoveDevice failed")
	ErrInvalidAddNode       = errors.New("Invalid AddNode type")
	ErrBusy                 = errors.New("Device is Busy")
	ErrDeviceIDExists       = errors.New("Device Id Exists")
	ErrEnxio                = errors.New("No such device or address")
)

var (
	dmSawBusy  bool
	dmSawExist bool
	dmSawEnxio bool // No Such Device or Address
)

type (
	// Task represents a devicemapper task (like lvcreate, etc.) ; a task is needed for each ioctl
	// command to execute.
	Task struct {
		unmanaged *cdmTask
	}
	// Deps represents dependents (layer) of a device.
	Deps struct {
		Count  uint32
		Filler uint32
		Device []uint64
	}
	// Info represents information about a device.
	Info struct {
		Exists         int
		Suspended      int
		LiveTable      int
		InactiveTable  int
		OpenCount      int32
		EventNr        uint32
		Major          uint32
		Minor          uint32
		ReadOnly       int
		TargetCount    int32
		DeferredRemove int
	}
	// TaskType represents a type of task
	TaskType int
	// AddNodeType represents a type of node to be added
	AddNodeType int
)

// DeviceIDExists returns whether error conveys the information about device Id already
// exist or not. This will be true if device creation or snap creation
// operation fails if device or snap device already exists in pool.
// Current implementation is little crude as it scans the error string
// for exact pattern match. Replacing it with more robust implementation
// is desirable.
func DeviceIDExists(err error) bool {
	return fmt.Sprint(err) == fmt.Sprint(ErrDeviceIDExists)
}

func (t *Task) destroy() {
	if t != nil {
		DmTaskDestroy(t.unmanaged)
		runtime.SetFinalizer(t, nil)
	}
}

// TaskCreateNamed is a convenience function for TaskCreate when a name
// will be set on the task as well
func TaskCreateNamed(t TaskType, name string) (*Task, error) {
	task := TaskCreate(t)
	if task == nil {
		return nil, fmt.Errorf("devicemapper: Can't create task of type %d", int(t))
	}
	if err := task.setName(name); err != nil {
		return nil, fmt.Errorf("devicemapper: Can't set task name %s", name)
	}
	return task, nil
}

// TaskCreate initializes a devicemapper task of tasktype
func TaskCreate(tasktype TaskType) *Task {
	Ctask := DmTaskCreate(int(tasktype))
	if Ctask == nil {
		return nil
	}
	task := &Task{unmanaged: Ctask}
	runtime.SetFinalizer(task, (*Task).destroy)
	return task
}

func (t *Task) run() error {
	if res := DmTaskRun(t.unmanaged); res != 1 {
		return ErrTaskRun
	}
	runtime.KeepAlive(t)
	return nil
}

func (t *Task) setName(name string) error {
	if res := DmTaskSetName(t.unmanaged, name); res != 1 {
		return ErrTaskSetName
	}
	return nil
}

func (t *Task) setMessage(message string) error {
	if res := DmTaskSetMessage(t.unmanaged, message); res != 1 {
		return ErrTaskSetMessage
	}
	return nil
}

func (t *Task) setSector(sector uint64) error {
	if res := DmTaskSetSector(t.unmanaged, sector); res != 1 {
		return ErrTaskSetSector
	}
	return nil
}

func (t *Task) setCookie(cookie *uint, flags uint16) error {
	if cookie == nil {
		return ErrNilCookie
	}
	if res := DmTaskSetCookie(t.unmanaged, cookie, flags); res != 1 {
		return ErrTaskSetCookie
	}
	return nil
}

func (t *Task) setAddNode(addNode AddNodeType) error {
	if addNode != addNodeOnResume && addNode != addNodeOnCreate {
		return ErrInvalidAddNode
	}
	if res := DmTaskSetAddNode(t.unmanaged, addNode); res != 1 {
		return ErrTaskSetAddNode
	}
	return nil
}

func (t *Task) setRo() error {
	if res := DmTaskSetRo(t.unmanaged); res != 1 {
		return ErrTaskSetRo
	}
	return nil
}

func (t *Task) addTarget(start, size uint64, ttype, params string) error {
	if res := DmTaskAddTarget(t.unmanaged, start, size,
		ttype, params); res != 1 {
		return ErrTaskAddTarget
	}
	return nil
}

func (t *Task) getDeps() (*Deps, error) {
	var deps *Deps
	if deps = DmTaskGetDeps(t.unmanaged); deps == nil {
		return nil, ErrTaskGetDeps
	}
	return deps, nil
}

func (t *Task) getInfo() (*Info, error) {
	info := &Info{}
	if res := DmTaskGetInfo(t.unmanaged, info); res != 1 {
		return nil, ErrTaskGetInfo
	}
	return info, nil
}

func (t *Task) getInfoWithDeferred() (*Info, error) {
	info := &Info{}
	if res := DmTaskGetInfoWithDeferred(t.unmanaged, info); res != 1 {
		return nil, ErrTaskGetInfo
	}
	return info, nil
}

func (t *Task) getDriverVersion() (string, error) {
	res := DmTaskGetDriverVersion(t.unmanaged)
	if res == "" {
		return "", ErrTaskGetDriverVersion
	}
	return res, nil
}

func (t *Task) getNextTarget(next unsafe.Pointer) (nextPtr unsafe.Pointer, start uint64,
	length uint64, targetType string, params string) {

	return DmGetNextTarget(t.unmanaged, next, &start, &length,
			&targetType, &params),
		start, length, targetType, params
}

// UdevWait waits for any processes that are waiting for udev to complete the specified cookie.
func UdevWait(cookie *uint) error {
	if res := DmUdevWait(*cookie); res != 1 {
		logrus.Debugf("devicemapper: Failed to wait on udev cookie %d, %d", *cookie, res)
		return ErrUdevWait
	}
	return nil
}

// SetDevDir sets the dev folder for the device mapper library (usually /dev).
func SetDevDir(dir string) error {
	if res := DmSetDevDir(dir); res != 1 {
		logrus.Debug("devicemapper: Error dm_set_dev_dir")
		return ErrSetDevDir
	}
	return nil
}

// GetLibraryVersion returns the device mapper library version.
func GetLibraryVersion() (string, error) {
	var version string
	if res := DmGetLibraryVersion(&version); res != 1 {
		return "", ErrGetLibraryVersion
	}
	return version, nil
}

// UdevSyncSupported returns whether device-mapper is able to sync with udev
//
// This is essential otherwise race conditions can arise where both udev and
// device-mapper attempt to create and destroy devices.
func UdevSyncSupported() bool {
	return DmUdevGetSyncSupport() != 0
}

// UdevSetSyncSupport allows setting whether the udev sync should be enabled.
// The return bool indicates the state of whether the sync is enabled.
func UdevSetSyncSupport(enable bool) bool {
	if enable {
		DmUdevSetSyncSupport(1)
	} else {
		DmUdevSetSyncSupport(0)
	}

	return UdevSyncSupported()
}

// CookieSupported returns whether the version of device-mapper supports the
// use of cookie's in the tasks.
// This is largely a lower level call that other functions use.
func CookieSupported() bool {
	return DmCookieSupported() != 0
}

// RemoveDevice is a useful helper for cleaning up a device.
func RemoveDevice(name string) error {
	task, err := TaskCreateNamed(deviceRemove, name)
	if task == nil {
		return err
	}

	cookie := new(uint)
	if err := task.setCookie(cookie, 0); err != nil {
		return fmt.Errorf("devicemapper: Can not set cookie: %s", err)
	}
	defer UdevWait(cookie)

	dmSawBusy = false // reset before the task is run
	dmSawEnxio = false
	if err = task.run(); err != nil {
		if dmSawBusy {
			return ErrBusy
		}
		if dmSawEnxio {
			return ErrEnxio
		}
		return fmt.Errorf("devicemapper: Error running RemoveDevice %s", err)
	}

	return nil
}

// RemoveDeviceDeferred is a useful helper for cleaning up a device, but deferred.
func RemoveDeviceDeferred(name string) error {
	logrus.Debugf("devicemapper: RemoveDeviceDeferred START(%s)", name)
	defer logrus.Debugf("devicemapper: RemoveDeviceDeferred END(%s)", name)
	task, err := TaskCreateNamed(deviceRemove, name)
	if task == nil {
		return err
	}

	if err := DmTaskDeferredRemove(task.unmanaged); err != 1 {
		return ErrTaskDeferredRemove
	}

	// set a task cookie and disable library fallback, or else libdevmapper will
	// disable udev dm rules and delete the symlink under /dev/mapper by itself,
	// even if the removal is deferred by the kernel.
	cookie := new(uint)
	flags := uint16(DmUdevDisableLibraryFallback)
	if err := task.setCookie(cookie, flags); err != nil {
		return fmt.Errorf("devicemapper: Can not set cookie: %s", err)
	}

	// libdevmapper and udev relies on System V semaphore for synchronization,
	// semaphores created in `task.setCookie` will be cleaned up in `UdevWait`.
	// So these two function call must come in pairs, otherwise semaphores will
	// be leaked, and the  limit of number of semaphores defined in `/proc/sys/kernel/sem`
	// will be reached, which will eventually make all following calls to 'task.SetCookie'
	// fail.
	// this call will not wait for the deferred removal's final executing, since no
	// udev event will be generated, and the semaphore's value will not be incremented
	// by udev, what UdevWait is just cleaning up the semaphore.
	defer UdevWait(cookie)

	dmSawEnxio = false
	if err = task.run(); err != nil {
		if dmSawEnxio {
			return ErrEnxio
		}
		return fmt.Errorf("devicemapper: Error running RemoveDeviceDeferred %s", err)
	}

	return nil
}

// CancelDeferredRemove cancels a deferred remove for a device.
func CancelDeferredRemove(deviceName string) error {
	task, err := TaskCreateNamed(deviceTargetMsg, deviceName)
	if task == nil {
		return err
	}

	if err := task.setSector(0); err != nil {
		return fmt.Errorf("devicemapper: Can't set sector %s", err)
	}

	if err := task.setMessage(fmt.Sprintf("@cancel_deferred_remove")); err != nil {
		return fmt.Errorf("devicemapper: Can't set message %s", err)
	}

	dmSawBusy = false
	dmSawEnxio = false
	if err := task.run(); err != nil {
		// A device might be being deleted already
		if dmSawBusy {
			return ErrBusy
		} else if dmSawEnxio {
			return ErrEnxio
		}
		return fmt.Errorf("devicemapper: Error running CancelDeferredRemove %s", err)

	}
	return nil
}

// GetBlockDeviceSize returns the size of a block device identified by the specified file.
func GetBlockDeviceSize(file *os.File) (uint64, error) {
	size, err := ioctlBlkGetSize64(file.Fd())
	if err != nil {
		logrus.Errorf("devicemapper: Error getblockdevicesize: %s", err)
		return 0, ErrGetBlockSize
	}
	return uint64(size), nil
}

// BlockDeviceDiscard runs discard for the given path.
// This is used as a workaround for the kernel not discarding block so
// on the thin pool when we remove a thinp device, so we do it
// manually
func BlockDeviceDiscard(path string) error {
	file, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		return err
	}
	defer file.Close()

	size, err := GetBlockDeviceSize(file)
	if err != nil {
		return err
	}

	if err := ioctlBlkDiscard(file.Fd(), 0, size); err != nil {
		return err
	}

	// Without this sometimes the remove of the device that happens after
	// discard fails with EBUSY.
	unix.Sync()

	return nil
}

// CreatePool is the programmatic example of "dmsetup create".
// It creates a device with the specified poolName, data and metadata file and block size.
func CreatePool(poolName string, dataFile, metadataFile *os.File, poolBlockSize uint32) error {
	task, err := TaskCreateNamed(deviceCreate, poolName)
	if task == nil {
		return err
	}

	size, err := GetBlockDeviceSize(dataFile)
	if err != nil {
		return fmt.Errorf("devicemapper: Can't get data size %s", err)
	}

	params := fmt.Sprintf("%s %s %d 32768 1 skip_block_zeroing", metadataFile.Name(), dataFile.Name(), poolBlockSize)
	if err := task.addTarget(0, size/512, "thin-pool", params); err != nil {
		return fmt.Errorf("devicemapper: Can't add target %s", err)
	}

	cookie := new(uint)
	flags := uint16(DmUdevDisableSubsystemRulesFlag | DmUdevDisableDiskRulesFlag | DmUdevDisableOtherRulesFlag)
	if err := task.setCookie(cookie, flags); err != nil {
		return fmt.Errorf("devicemapper: Can't set cookie %s", err)
	}
	defer UdevWait(cookie)

	if err := task.run(); err != nil {
		return fmt.Errorf("devicemapper: Error running deviceCreate (CreatePool) %s", err)
	}

	return nil
}

// ReloadPool is the programmatic example of "dmsetup reload".
// It reloads the table with the specified poolName, data and metadata file and block size.
func ReloadPool(poolName string, dataFile, metadataFile *os.File, poolBlockSize uint32) error {
	task, err := TaskCreateNamed(deviceReload, poolName)
	if task == nil {
		return err
	}

	size, err := GetBlockDeviceSize(dataFile)
	if err != nil {
		return fmt.Errorf("devicemapper: Can't get data size %s", err)
	}

	params := fmt.Sprintf("%s %s %d 32768 1 skip_block_zeroing", metadataFile.Name(), dataFile.Name(), poolBlockSize)
	if err := task.addTarget(0, size/512, "thin-pool", params); err != nil {
		return fmt.Errorf("devicemapper: Can't add target %s", err)
	}

	if err := task.run(); err != nil {
		return fmt.Errorf("devicemapper: Error running ReloadPool %s", err)
	}

	return nil
}

// GetDeps is the programmatic example of "dmsetup deps".
// It outputs a list of devices referenced by the live table for the specified device.
func GetDeps(name string) (*Deps, error) {
	task, err := TaskCreateNamed(deviceDeps, name)
	if task == nil {
		return nil, err
	}
	if err := task.run(); err != nil {
		return nil, err
	}
	return task.getDeps()
}

// GetInfo is the programmatic example of "dmsetup info".
// It outputs some brief information about the device.
func GetInfo(name string) (*Info, error) {
	task, err := TaskCreateNamed(deviceInfo, name)
	if task == nil {
		return nil, err
	}
	if err := task.run(); err != nil {
		return nil, err
	}
	return task.getInfo()
}

// GetInfoWithDeferred is the programmatic example of "dmsetup info", but deferred.
// It outputs some brief information about the device.
func GetInfoWithDeferred(name string) (*Info, error) {
	task, err := TaskCreateNamed(deviceInfo, name)
	if task == nil {
		return nil, err
	}
	if err := task.run(); err != nil {
		return nil, err
	}
	return task.getInfoWithDeferred()
}

// GetDriverVersion is the programmatic example of "dmsetup version".
// It outputs version information of the driver.
func GetDriverVersion() (string, error) {
	task := TaskCreate(deviceVersion)
	if task == nil {
		return "", fmt.Errorf("devicemapper: Can't create deviceVersion task")
	}
	if err := task.run(); err != nil {
		return "", err
	}
	return task.getDriverVersion()
}

// GetStatus is the programmatic example of "dmsetup status".
// It outputs status information for the specified device name.
func GetStatus(name string) (uint64, uint64, string, string, error) {
	task, err := TaskCreateNamed(deviceStatus, name)
	if task == nil {
		logrus.Debugf("devicemapper: GetStatus() Error TaskCreateNamed: %s", err)
		return 0, 0, "", "", err
	}
	if err := task.run(); err != nil {
		logrus.Debugf("devicemapper: GetStatus() Error Run: %s", err)
		return 0, 0, "", "", err
	}

	devinfo, err := task.getInfo()
	if err != nil {
		logrus.Debugf("devicemapper: GetStatus() Error GetInfo: %s", err)
		return 0, 0, "", "", err
	}
	if devinfo.Exists == 0 {
		logrus.Debugf("devicemapper: GetStatus() Non existing device %s", name)
		return 0, 0, "", "", fmt.Errorf("devicemapper: Non existing device %s", name)
	}

	_, start, length, targetType, params := task.getNextTarget(unsafe.Pointer(nil))
	return start, length, targetType, params, nil
}

// GetTable is the programmatic example for "dmsetup table".
// It outputs the current table for the specified device name.
func GetTable(name string) (uint64, uint64, string, string, error) {
	task, err := TaskCreateNamed(deviceTable, name)
	if task == nil {
		logrus.Debugf("devicemapper: GetTable() Error TaskCreateNamed: %s", err)
		return 0, 0, "", "", err
	}
	if err := task.run(); err != nil {
		logrus.Debugf("devicemapper: GetTable() Error Run: %s", err)
		return 0, 0, "", "", err
	}

	devinfo, err := task.getInfo()
	if err != nil {
		logrus.Debugf("devicemapper: GetTable() Error GetInfo: %s", err)
		return 0, 0, "", "", err
	}
	if devinfo.Exists == 0 {
		logrus.Debugf("devicemapper: GetTable() Non existing device %s", name)
		return 0, 0, "", "", fmt.Errorf("devicemapper: Non existing device %s", name)
	}

	_, start, length, targetType, params := task.getNextTarget(unsafe.Pointer(nil))
	return start, length, targetType, params, nil
}

// SetTransactionID sets a transaction id for the specified device name.
func SetTransactionID(poolName string, oldID uint64, newID uint64) error {
	task, err := TaskCreateNamed(deviceTargetMsg, poolName)
	if task == nil {
		return err
	}

	if err := task.setSector(0); err != nil {
		return fmt.Errorf("devicemapper: Can't set sector %s", err)
	}

	if err := task.setMessage(fmt.Sprintf("set_transaction_id %d %d", oldID, newID)); err != nil {
		return fmt.Errorf("devicemapper: Can't set message %s", err)
	}

	if err := task.run(); err != nil {
		return fmt.Errorf("devicemapper: Error running SetTransactionID %s", err)
	}
	return nil
}

// SuspendDevice is the programmatic example of "dmsetup suspend".
// It suspends the specified device.
func SuspendDevice(name string) error {
	task, err := TaskCreateNamed(deviceSuspend, name)
	if task == nil {
		return err
	}
	if err := task.run(); err != nil {
		return fmt.Errorf("devicemapper: Error running deviceSuspend %s", err)
	}
	return nil
}

// ResumeDevice is the programmatic example of "dmsetup resume".
// It un-suspends the specified device.
func ResumeDevice(name string) error {
	task, err := TaskCreateNamed(deviceResume, name)
	if task == nil {
		return err
	}

	cookie := new(uint)
	if err := task.setCookie(cookie, 0); err != nil {
		return fmt.Errorf("devicemapper: Can't set cookie %s", err)
	}
	defer UdevWait(cookie)

	if err := task.run(); err != nil {
		return fmt.Errorf("devicemapper: Error running deviceResume %s", err)
	}

	return nil
}

// CreateDevice creates a device with the specified poolName with the specified device id.
func CreateDevice(poolName string, deviceID int) error {
	logrus.Debugf("devicemapper: CreateDevice(poolName=%v, deviceID=%v)", poolName, deviceID)
	task, err := TaskCreateNamed(deviceTargetMsg, poolName)
	if task == nil {
		return err
	}

	if err := task.setSector(0); err != nil {
		return fmt.Errorf("devicemapper: Can't set sector %s", err)
	}

	if err := task.setMessage(fmt.Sprintf("create_thin %d", deviceID)); err != nil {
		return fmt.Errorf("devicemapper: Can't set message %s", err)
	}

	dmSawExist = false // reset before the task is run
	if err := task.run(); err != nil {
		// Caller wants to know about ErrDeviceIDExists so that it can try with a different device id.
		if dmSawExist {
			return ErrDeviceIDExists
		}

		return fmt.Errorf("devicemapper: Error running CreateDevice %s", err)

	}
	return nil
}

// DeleteDevice deletes a device with the specified poolName with the specified device id.
func DeleteDevice(poolName string, deviceID int) error {
	task, err := TaskCreateNamed(deviceTargetMsg, poolName)
	if task == nil {
		return err
	}

	if err := task.setSector(0); err != nil {
		return fmt.Errorf("devicemapper: Can't set sector %s", err)
	}

	if err := task.setMessage(fmt.Sprintf("delete %d", deviceID)); err != nil {
		return fmt.Errorf("devicemapper: Can't set message %s", err)
	}

	dmSawBusy = false
	if err := task.run(); err != nil {
		if dmSawBusy {
			return ErrBusy
		}
		return fmt.Errorf("devicemapper: Error running DeleteDevice %s", err)
	}
	return nil
}

// ActivateDevice activates the device identified by the specified
// poolName, name and deviceID with the specified size.
func ActivateDevice(poolName string, name string, deviceID int, size uint64) error {
	return activateDevice(poolName, name, deviceID, size, "")
}

// ActivateDeviceWithExternal activates the device identified by the specified
// poolName, name and deviceID with the specified size.
func ActivateDeviceWithExternal(poolName string, name string, deviceID int, size uint64, external string) error {
	return activateDevice(poolName, name, deviceID, size, external)
}

func activateDevice(poolName string, name string, deviceID int, size uint64, external string) error {
	task, err := TaskCreateNamed(deviceCreate, name)
	if task == nil {
		return err
	}

	var params string
	if len(external) > 0 {
		params = fmt.Sprintf("%s %d %s", poolName, deviceID, external)
	} else {
		params = fmt.Sprintf("%s %d", poolName, deviceID)
	}
	if err := task.addTarget(0, size/512, "thin", params); err != nil {
		return fmt.Errorf("devicemapper: Can't add target %s", err)
	}
	if err := task.setAddNode(addNodeOnCreate); err != nil {
		return fmt.Errorf("devicemapper: Can't add node %s", err)
	}

	cookie := new(uint)
	if err := task.setCookie(cookie, 0); err != nil {
		return fmt.Errorf("devicemapper: Can't set cookie %s", err)
	}

	defer UdevWait(cookie)

	if err := task.run(); err != nil {
		return fmt.Errorf("devicemapper: Error running deviceCreate (ActivateDevice) %s", err)
	}

	return nil
}

// CreateSnapDeviceRaw creates a snapshot device. Caller needs to suspend and resume the origin device if it is active.
func CreateSnapDeviceRaw(poolName string, deviceID int, baseDeviceID int) error {
	task, err := TaskCreateNamed(deviceTargetMsg, poolName)
	if task == nil {
		return err
	}

	if err := task.setSector(0); err != nil {
		return fmt.Errorf("devicemapper: Can't set sector %s", err)
	}

	if err := task.setMessage(fmt.Sprintf("create_snap %d %d", deviceID, baseDeviceID)); err != nil {
		return fmt.Errorf("devicemapper: Can't set message %s", err)
	}

	dmSawExist = false // reset before the task is run
	if err := task.run(); err != nil {
		// Caller wants to know about ErrDeviceIDExists so that it can try with a different device id.
		if dmSawExist {
			return ErrDeviceIDExists
		}
		return fmt.Errorf("devicemapper: Error running deviceCreate (CreateSnapDeviceRaw) %s", err)
	}

	return nil
}

// CreateSnapDevice creates a snapshot based on the device identified by the baseName and baseDeviceId,
func CreateSnapDevice(poolName string, deviceID int, baseName string, baseDeviceID int) error {
	devinfo, _ := GetInfo(baseName)
	doSuspend := devinfo != nil && devinfo.Exists != 0

	if doSuspend {
		if err := SuspendDevice(baseName); err != nil {
			return err
		}
	}

	if err := CreateSnapDeviceRaw(poolName, deviceID, baseDeviceID); err != nil {
		if doSuspend {
			if err2 := ResumeDevice(baseName); err2 != nil {
				return fmt.Errorf("CreateSnapDeviceRaw Error: (%v): ResumeDevice Error: (%v)", err, err2)
			}
		}
		return err
	}

	if doSuspend {
		if err := ResumeDevice(baseName); err != nil {
			return err
		}
	}

	return nil
}
