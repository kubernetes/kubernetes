package types

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// DeviceScanType is a type of device scan algorithm.
type DeviceScanType int

const (
	// LSXCmdInstanceID is the command to execute to get the instance ID.
	LSXCmdInstanceID = "instanceID"

	// LSXCmdLocalDevices is the command to execute to get the local devices
	// map.
	LSXCmdLocalDevices = "localDevices"

	// LSXCmdNextDevice is the command to execute to get the next device.
	LSXCmdNextDevice = "nextDevice"

	// LSXCmdWaitForDevice is the command to execute to wait until a device,
	// identified by volume ID, is presented to the system.
	LSXCmdWaitForDevice = "wait"
)

const (

	// DeviceScanQuick performs a shallow, quick scan.
	DeviceScanQuick DeviceScanType = iota

	// DeviceScanDeep performs a deep, longer scan.
	DeviceScanDeep
)

// String returns the string representation of a DeviceScanType.
func (st DeviceScanType) String() string {
	switch st {
	case DeviceScanQuick:
		return "quick"
	case DeviceScanDeep:
		return "deep"
	}
	return ""
}

// ParseDeviceScanType parses a device scan type.
func ParseDeviceScanType(i interface{}) DeviceScanType {
	switch ti := i.(type) {
	case string:
		lti := strings.ToLower(ti)
		if lti == DeviceScanQuick.String() {
			return DeviceScanQuick
		} else if lti == DeviceScanDeep.String() {
			return DeviceScanDeep
		}
		i, err := strconv.Atoi(ti)
		if err != nil {
			return DeviceScanQuick
		}
		return ParseDeviceScanType(i)
	case int:
		st := DeviceScanType(ti)
		if st == DeviceScanQuick || st == DeviceScanDeep {
			return st
		}
		return DeviceScanQuick
	default:
		return ParseDeviceScanType(fmt.Sprintf("%v", ti))
	}
}

// LocalDevicesOpts are options when getting a list of local devices.
type LocalDevicesOpts struct {
	ScanType DeviceScanType
	Opts     Store
}

// WaitForDeviceOpts are options when waiting on specific local device to
// appear.
type WaitForDeviceOpts struct {
	LocalDevicesOpts

	// Token is the value returned by a remote VolumeAttach call that the
	// client can use to block until a specific device has appeared in the
	// local devices list.
	Token string

	// Timeout is the maximum duration for which to wait for a device to
	// appear in the local devices list.
	Timeout time.Duration
}

// NewStorageExecutor is a function that constructs a new StorageExecutors.
type NewStorageExecutor func() StorageExecutor

// StorageExecutor is the part of a storage driver that is downloaded at
// runtime by the libStorage client.
type StorageExecutor interface {
	Driver
	StorageExecutorFunctions
}

// StorageExecutorFunctions is the collection of functions that are required of
// a StorageExecutor.
type StorageExecutorFunctions interface {
	// InstanceID returns the local system's InstanceID.
	InstanceID(
		ctx Context,
		opts Store) (*InstanceID, error)

	// NextDevice returns the next available device.
	NextDevice(
		ctx Context,
		opts Store) (string, error)

	// LocalDevices returns a map of the system's local devices.
	LocalDevices(
		ctx Context,
		opts *LocalDevicesOpts) (*LocalDevices, error)
}

// ProvidesStorageExecutorCLI is a type that provides the StorageExecutorCLI.
type ProvidesStorageExecutorCLI interface {
	// XCLI returns the StorageExecutorCLI.
	XCLI() StorageExecutorCLI
}

// StorageExecutorCLI provides a way to interact with the CLI tool built with
// the driver implementations of the StorageExecutor interface.
type StorageExecutorCLI interface {
	StorageExecutorFunctions

	// WaitForDevice blocks until the provided attach token appears in the
	// map returned from LocalDevices or until the timeout expires, whichever
	// occurs first.
	//
	// The return value is a boolean flag indicating whether or not a match was
	// discovered as well as the result of the last LocalDevices call before a
	// match is discovered or the timeout expires.
	WaitForDevice(
		ctx Context,
		opts *WaitForDeviceOpts) (bool, *LocalDevices, error)
}
