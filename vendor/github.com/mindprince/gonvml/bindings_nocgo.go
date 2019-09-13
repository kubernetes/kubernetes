// +build !cgo

package gonvml

import (
	"errors"
	"time"
)

var errNoCgo = errors.New("this binary is built without CGO, NVML is disabled")

// Initialize initializes NVML.
// Call this before calling any other methods.
func Initialize() error {
	return errNoCgo
}

// Shutdown shuts down NVML.
// Call this once NVML is no longer being used.
func Shutdown() error {
	return errNoCgo
}

// SystemDriverVersion returns the the driver version on the system.
func SystemDriverVersion() (string, error) {
	return "", errNoCgo
}

// DeviceCount returns the number of nvidia devices on the system.
func DeviceCount() (uint, error) {
	return 0, errNoCgo
}

// Device is the handle for the device.
// This handle is obtained by calling DeviceHandleByIndex().
type Device struct {
}

// DeviceHandleByIndex returns the device handle for a particular index.
// The indices range from 0 to DeviceCount()-1. The order in which NVML
// enumerates devices has no guarantees of consistency between reboots.
func DeviceHandleByIndex(idx uint) (Device, error) {
	return Device{}, errNoCgo
}

// MinorNumber returns the minor number for the device.
// The minor number for the device is such that the Nvidia device node
// file for each GPU will have the form /dev/nvidia[minor number].
func (d Device) MinorNumber() (uint, error) {
	return 0, errNoCgo
}

// UUID returns the globally unique immutable UUID associated with this device.
func (d Device) UUID() (string, error) {
	return "", errNoCgo
}

// Name returns the product name of the device.
func (d Device) Name() (string, error) {
	return "", errNoCgo
}

// MemoryInfo returns the total and used memory (in bytes) of the device.
func (d Device) MemoryInfo() (uint64, uint64, error) {
	return 0, 0, errNoCgo
}

// UtilizationRates returns the percent of time over the past sample period during which:
// utilization.gpu: one or more kernels were executing on the GPU.
// utilizatoin.memory: global (device) memory was being read or written.
func (d Device) UtilizationRates() (uint, uint, error) {
	return 0, 0, errNoCgo
}

// PowerUsage returns the power usage for this GPU and its associated circuitry
// in milliwatts. The reading is accurate to within +/- 5% of current power draw.
func (d Device) PowerUsage() (uint, error) {
	return 0, errNoCgo
}

// AveragePowerUsage returns the power usage for this GPU and its associated circuitry
// in milliwatts averaged over the samples collected in the last `since` duration.
func (d Device) AveragePowerUsage(since time.Duration) (uint, error) {
	return 0, errNoCgo
}

// AverageGPUUtilization returns the utilization.gpu metric (percent of time
// one of more kernels were executing on the GPU) averaged over the samples
// collected in the last `since` duration.
func (d Device) AverageGPUUtilization(since time.Duration) (uint, error) {
	return 0, errNoCgo
}

// Temperature returns the temperature for this GPU in Celsius.
func (d Device) Temperature() (uint, error) {
	return 0, errNoCgo
}

// FanSpeed returns the temperature for this GPU in the percentage of its full
// speed, with 100 being the maximum.
func (d Device) FanSpeed() (uint, error) {
	return 0, errNoCgo
}

// EncoderUtilization returns the percent of time over the last sample period during which the GPU video encoder was being used.
// The sampling period is variable and is returned in the second return argument in microseconds.
func (d Device) EncoderUtilization() (uint, uint, error) {
	return 0, 0, errNoCgo
}

// DecoderUtilization returns the percent of time over the last sample period during which the GPU video decoder was being used.
// The sampling period is variable and is returned in the second return argument in microseconds.
func (d Device) DecoderUtilization() (uint, uint, error) {
	return 0, 0, errNoCgo
}
