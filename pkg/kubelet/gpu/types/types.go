package types

import (
	dockerClient "github.com/fsouza/go-dockerclient"
	"sync"
)

type GPUDeviceState struct {
	IsOccupied bool
	// ContainerId string
	// PodId       string
}

type GPUState struct {
	IsInit bool
}

type GPUDevice struct {
	Path   string
	Cores  uint
	Memory uint
	Family string

	GPUDeviceState
}

type GPUPlatform struct {
	Name            string
	DriverVersion   string
	PlatformVersion string
}

type GPUDevices struct {
	GPUState
	GPUPlatform
	Devices []GPUDevice
}

// GPUInterface is an abstract interface for gpu related operatons.
type GPUPlugin interface {
	// Init plugin before use it
	InitPlugin() error

	// Rlease the plugin at last
	ReleasePlugin() error

	// Detect the gpu hardware and the corresponding environment
	Detect() (*GPUDevices, error)

	// Init GPU environment before launch container
	InitGPUEnv() error

	// Alloc gpu device for container
	AllocGPU(gpuReqs uint) ([]uint, error)

	// Free gpu device from container
	FreeGPU(gpuIdxs []uint) error

	// Whether the host environment match the image request
	IsImageSupported(image string) (bool, error)

	// Generate Device Options
	GenerateDeviceOpts(gpuIdxs []uint) ([]dockerClient.Device, error)

	// Generate Volume Options
	GenerateVolumeOpts(image string) (map[string]struct{}, error)

	// Name returns the plugin's name. This will be used when searching
	// for a plugin by name, e.g.
	Name() string
}

type GPUCommonInfo struct {
	Name string
}

type GPUInfo struct {
	Lock       sync.RWMutex
	CommonInfo GPUCommonInfo
	GPUDevices *GPUDevices
}
