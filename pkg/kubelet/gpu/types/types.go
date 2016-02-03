package types

import (
// docker "github.com/fsouza/go-dockerclient"
)

type GPUDevice struct {
	Path string

	Cores  uint
	Memory uint

	Family string
}

type GPUPlatform struct {
	Name            string
	DriverVersion   string
	PlatformVersion string
}

type GPUDevices struct {
	GPUPlatform
	Devices []GPUDevice
}

// type GPUOptions struct {
// }

// GPUInterface is an abstract interface for gpu related operatons.
type GPUPlugin interface {
	// Init plugin before use it
	InitPlugin() error

	// detect the gpu hardware and the corresponding environment
	Detect() (*GPUDevices, error)

	// Init GPU environment before launch container
	InitGPUEnv() error

	// // parepare the overall environment
	// Prepare(id string) (*docker.Container, error)
	// // setup the launched parameter for each container
	// SetUpEnvForEachContainer(*docker.HostConfig) error

	// Name returns the plugin's name. This will be used when searching
	// for a plugin by name, e.g.
	Name() string
}

type GPUCommonInfo struct {
	Name string
}

type GPUDeviceState struct {
	IsOccupied bool
}

type GPUState struct {
	IsInit    bool
	DevsState []*GPUDeviceState
}
