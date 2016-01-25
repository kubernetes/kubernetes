package gpu

import (
	docker "github.com/fsouza/go-dockerclient"
)

type GPUDeviceDetailInfo struct {
	name  string
	mem   uint64
	clock uint
	path  []string
}

type GPUDevice struct {
	gpuType            string
	num                uint
	driverVersion      string
	gpuPlatformVersion string
	isReady            bool
	deviceDetailInfo   []GPUDeviceDetailInfo
}

type GPUDevices struct {
	num    uint
	device []GPUDevice
}

type GPUOptions struct {
}

// GPUInterface is an abstract interface for gpu related operatons.
type GPUInterface interface {
	// detect the gpu hardware and the corressponding environment
	Detect(gpuDevice *GPUDevice) error
	// parepare the overall environment
	Prepare(id string) (*docker.Container, error)
	// setup the launched parameter for each container
	SetUpEnvForEachContainer(*docker.HostConfig) error
}
