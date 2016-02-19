package types

import (
	dockerClient "github.com/fsouza/go-dockerclient"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"sync"
)

const (
	// Keep gpu information in label
	KubernetesContainerGPUNameLabel  = "io.kubernetes.container.gpu.name"
	KubernetesContainerGPUIndexLabel = "io.kubernetes.container.gpu.index"
)

type GPUState struct {
	IsInit bool
}

type GPUDevice struct {
	Path   string
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
	AllocGPU(gpuReqs int, PodUID types.UID, container *api.Container) ([]int, error)

	// Free gpu device occupied by the container
	FreeGPU(PodUID types.UID, container *api.Container) error

	// Whether the host environment match the image request
	IsImageSupported(image string) (bool, error)

	// Generate Device Options
	GenerateDeviceOpts(gpuIdxes []int) ([]dockerClient.Device, error)

	// Generate Volume Options
	GenerateVolumeOpts(image string) ([]string, error)

	UpdateGPUUsageStatus(newGPUStatus *map[PodCotainerHashID]GPUUsageStatus)

	// Name returns the plugin's name. This will be used when searching
	// for a plugin by name, e.g.
	Name() string
}

type GPUCommonInfo struct {
	Name string
}

type GPUUsageStatus struct {
	// ID of the pod.
	// PodID types.UID

	// ContainerName string

	GPUIndexes []int
}

type PodCotainerHashID uint64

type GPUInfo struct {
	Lock       sync.RWMutex
	CommonInfo GPUCommonInfo
	GPUDevices *GPUDevices
	GPUStatus  map[PodCotainerHashID]GPUUsageStatus
}
