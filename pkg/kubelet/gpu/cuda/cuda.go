package cuda

import (
	"fmt"
	"github.com/golang/glog"
	cudaTools "github.com/haniceboy/nvidia-docker/tools/src/cuda"
	"github.com/haniceboy/nvidia-docker/tools/src/nvidia"
	"github.com/haniceboy/nvidia-docker/tools/src/nvml"
	gpuTypes "k8s.io/kubernetes/pkg/kubelet/gpu/types"
)

const (
	CudaName string = "cuda"
)

type Cuda struct {
	commonInfo gpuTypes.GPUCommonInfo
}

func ProbeGPUPlugin() gpuTypes.GPUPlugin {
	glog.Infof("Hans: cuda.ProbeGPUPlugin()")
	return &Cuda{
		commonInfo: gpuTypes.GPUCommonInfo{Name: CudaName},
	}
}

func (cuda *Cuda) Detect() (*gpuTypes.GPUDevices, error) {
	glog.Infof("Hans: cuda.Detect()")
	gpuDevices := gpuTypes.GPUDevices{}

	cudaDevices, err := nvidia.LookupDevices()
	if err != nil {
		glog.Warningf("Cannot detect the nvidia gpu %q", err)
		return nil, fmt.Errorf("Failed to detect the nvidia gpu device")
	}

	gpuDevs := []gpuTypes.GPUDevice{}
	for _, cudaDevice := range cudaDevices {
		dev := gpuTypes.GPUDevice{}
		dev.Path = cudaDevice.NVMLDev.Path
		dev.Cores = cudaDevice.CUDADev.Cores
		dev.Memory = cudaDevice.CUDADev.Memory.Global
		dev.Family = cudaDevice.CUDADev.Family
		gpuDevs = append(gpuDevs, dev)
	}

	driverVersion, err := nvml.GetDriverVersion()
	if err == nil {
		gpuDevices.GPUPlatform.DriverVersion = driverVersion
	}

	platformVersion, err := cudaTools.GetDriverVersion()
	if err == nil {
		gpuDevices.GPUPlatform.PlatformVersion = platformVersion
	}

	gpuDevices.GPUPlatform.Name = cuda.commonInfo.Name

	glog.Infof("Hans: cuda.Detect(): gpuDevices:%+v", gpuDevices)
	return &gpuDevices, nil
}

func (cuda *Cuda) Name() string {
	return cuda.commonInfo.Name
}
