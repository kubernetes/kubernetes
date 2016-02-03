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
	gpuDevices *gpuTypes.GPUDevices
}

func ProbeGPUPlugin() gpuTypes.GPUPlugin {
	glog.Infof("Hans: cuda.ProbeGPUPlugin()")
	return &Cuda{
		commonInfo: gpuTypes.GPUCommonInfo{Name: CudaName},
	}
}

func (cuda *Cuda) Name() string {
	return cuda.commonInfo.Name
}

func (cuda *Cuda) InitPlugin() error {
	return nvidia.Init()
}

func (cuda *Cuda) Detect() (*gpuTypes.GPUDevices, error) {
	glog.Infof("Hans: cuda.Detect()")

	if cuda.gpuDevices != nil {
		return cuda.gpuDevices, nil
	}

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

	gpuDevices.Devices = gpuDevs

	driverVersion, err := nvml.GetDriverVersion()
	if err == nil {
		gpuDevices.GPUPlatform.DriverVersion = driverVersion
	}

	platformVersion, err := cudaTools.GetDriverVersion()
	if err == nil {
		gpuDevices.GPUPlatform.PlatformVersion = platformVersion
	}

	gpuDevices.GPUPlatform.Name = cuda.commonInfo.Name

	cuda.gpuDevices = &gpuDevices
	glog.Infof("Hans: cuda.Detect(): gpuDevices:%+v", gpuDevices)
	return cuda.gpuDevices, nil
}

func (cuda *Cuda) InitGPUEnv() error {
	return createLocalVolumes()
}

func createLocalVolumes() error {
	// drv, err := nvidia.GetDriverVersion()
	// if err != nil {
	// 	return err
	// }
	// vols, err := nvidia.LookupVolumes("")
	// if err != nil {
	// 	return err
	// }

	// for _, v := range vols {
	// 	n := fmt.Sprintf("%s_%s", v.Name, drv)
	// 	if _, err := docker.InspectVolume(n); err == nil {
	// 		if err = docker.RemoveVolume(n); err != nil {
	// 			return fmt.Errorf("cannot remove %s: volume is in use", n)
	// 		}
	// 	}

	// 	if err := docker.CreateVolume(n); err != nil {
	// 		return err
	// 	}
	// 	path, err := docker.InspectVolume(n)
	// 	if err != nil {
	// 		docker.RemoveVolume(n)
	// 		return err
	// 	}
	// 	if err := v.CreateAt(path); err != nil {
	// 		docker.RemoveVolume(n)
	// 		return err
	// 	}
	// 	fmt.Println(n)
	// }
	return nil
}
