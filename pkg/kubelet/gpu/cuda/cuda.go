package cuda

import (
	"fmt"
	"github.com/golang/glog"
	cudaTools "github.com/haniceboy/nvidia-docker/tools/src/cuda"
	"github.com/haniceboy/nvidia-docker/tools/src/docker"
	"github.com/haniceboy/nvidia-docker/tools/src/nvidia"
	"github.com/haniceboy/nvidia-docker/tools/src/nvml"
	gpuTypes "k8s.io/kubernetes/pkg/kubelet/gpu/types"

	dockerClient "github.com/fsouza/go-dockerclient"
)

const (
	CudaName string = "cuda"
	CudaDeviceCtl string = "/dev/nvidiactl"
	CudaDeviceUVM string = "/dev/nvidia-uvm"
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
	nvidia.LoadUVM()
	return nvidia.Init()
}

func (cuda *Cuda) ReleasePlugin() error {
	return nvidia.Shutdown()
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

func (cuda *Cuda) InitGPUEnv(client DockerInterface) error {
	return createLocalVolumes()
}

func (cuda *Cuda) createLocalVolumes() error {
	drv, err := nvidia.GetDriverVersion()
	if err != nil {
		return err
	}
	vols, err := nvidia.LookupVolumes("")
	if err != nil {
		return err
	}

	for _, v := range vols {
		n := fmt.Sprintf("%s_%s", v.Name, drv)
		if _, err := docker.InspectVolume(n); err == nil {
			if err = docker.RemoveVolume(n); err != nil {
				return fmt.Errorf("cannot remove %s: volume is in use", n)
			}
		}

		if err := docker.CreateVolume(n); err != nil {
			return err
		}
		path, err := docker.InspectVolume(n)
		if err != nil {
			docker.RemoveVolume(n)
			return err
		}
		if err := v.CreateAt(path); err != nil {
			docker.RemoveVolume(n)
			return err
		}
		fmt.Println(n)
	}
	return nil
}

func (cuda *Cuda)IsImageSupported(image string) (bool, error) {
	cv, err := nvidia.GetCUDAVersion()
	if err != nil {
		return false, fmt.Errorf("Failed to detect the host cuda version(%s)", err)
	}

	if isSupported, err := cuda.cudaSupported(image, cv); err != nil {
		return false, fmt.Errorf("Failed to compare the cuda version of image(%s): %s", image, err)
	}

	return isSupported, nil
}

func (cuda *Cuda)GenerateDeviceOpts(gpuIdxs []int) ([]dockerClient.Device, error) {
	devicesOpts := make([]dockerClient.Device)

	for _, idx := range gpuIdxs {
 		var device docker.Device
        device.CgroupPermissions = "rwm"
        if idx > 0 && idx < len(cuda.gpuDevices.Devices) {
	        device.PathOnHost = cuda.gpuDevices.Devices[idx].Path
        	device.PathInContainer = cuda.gpuDevices.Devices[idx].Path                                                                                                              
        	devicesOpts = append(devicesOpts, device)      	
        } else {
        	return nil, fmt.Errorf("Cannot find the cuda gpu idx(%d) of total %d", idx, len(cuda.gpuDevices.Devices))
        }

	}

	extraDevs := []string{CudaDeviceCtl, CudaDeviceUVM}
	for _, path := range gpuIdxs {
 		var device docker.Device
        device.CgroupPermissions = "rwm"   
        device.PathOnHost = path
        device.PathInContainer = path                                                                                                             
        devicesOpts = append(devicesOpts, device)
	}

	return devicesOpts, nil
}

func (cuda *Cuda)GenerateVolumeOpts(image string) (map[string]struct{}, error) {
	result := make(map[string]struct{})

	// check whether the image need cuda support
	vols, err := cuda.volumesNeeded(image)
	if err != nil {
		return result, fmt.Errorf("Failed to detect whether cuda volume need or not(%s)", err)
	}
	// the image needn't cuda support
	if vols == nil {
		return result, nil
	} 

	drv, err := nvidia.GetDriverVersion()
	if err != nil {
		return nil, err
	}
	for _, vol := range nvidia.Volumes {
		for _, v := range vols {
			if v == vol.Name {
				// Check if the volume exists locally otherwise fallback to using the plugin
				n := fmt.Sprintf("%s_%s", vol.Name, drv)
				if _, err := docker.InspectVolume(n); err == nil {
					result[fmt.Sprintf("%s:%s:ro", n, vol.Mountpoint)] = struct{}{}
				} else {
					return result, fmt.Errorf("Cannot find the volume %s for cuda", n)
				}
				break
			}
		}
	}	

	
	return result, nil
}

func (cuda *Cuda) generateOptions(image string, vols []string) (deviceOptions []string, volumeOptions []string, error) {
	cv, err := nvidia.GetCUDAVersion()
	if err != nil {
		return nil, nil, err
	}
	if err := cudaSupported(image, cv); err != nil {
		return nil, nil, err
	}

	deviceOptions, err := devicesOptions()
	if err != nil {
		return nil, nil, err
	}

	volumeOptions, err := volumesOptions(vols)
	if err != nil {
		return nil, nil, err
	}
	return deviceOptions, volumeOptions, nil
}

func (cuda *Cuda) devicesOptions() ([]string, error) {
	args := []string{"--device=/dev/nvidiactl", "--device=/dev/nvidia-uvm"}

	devs, err := nvidia.LookupDevicePaths()
	if err != nil {
		return nil, err
	}

	if len(GPU) == 0 {
		for i := range devs {
			args = append(args, fmt.Sprintf("--device=%s", devs[i]))
		}
	} else {
		for _, id := range GPU {
			i, err := strconv.Atoi(id)
			if err != nil || i < 0 || i >= len(devs) {
				return nil, fmt.Errorf("invalid device: %s", id)
			}
			args = append(args, fmt.Sprintf("--device=%s", devs[i]))
		}
	}
	return args, nil
}

func (cuda *Cuda) volumesOptions(vols []string) ([]string, error) {
	args := make([]string, 0, len(vols))

	drv, err := nvidia.GetDriverVersion()
	if err != nil {
		return nil, err
	}
	for _, vol := range nvidia.Volumes {
		for _, v := range vols {
			if v == vol.Name {
				// Check if the volume exists locally otherwise fallback to using the plugin
				n := fmt.Sprintf("%s_%s", vol.Name, drv)
				if _, err := docker.InspectVolume(n); err == nil {
					args = append(args, fmt.Sprintf("--volume=%s:%s:ro", n, vol.Mountpoint))
				} else {
					args = append(args, fmt.Sprintf("--volume-driver=%s", nvidia.DockerPlugin))
					args = append(args, fmt.Sprintf("--volume=%s:%s:ro", n, vol.Mountpoint))
				}
				break
			}
		}
	}
	return args, nil
}

func (cuda *Cuda) volumesNeeded(image string) ([]string, error) {
	// it already pulled image.
	label, err := docker.Label(image, labelVolumesNeeded)
	if err != nil {
		return nil, err
	}
	if label == "" {
		return nil, nil
	}
	return strings.Split(label, " "), nil
}

func (cuda *Cuda) cudaSupported(image, version string) (bool, error) {
	var vmaj, vmin int
	var lmaj, lmin int

	label, err := docker.Label(image, labelCUDAVersion)
	if err != nil {
		return false, err
	}
	if label == "" {
		return true, nil
	}
	if _, err := fmt.Sscanf(version, "%d.%d", &vmaj, &vmin); err != nil {
		return false, err
	}
	if _, err := fmt.Sscanf(label, "%d.%d", &lmaj, &lmin); err != nil {
		return false, err
	}
	if lmaj > vmaj || (lmaj == vmaj && lmin > vmin) {
		glog.Warnf("%s", fmt.Errorf("Unsupported CUDA version: %s < %s", label, version))
		return false, nil 
	}
	return true, nil
}
