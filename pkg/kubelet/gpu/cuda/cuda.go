package cuda

import (
	"fmt"
	"github.com/golang/glog"
	cudaTools "github.com/haniceboy/nvidia-docker/tools/src/cuda"
	"github.com/haniceboy/nvidia-docker/tools/src/docker"
	"github.com/haniceboy/nvidia-docker/tools/src/nvidia"
	"github.com/haniceboy/nvidia-docker/tools/src/nvml"
	gpuTypes "k8s.io/kubernetes/pkg/kubelet/gpu/types"
	"sync"

	dockerClient "github.com/fsouza/go-dockerclient"
)

const (
	CudaName      string = "cuda"
	CudaDeviceCtl string = "/dev/nvidiactl"
	CudaDeviceUVM string = "/dev/nvidia-uvm"
)

type Cuda struct {
	gpuInfo gpuTypes.GPUInfo
}

func ProbeGPUPlugin() gpuTypes.GPUPlugin {
	glog.Infof("Hans: cuda.ProbeGPUPlugin()")
	return &Cuda{
		gpuInfo: gpuTypes.GPUInfo{commonInfo: gpuTypes.GPUCommonInfo{Name: CudaName}},
	}
}

func (cuda *Cuda) Name() string {
	glog.Infof("Hans: cuda.Name()")
	cuda.gpuInfo.lock.RLock()
	def cuda.gpuInfo.lock.RUnlock()
	return cuda.gpuInfo.commonInfo.Name
}

func (cuda *Cuda) InitPlugin() error {
	glog.Infof("Hans: cuda.InitPlugin()")
	nvidia.LoadUVM()
	return nvidia.Init()
}

func (cuda *Cuda) ReleasePlugin() error {
	glog.Infof("Hans: cuda.ReleasePlugin()")
	return nvidia.Shutdown()
}

func (cuda *Cuda) AllocGPU(gpuReqs uint) ([]uint, error) {
	glog.Infof("Hans: cuda.AllocGPU()")
	totalGPUNum := len(cuda.gpuInfo.gpuDevices.Devices)
	if gpuReqs >  totalGPUNum{
		return []uint{}, fmt.Errorf("Cannot alloc %d cuda gpu, because there are %d", gpuReqs, totalGPUNum)
	}

	cuda.gpuInfo.lock.Lock()
	def cuda.gpuInfo.lock.Unlock()

	// check whether there are enough free gpus
	freeGPUIdx := []uint{}
	freedGPUNum := uint(0)
	for idx, gpuDevice :=range cuda.gpuInfo.gpuDevices.Devices {
		if !gpuDevice.GPUDeviceState.IsOccupied {
			freedGPUNum++
			freeGPUIdx = append(freeGPUIdx, idx)
		}
	}

	if freedGPUNum >= gpuReqs {
		var result =make([]uint, gpuReqs);
		cc := copy(result, freeGPUIdx[:gpuReqs])
		if cc == gpuReqs {
			for _, idx := range result {
				cuda.gpuInfo.gpuDevices.Devices[idx].GPUDeviceState.IsOccupied = true
			}
			return result, nil
		} else {
			return []uint{}, fmt.Errorf("Failed to generate gpu index slice")
		}
	} else {
		return []uint{}, fmt.Errorf("Cannot meet the required gpu number %d and only have %d freed gpus", gpuReqs, freedGPUNum)
	}


}

func (cuda *Cuda) FreeGPU(gpuIdxs []uint) error {
	defer func(){
        if err:=recover();err!=nil{
            glog.Errf("Failed to free GPU(%s). Reason: %s",gpuIdxs, err)
        }
    }()

	glog.Infof("Hans: cuda.FreeGPU()")
	if len(gpuIdxs) == 0 {
		return nil
	}

	cuda.gpuInfo.lock.Lock()
	def cuda.gpuInfo.lock.Unlock()

	// check whether the passed gpuIdxs is valid or not
	for _, idx := range gpuIdxs {
		if !cuda.gpuInfo.gpuDevices.Devices[idx].GPUDeviceState.IsOccupied {
			return fmt.Errorf("Failed to free gpu %d, because it is not occupied", idx)
		}
	}

	// remove the occupy flag
	for _, idx := range gpuIdxs {
		cuda.gpuInfo.gpuDevices.Devices[idx].GPUDeviceState.IsOccupied  = false
	}

	return nil
}

func (cuda *Cuda) Detect() (*gpuTypes.GPUDevices, error) {
	glog.Infof("Hans: cuda.Detect()")

	cuda.gpuInfo.lock.Lock()
	def cuda.gpuInfo.lock.Unlock()
	if cuda.gpuDevices != nil {
		return cuda.gpuDevices, nil
	}

	gpuDevices := gpuTypes.GPUDevices{GPUState: gpuTypes.GPUState{IsInit: false}}

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

		// init gpu device state
		dev.GPUDeviceState = gpuTypes.GPUDeviceState{IsOccupied: false}

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

	cuda.gpuInfo.gpuDevices = &gpuDevices
	glog.Infof("Hans: cuda.Detect(): gpuDevices:%+v", gpuDevices)
	return cuda.gpuInfo.gpuDevices, nil
}

func (cuda *Cuda) InitGPUEnv(client DockerInterface) error {
	glog.Infof("Hans: cuda.InitGPUEnv()")
	cuda.gpuInfo.lock.Lock()
	def cuda.gpuInfo.lock.Unlock()
	// check whether it already done init
	if cuda.gpuInfo.gpuDevices.GPUState.IsInit {
		return nil
	}

	if err:= cuda.createLocalVolumes(); err == nil {
		cuda.gpuInfo.lock.Lock()
		def cuda.gpuInfo.lock.Unlock()
		cuda.gpuInfo.gpuDevices.GPUState.IsInit = true;
	}

	return err
}

func (cuda *Cuda) createLocalVolumes() error {
	glog.Infof("Hans: cuda.createLocalVolumes()")
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

func (cuda *Cuda) IsImageSupported(image string) (bool, error) {
	glog.Infof("Hans: cuda.IsImageSupported()")
	cv, err := nvidia.GetCUDAVersion()
	if err != nil {
		return false, fmt.Errorf("Failed to detect the host cuda version(%s)", err)
	}

	if isSupported, err := cuda.cudaSupported(image, cv); err != nil {
		return false, fmt.Errorf("Failed to compare the cuda version of image(%s): %s", image, err)
	}

	return isSupported, nil
}

func (cuda *Cuda) GenerateDeviceOpts(gpuIdxs []uint) ([]dockerClient.Device, error) {
	glog.Infof("Hans: cuda.GenerateDeviceOpts()")
	devicesOpts := make([]dockerClient.Device)

	cuda.gpuInfo.lock.RLock()
	def cuda.gpuInfo.lock.RUnlock()

	for _, idx := range gpuIdxs {
		var device docker.Device
		device.CgroupPermissions = "rwm"
		if idx < len(cuda.gpuDevices.Devices) {
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

func (cuda *Cuda) GenerateVolumeOpts(image string) (map[string]struct{}, error) {
	glog.Infof("Hans: cuda.GenerateVolumeOpts()")
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

func (cuda *Cuda) volumesNeeded(image string) ([]string, error) {
	glog.Infof("Hans: cuda.volumesNeeded()")
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
	glog.Infof("Hans: cuda.cudaSupported()")
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
