package cuda

import (
	"fmt"
	"github.com/golang/glog"
	cudaTools "github.com/haniceboy/nvidia-docker/tools/src/cuda"
	"github.com/haniceboy/nvidia-docker/tools/src/docker"
	"github.com/haniceboy/nvidia-docker/tools/src/nvidia"
	"github.com/haniceboy/nvidia-docker/tools/src/nvml"
	"k8s.io/kubernetes/pkg/api"
	gpuTypes "k8s.io/kubernetes/pkg/kubelet/gpu/types"
	gpuUtil "k8s.io/kubernetes/pkg/kubelet/gpu/util"
	"k8s.io/kubernetes/pkg/types"
	"strings"

	dockerClient "github.com/fsouza/go-dockerclient"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	CudaName string = "cuda"

	cudaDeviceCtl string = "/dev/nvidiactl"
	cudaDeviceUVM string = "/dev/nvidia-uvm"

	labelCUDAVersion   string = "com.nvidia.cuda.version"
	labelVolumesNeeded string = "com.nvidia.volumes.needed"
)

type Cuda struct {
	gpuInfo gpuTypes.GPUInfo
}

func ProbeGPUPlugin() gpuTypes.GPUPlugin {
	glog.Infof("Hans: cuda.ProbeGPUPlugin()")
	return &Cuda{
		gpuInfo: gpuTypes.GPUInfo{CommonInfo: gpuTypes.GPUCommonInfo{Name: CudaName}},
	}
}

func (cuda *Cuda) Name() string {
	glog.Infof("Hans: cuda.Name()")
	cuda.gpuInfo.Lock.RLock()
	defer cuda.gpuInfo.Lock.RUnlock()
	return cuda.gpuInfo.CommonInfo.Name
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

func (cuda *Cuda) AllocGPU(gpuReqs int, podUID types.UID, container *api.Container) ([]int, error) {
	glog.Infof("Hans: cuda.AllocGPU(): gpuReqs: %d, container(%s), pod(%s)", gpuReqs, container.Name, string(podUID))

	cuda.gpuInfo.Lock.Lock()
	defer cuda.gpuInfo.Lock.Unlock()

	// Check whether it already alloced GPU for the container.
	// It avoid to re-alloc gpu for a restart container.
	containerHash := gpuUtil.HashContainerFromData(podUID, container)
	gpuStatus, found := cuda.gpuInfo.GPUStatus[containerHash]
	if found {
		glog.Errorf("Hans: cuda.AllocGPU(): it already alloced gpu for(container(%s), pod(%s))", container.Name, string(podUID))
		return gpuStatus.GPUIndexes, nil
	}

	totalGPUNum := len(cuda.gpuInfo.GPUDevices.Devices)
	if gpuReqs > totalGPUNum {
		return []int{}, fmt.Errorf("Cannot alloc %d cuda gpu, because there are only %d", gpuReqs, totalGPUNum)
	}

	availableGPUIdx, err := cuda.getAvailableGPUs()
	if err != nil {
		return []int{}, fmt.Errorf("Cannot alloc cuda gpu (%s)", err)
	}

	availableGPUNum := len(availableGPUIdx)
	glog.Infof("Hans: cuda.AllocGPU(): availableGPUNum: %d, gpuReqs: %d, availableGPUIdx: %+v", availableGPUNum, gpuReqs, availableGPUIdx)
	if availableGPUNum >= gpuReqs {
		glog.Infof("Hans: cuda.AllocGPU(): before alloc: gpustatus: %+v", cuda.gpuInfo.GPUStatus)
		var result = make([]int, gpuReqs)
		cc := copy(result, availableGPUIdx[:gpuReqs])
		glog.Infof("Hans: cuda.AllocGPU(): cc: %d, result: %+v", cc,  result)
		if cc == gpuReqs {
			// gpuUsageStatus := gpuTypes.GPUUsageStatus{PodID: podUID, ContainerName: container.Name, GPUIndexes: result}
			gpuUsageStatus := gpuTypes.GPUUsageStatus{GPUIndexes: result}
			glog.Infof("Hans: cuda.AllocGPU(): gupUsageStatus: %+v, containerHash: %d", gpuUsageStatus, containerHash)
			if cuda.gpuInfo.GPUStatus == nil {
				cuda.gpuInfo.GPUStatus = map[gpuTypes.PodCotainerHashID]gpuTypes.GPUUsageStatus{}
			}
			cuda.gpuInfo.GPUStatus[containerHash] = gpuUsageStatus
			glog.Infof("Hans: cuda.AllocGPU(): after alloc: gpustatus: %+v", cuda.gpuInfo.GPUStatus)
			return result, nil
		} else {
			return []int{}, fmt.Errorf("Failed to generate gpu index slice")
		}
	} else {
		return []int{}, fmt.Errorf("Cannot meet the required gpu number %d and only have %d available gpus", gpuReqs, availableGPUNum)
	}

}

func (cuda *Cuda) FreeGPU(PodUID types.UID, container *api.Container) error {
	defer func() {
		if err := recover(); err != nil {
			glog.Errorf("Failed to free GPU for the container(%s) of pod(%s). Reason: %s", container.Name, string(PodUID), err)
		}
	}()

	glog.Infof("Hans: cuda.FreeGPU(): podId: %s, container: %s", string(PodUID), container.Name)

	containerHash := gpuUtil.HashContainerFromData(PodUID, container)

	cuda.gpuInfo.Lock.Lock()
	defer cuda.gpuInfo.Lock.Unlock()
	delete(cuda.gpuInfo.GPUStatus, containerHash)

	// double check whether free it or not
	_, found := cuda.gpuInfo.GPUStatus[containerHash]
	if found {
		glog.Errorf("Failed to free GPU for the container(%s) of pod(%s). Reason: internal error", container.Name, string(PodUID))
	}

	return nil
}

func (cuda *Cuda) Detect() (*gpuTypes.GPUDevices, error) {
	glog.Infof("Hans: cuda.Detect()")

	cuda.gpuInfo.Lock.Lock()
	defer cuda.gpuInfo.Lock.Unlock()
	if cuda.gpuInfo.GPUDevices != nil {
		return cuda.gpuInfo.GPUDevices, nil
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

	gpuDevices.GPUPlatform.Name = cuda.gpuInfo.CommonInfo.Name

	cuda.gpuInfo.GPUDevices = &gpuDevices
	glog.Infof("Hans: cuda.Detect(): gpuDevices:%+v", gpuDevices)
	return cuda.gpuInfo.GPUDevices, nil
}

func (cuda *Cuda) InitGPUEnv() error {
	glog.Infof("Hans: cuda.InitGPUEnv()")
	cuda.gpuInfo.Lock.Lock()
	defer cuda.gpuInfo.Lock.Unlock()
	// check whether it already done init
	if cuda.gpuInfo.GPUDevices.GPUState.IsInit {
		return nil
	}

	if err := cuda.createLocalVolumes(); err == nil {
		cuda.gpuInfo.GPUDevices.GPUState.IsInit = true
	} else {
		return err
	}

	return nil
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
			glog.Infof("Hans: volumes %s already exist", n)
			return nil
		} else {
			// volumes is not exist and create it.
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
		}

	}

	return nil
}

func (cuda *Cuda) IsImageSupported(image string) (bool, error) {
	glog.Infof("Hans: cuda.IsImageSupported(): image: %s", image)
	cv, err := nvidia.GetCUDAVersion()
	if err != nil {
		return false, fmt.Errorf("Failed to detect the host cuda version(%s)", err)
	}

	isSupported, err := cuda.cudaSupported(image, cv)
	if err != nil {
		return false, fmt.Errorf("Failed to compare the cuda version of image(%s): %s", image, err)
	}

	return isSupported, nil
}

func (cuda *Cuda) GenerateDeviceOpts(gpuIdxes []int) ([]dockerClient.Device, error) {
	glog.Infof("Hans: cuda.GenerateDeviceOpts()")
	devicesOpts := []dockerClient.Device{}

	cuda.gpuInfo.Lock.RLock()
	defer cuda.gpuInfo.Lock.RUnlock()

	for _, idx := range gpuIdxes {
		var device dockerClient.Device
		device.CgroupPermissions = "rwm"
		if idx < len(cuda.gpuInfo.GPUDevices.Devices) {
			device.PathOnHost = cuda.gpuInfo.GPUDevices.Devices[idx].Path
			device.PathInContainer = cuda.gpuInfo.GPUDevices.Devices[idx].Path
			devicesOpts = append(devicesOpts, device)
		} else {
			return nil, fmt.Errorf("Cannot find the cuda gpu idx(%d) of total %d", idx, len(cuda.gpuInfo.GPUDevices.Devices))
		}

	}

	extraDevs := []string{cudaDeviceCtl, cudaDeviceUVM}
	for _, path := range extraDevs {
		var device dockerClient.Device
		device.CgroupPermissions = "rwm"
		device.PathOnHost = path
		device.PathInContainer = path
		devicesOpts = append(devicesOpts, device)
	}

	return devicesOpts, nil
}

func (cuda *Cuda) GenerateVolumeOpts(image string) ([]string, error) {
	glog.Infof("Hans: cuda.GenerateVolumeOpts()")
	result := []string{}

	// check whether the image need cuda support
	vols, err := cuda.volumesNeeded(image)
	if err != nil {
		return result, fmt.Errorf("Failed to detect whether cuda volume need or not(%s)", err)
	}
	glog.Infof("Hans: cuda.GenerateVolumeOpts(): volumesNeeded(): vols: %+v, err: %+v", vols, err)
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
					result = append(result, fmt.Sprintf("%s:%s:ro", n, vol.Mountpoint))
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

func (cuda *Cuda) cudaSupported(image string, version string) (bool, error) {
	glog.Infof("Hans: cuda.cudaSupported(): image: %s, version: %s", image, version)
	var vmaj, vmin int
	var lmaj, lmin int

	label, err := docker.Label(image, labelCUDAVersion)
	glog.Infof("Hans: cuda.cudaSupported(): get label(%s): %s, err: %s", labelCUDAVersion, label, err)
	if err != nil {
		return false, err
	}
	// if there is no label $labelCUDAVersion, it assume cuda support this image.
	if label == "" {
		return true, nil
	}
	glog.Infof("Hans: cuda.cudaSupported(): version: %s, label: %s", version, label)
	if _, err := fmt.Sscanf(version, "%d.%d", &vmaj, &vmin); err != nil {
		return false, err
	}
	if _, err := fmt.Sscanf(label, "%d.%d", &lmaj, &lmin); err != nil {
		return false, err
	}
	glog.Infof("Hans: cuda.cudaSupported(): vmaj:%d, vmin:%d", vmaj, vmin)
	glog.Infof("Hans: cuda.cudaSupported(): lmaj:%d, lmin:%d", lmaj, lmin)
	if lmaj > vmaj || (lmaj == vmaj && lmin > vmin) {
		glog.Warningf("%s", fmt.Errorf("Unsupported CUDA version: %s < %s", label, version))
		return false, nil
	}
	return true, nil
}

// caculate the current available gpus.
// Note: it is used internally, so before invoke it,
//       it must hold lock
func (cuda *Cuda) getAvailableGPUs() ([]int, error) {
	glog.Infof("Hans: cuda.getAvailableGPUs()")
	occupiedGpuSet := sets.Int{}

	// collect gpu usage information
	for _, gpuStatus := range cuda.gpuInfo.GPUStatus {
		occupiedGpuSet.Insert(gpuStatus.GPUIndexes...)
	}

	totalGpuSet := sets.Int{}
	for i := 0; i < len(cuda.gpuInfo.GPUDevices.Devices); i++ {
		totalGpuSet.Insert(int(i))
	}

	availableGpuSet := totalGpuSet.Difference(occupiedGpuSet)
	return availableGpuSet.List(), nil
}

func (cuda *Cuda) UpdateGPUUsageStatus(newGPUStatus *map[gpuTypes.PodCotainerHashID]gpuTypes.GPUUsageStatus) {
	glog.Infof("Hans: cuda.UpdateGPUUsageStatus()")

	cuda.gpuInfo.Lock.Lock()
	defer cuda.gpuInfo.Lock.Unlock()

	cuda.gpuInfo.GPUStatus = *newGPUStatus
}
