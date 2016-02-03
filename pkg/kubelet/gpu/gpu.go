package gpu

import (
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/kubelet/gpu/cuda"
	gpuTypes "k8s.io/kubernetes/pkg/kubelet/gpu/types"
)

func ProbeGPUPlugins() []*gpuTypes.GPUPlugin {
	glog.Infof("Hans: ProbeGPUPlugins")
	allPlugins := []*gpuTypes.GPUPlugin{}

	// add cuda plugin
	cudaPlugin := cuda.ProbeGPUPlugin()

	if err := cudaPlugin.InitPlugin(); err == nil {
		allPlugins = append(allPlugins, cuda.ProbeGPUPlugin())
	} else {
		glog.Infof("Init cuda Plugin failed: %q", err)
	}

	glog.Infof("Hans: ProbeGPUPlugins: allPlugins: %+v, len: %d", allPlugins, len(allPlugins))

	return allPlugins
}

func IsGPUAvaiable(pods []*api.Pod, gpuCapacity int) bool {
	totalGPU := gpuCapacity
	totalGPURequest := int(0)

	for _, pod := range pods {
		totalGPURequest += getGPUResourceRequest(pod)
	}

	return totalGPURequest == 0 || (totalGPU-totalGPURequest) >= 0
}

func getGPUResourceRequest(pod *api.Pod) int {
	gpuReqNum := 0
	for _, container := range pod.Spec.Containers {
		requests := container.Resources.Requests
		gpuReqNum += requests.Gpu().MilliValue()
	}
	return gpuReqNum
}
