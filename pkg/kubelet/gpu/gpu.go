package gpu

import (
	"k8s.io/kubernetes/pkg/kubelet/gpu/cuda"
	gpuTypes "k8s.io/kubernetes/pkg/kubelet/gpu/types"
)

func ProbeGPUPlugins() []gpuTypes.GPUPlugin {
	allPlugins := []gpuTypes.GPUPlugin{}
	allPlugins = append(allPlugins, cuda.ProbeGPUPlugin())

	return allPlugins
}
