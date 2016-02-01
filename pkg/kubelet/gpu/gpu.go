package gpu

import (
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/kubelet/gpu/cuda"
	gpuTypes "k8s.io/kubernetes/pkg/kubelet/gpu/types"
)

func ProbeGPUPlugins() []gpuTypes.GPUPlugin {
	glog.Infof("Hans: ProbeGPUPlugins")
	allPlugins := []gpuTypes.GPUPlugin{}
	allPlugins = append(allPlugins, cuda.ProbeGPUPlugin())
	glog.Infof("Hans: ProbeGPUPlugins: allPlugins: %+v, len: %d", allPlugins, len(allPlugins))

	return allPlugins
}
