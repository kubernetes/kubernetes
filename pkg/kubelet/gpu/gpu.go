package gpu

import (
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/kubelet/gpu/cuda"
	gpuTypes "k8s.io/kubernetes/pkg/kubelet/gpu/types"
)

func ProbeGPUPlugins() []gpuTypes.GPUPlugin {
	glog.Infof("Hans: ProbeGPUPlugins")
	allPlugins := []gpuTypes.GPUPlugin{}

	// add cuda plugin
	cudaPlugin := cuda.ProbeGPUPlugin()
	// if err := cudaPlugin.Init(); err == nil {
	err := cudaPlugin.Init()
	glog.Infof("Hans: ProbeGPUPlugins: Init error: %q", err)
	if err == nil {
		allPlugins = append(allPlugins, cuda.ProbeGPUPlugin())
	}
	glog.Infof("Hans: ProbeGPUPlugins: allPlugins: %+v, len: %d", allPlugins, len(allPlugins))

	return allPlugins
}
