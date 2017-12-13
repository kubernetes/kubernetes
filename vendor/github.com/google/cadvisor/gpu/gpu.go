package gpu

import "github.com/google/cadvisor/gpu/cmd"

type GPUMonitor interface{
	Start()
	GetGPUFbSize(pid string) map[string]string
	GetGPUUtil(pid string) map[string][]string
}
func NewGPuMonitor() GPUMonitor {
	return cmd.NewGPUMonitor()
}
