// +build !windows

package runconfig

import (
	"github.com/docker/docker/api/types/container"
	networktypes "github.com/docker/docker/api/types/network"
)

// ContainerConfigWrapper is a Config wrapper that holds the container Config (portable)
// and the corresponding HostConfig (non-portable).
type ContainerConfigWrapper struct {
	*container.Config
	InnerHostConfig       *container.HostConfig          `json:"HostConfig,omitempty"`
	Cpuset                string                         `json:",omitempty"` // Deprecated. Exported for backwards compatibility.
	NetworkingConfig      *networktypes.NetworkingConfig `json:"NetworkingConfig,omitempty"`
	*container.HostConfig                                // Deprecated. Exported to read attributes from json that are not in the inner host config structure.
}

// getHostConfig gets the HostConfig of the Config.
// It's mostly there to handle Deprecated fields of the ContainerConfigWrapper
func (w *ContainerConfigWrapper) getHostConfig() *container.HostConfig {
	hc := w.HostConfig

	if hc == nil && w.InnerHostConfig != nil {
		hc = w.InnerHostConfig
	} else if w.InnerHostConfig != nil {
		if hc.Memory != 0 && w.InnerHostConfig.Memory == 0 {
			w.InnerHostConfig.Memory = hc.Memory
		}
		if hc.MemorySwap != 0 && w.InnerHostConfig.MemorySwap == 0 {
			w.InnerHostConfig.MemorySwap = hc.MemorySwap
		}
		if hc.CPUShares != 0 && w.InnerHostConfig.CPUShares == 0 {
			w.InnerHostConfig.CPUShares = hc.CPUShares
		}
		if hc.CpusetCpus != "" && w.InnerHostConfig.CpusetCpus == "" {
			w.InnerHostConfig.CpusetCpus = hc.CpusetCpus
		}

		if hc.VolumeDriver != "" && w.InnerHostConfig.VolumeDriver == "" {
			w.InnerHostConfig.VolumeDriver = hc.VolumeDriver
		}

		hc = w.InnerHostConfig
	}

	if hc != nil {
		if w.Cpuset != "" && hc.CpusetCpus == "" {
			hc.CpusetCpus = w.Cpuset
		}
	}

	// Make sure NetworkMode has an acceptable value. We do this to ensure
	// backwards compatible API behavior.
	SetDefaultNetModeIfBlank(hc)

	return hc
}
