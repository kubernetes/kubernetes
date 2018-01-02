// +build linux

package daemon

import (
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/libcontainerd"
)

func toContainerdResources(resources container.Resources) libcontainerd.Resources {
	var r libcontainerd.Resources
	r.BlkioWeight = uint64(resources.BlkioWeight)
	r.CpuShares = uint64(resources.CPUShares)
	if resources.NanoCPUs != 0 {
		r.CpuPeriod = uint64(100 * time.Millisecond / time.Microsecond)
		r.CpuQuota = uint64(resources.NanoCPUs) * r.CpuPeriod / 1e9
	} else {
		r.CpuPeriod = uint64(resources.CPUPeriod)
		r.CpuQuota = uint64(resources.CPUQuota)
	}
	r.CpusetCpus = resources.CpusetCpus
	r.CpusetMems = resources.CpusetMems
	r.MemoryLimit = uint64(resources.Memory)
	if resources.MemorySwap > 0 {
		r.MemorySwap = uint64(resources.MemorySwap)
	}
	r.MemoryReservation = uint64(resources.MemoryReservation)
	r.KernelMemoryLimit = uint64(resources.KernelMemory)
	return r
}
