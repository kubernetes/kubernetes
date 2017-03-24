package cputopology

import (
	"fmt"
	"sync"
)

// CPU struct contains information about logical CPU.
type CPU struct {
	// SocketID represents socket where CPU is.
	SocketID int
	// CoreID represents core where CPU is.
	CoreID int
	// CPUID is unique CPU number.
	CPUID int
	// IsIsolated stores information is CPU is not use by OS scheduler.
	IsIsolated bool
	// IsInUse stores information is CPU already reserved.
	IsInUse bool
}

type CPUTopology struct {
	CPU  []CPU
	Lock sync.Mutex
}

func (c *CPUTopology) GetNumSockets() int {
	socketCounter := 0
	for _, cpu := range c.CPU {
		if cpu.SocketID > socketCounter {
			socketCounter = cpu.SocketID
		}
	}
	return socketCounter + 1
}

func (c *CPUTopology) GetSocket(socketID int) (cores []CPU) {
	for _, cpu := range c.CPU {
		if cpu.SocketID == socketID {
			cores = append(cores, cpu)
		}
	}
	return
}

func (c *CPUTopology) GetCore(socketID, coreID int) (cpus []CPU) {
	for _, cpu := range c.CPU {
		if cpu.SocketID == socketID && cpu.CoreID == coreID {
			cpus = append(cpus, cpu)
		}
	}
	return
}

func (c *CPUTopology) GetCPU(cpuid int) *CPU {
	for idx, _ := range c.CPU {
		if c.CPU[idx].CPUID == cpuid {
			return &c.CPU[idx]
		}
	}
	return nil
}

func (c *CPUTopology) Reserve(cpuid int) error {
	c.Lock.Lock()
	defer c.Lock.Unlock()
	cpuAddress := c.GetCPU(cpuid)
	if cpuAddress.IsInUse {
		return fmt.Errorf("selected core(%d) is in use.", cpuid)
	}
	cpuAddress.IsInUse = true
	return nil
}

func (c *CPUTopology) Reclaim(cpuid int) {
	c.Lock.Lock()
	defer c.Lock.Unlock()
	cpuAddress := c.GetCPU(cpuid)
	cpuAddress.IsInUse = false
}
