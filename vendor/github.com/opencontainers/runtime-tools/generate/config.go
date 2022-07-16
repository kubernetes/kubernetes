package generate

import (
	rspec "github.com/opencontainers/runtime-spec/specs-go"
)

func (g *Generator) initConfig() {
	if g.Config == nil {
		g.Config = &rspec.Spec{}
	}
}

func (g *Generator) initConfigProcess() {
	g.initConfig()
	if g.Config.Process == nil {
		g.Config.Process = &rspec.Process{}
	}
}

func (g *Generator) initConfigProcessConsoleSize() {
	g.initConfigProcess()
	if g.Config.Process.ConsoleSize == nil {
		g.Config.Process.ConsoleSize = &rspec.Box{}
	}
}

func (g *Generator) initConfigProcessCapabilities() {
	g.initConfigProcess()
	if g.Config.Process.Capabilities == nil {
		g.Config.Process.Capabilities = &rspec.LinuxCapabilities{}
	}
}

func (g *Generator) initConfigRoot() {
	g.initConfig()
	if g.Config.Root == nil {
		g.Config.Root = &rspec.Root{}
	}
}

func (g *Generator) initConfigAnnotations() {
	g.initConfig()
	if g.Config.Annotations == nil {
		g.Config.Annotations = make(map[string]string)
	}
}

func (g *Generator) initConfigHooks() {
	g.initConfig()
	if g.Config.Hooks == nil {
		g.Config.Hooks = &rspec.Hooks{}
	}
}

func (g *Generator) initConfigLinux() {
	g.initConfig()
	if g.Config.Linux == nil {
		g.Config.Linux = &rspec.Linux{}
	}
}

func (g *Generator) initConfigLinuxIntelRdt() {
	g.initConfigLinux()
	if g.Config.Linux.IntelRdt == nil {
		g.Config.Linux.IntelRdt = &rspec.LinuxIntelRdt{}
	}
}

func (g *Generator) initConfigLinuxSysctl() {
	g.initConfigLinux()
	if g.Config.Linux.Sysctl == nil {
		g.Config.Linux.Sysctl = make(map[string]string)
	}
}

func (g *Generator) initConfigLinuxSeccomp() {
	g.initConfigLinux()
	if g.Config.Linux.Seccomp == nil {
		g.Config.Linux.Seccomp = &rspec.LinuxSeccomp{}
	}
}

func (g *Generator) initConfigLinuxResources() {
	g.initConfigLinux()
	if g.Config.Linux.Resources == nil {
		g.Config.Linux.Resources = &rspec.LinuxResources{}
	}
}

func (g *Generator) initConfigLinuxResourcesBlockIO() {
	g.initConfigLinuxResources()
	if g.Config.Linux.Resources.BlockIO == nil {
		g.Config.Linux.Resources.BlockIO = &rspec.LinuxBlockIO{}
	}
}

// InitConfigLinuxResourcesCPU initializes CPU of Linux resources
func (g *Generator) InitConfigLinuxResourcesCPU() {
	g.initConfigLinuxResources()
	if g.Config.Linux.Resources.CPU == nil {
		g.Config.Linux.Resources.CPU = &rspec.LinuxCPU{}
	}
}

func (g *Generator) initConfigLinuxResourcesMemory() {
	g.initConfigLinuxResources()
	if g.Config.Linux.Resources.Memory == nil {
		g.Config.Linux.Resources.Memory = &rspec.LinuxMemory{}
	}
}

func (g *Generator) initConfigLinuxResourcesNetwork() {
	g.initConfigLinuxResources()
	if g.Config.Linux.Resources.Network == nil {
		g.Config.Linux.Resources.Network = &rspec.LinuxNetwork{}
	}
}

func (g *Generator) initConfigLinuxResourcesPids() {
	g.initConfigLinuxResources()
	if g.Config.Linux.Resources.Pids == nil {
		g.Config.Linux.Resources.Pids = &rspec.LinuxPids{}
	}
}

func (g *Generator) initConfigSolaris() {
	g.initConfig()
	if g.Config.Solaris == nil {
		g.Config.Solaris = &rspec.Solaris{}
	}
}

func (g *Generator) initConfigSolarisCappedCPU() {
	g.initConfigSolaris()
	if g.Config.Solaris.CappedCPU == nil {
		g.Config.Solaris.CappedCPU = &rspec.SolarisCappedCPU{}
	}
}

func (g *Generator) initConfigSolarisCappedMemory() {
	g.initConfigSolaris()
	if g.Config.Solaris.CappedMemory == nil {
		g.Config.Solaris.CappedMemory = &rspec.SolarisCappedMemory{}
	}
}

func (g *Generator) initConfigWindows() {
	g.initConfig()
	if g.Config.Windows == nil {
		g.Config.Windows = &rspec.Windows{}
	}
}

func (g *Generator) initConfigWindowsNetwork() {
	g.initConfigWindows()
	if g.Config.Windows.Network == nil {
		g.Config.Windows.Network = &rspec.WindowsNetwork{}
	}
}

func (g *Generator) initConfigWindowsHyperV() {
	g.initConfigWindows()
	if g.Config.Windows.HyperV == nil {
		g.Config.Windows.HyperV = &rspec.WindowsHyperV{}
	}
}

func (g *Generator) initConfigWindowsResources() {
	g.initConfigWindows()
	if g.Config.Windows.Resources == nil {
		g.Config.Windows.Resources = &rspec.WindowsResources{}
	}
}

func (g *Generator) initConfigWindowsResourcesMemory() {
	g.initConfigWindowsResources()
	if g.Config.Windows.Resources.Memory == nil {
		g.Config.Windows.Resources.Memory = &rspec.WindowsMemoryResources{}
	}
}

func (g *Generator) initConfigVM() {
	g.initConfig()
	if g.Config.VM == nil {
		g.Config.VM = &rspec.VM{}
	}
}

func (g *Generator) initConfigVMHypervisor() {
	g.initConfigVM()
	if &g.Config.VM.Hypervisor == nil {
		g.Config.VM.Hypervisor = rspec.VMHypervisor{}
	}
}

func (g *Generator) initConfigVMKernel() {
	g.initConfigVM()
	if &g.Config.VM.Kernel == nil {
		g.Config.VM.Kernel = rspec.VMKernel{}
	}
}

func (g *Generator) initConfigVMImage() {
	g.initConfigVM()
	if &g.Config.VM.Image == nil {
		g.Config.VM.Image = rspec.VMImage{}
	}
}
