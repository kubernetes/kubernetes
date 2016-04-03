package sysinfo

// SysInfo stores information about which features a kernel supports.
// TODO Windows: Factor out platform specific capabilities.
type SysInfo struct {
	AppArmor bool
	*cgroupMemInfo
	*cgroupCpuInfo
	IPv4ForwardingDisabled        bool
	BridgeNfCallIptablesDisabled  bool
	BridgeNfCallIp6tablesDisabled bool
	CgroupDevicesEnabled          bool
}

type cgroupMemInfo struct {
	MemoryLimit      bool
	SwapLimit        bool
	OomKillDisable   bool
	MemorySwappiness bool
}

type cgroupCpuInfo struct {
	CpuCfsPeriod bool
	CpuCfsQuota  bool
}
