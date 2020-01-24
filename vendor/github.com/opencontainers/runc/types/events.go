package types

// Event struct for encoding the event data to json.
type Event struct {
	Type string      `json:"type"`
	ID   string      `json:"id"`
	Data interface{} `json:"data,omitempty"`
}

// stats is the runc specific stats structure for stability when encoding and decoding stats.
type Stats struct {
	CPU               Cpu                 `json:"cpu"`
	Memory            Memory              `json:"memory"`
	Pids              Pids                `json:"pids"`
	Blkio             Blkio               `json:"blkio"`
	Hugetlb           map[string]Hugetlb  `json:"hugetlb"`
	IntelRdt          IntelRdt            `json:"intel_rdt"`
	NetworkInterfaces []*NetworkInterface `json:"network_interfaces"`
}

type Hugetlb struct {
	Usage   uint64 `json:"usage,omitempty"`
	Max     uint64 `json:"max,omitempty"`
	Failcnt uint64 `json:"failcnt"`
}

type BlkioEntry struct {
	Major uint64 `json:"major,omitempty"`
	Minor uint64 `json:"minor,omitempty"`
	Op    string `json:"op,omitempty"`
	Value uint64 `json:"value,omitempty"`
}

type Blkio struct {
	IoServiceBytesRecursive []BlkioEntry `json:"ioServiceBytesRecursive,omitempty"`
	IoServicedRecursive     []BlkioEntry `json:"ioServicedRecursive,omitempty"`
	IoQueuedRecursive       []BlkioEntry `json:"ioQueueRecursive,omitempty"`
	IoServiceTimeRecursive  []BlkioEntry `json:"ioServiceTimeRecursive,omitempty"`
	IoWaitTimeRecursive     []BlkioEntry `json:"ioWaitTimeRecursive,omitempty"`
	IoMergedRecursive       []BlkioEntry `json:"ioMergedRecursive,omitempty"`
	IoTimeRecursive         []BlkioEntry `json:"ioTimeRecursive,omitempty"`
	SectorsRecursive        []BlkioEntry `json:"sectorsRecursive,omitempty"`
}

type Pids struct {
	Current uint64 `json:"current,omitempty"`
	Limit   uint64 `json:"limit,omitempty"`
}

type Throttling struct {
	Periods          uint64 `json:"periods,omitempty"`
	ThrottledPeriods uint64 `json:"throttledPeriods,omitempty"`
	ThrottledTime    uint64 `json:"throttledTime,omitempty"`
}

type CpuUsage struct {
	// Units: nanoseconds.
	Total  uint64   `json:"total,omitempty"`
	Percpu []uint64 `json:"percpu,omitempty"`
	Kernel uint64   `json:"kernel"`
	User   uint64   `json:"user"`
}

type Cpu struct {
	Usage      CpuUsage   `json:"usage,omitempty"`
	Throttling Throttling `json:"throttling,omitempty"`
}

type MemoryEntry struct {
	Limit   uint64 `json:"limit"`
	Usage   uint64 `json:"usage,omitempty"`
	Max     uint64 `json:"max,omitempty"`
	Failcnt uint64 `json:"failcnt"`
}

type Memory struct {
	Cache     uint64            `json:"cache,omitempty"`
	Usage     MemoryEntry       `json:"usage,omitempty"`
	Swap      MemoryEntry       `json:"swap,omitempty"`
	Kernel    MemoryEntry       `json:"kernel,omitempty"`
	KernelTCP MemoryEntry       `json:"kernelTCP,omitempty"`
	Raw       map[string]uint64 `json:"raw,omitempty"`
}

type L3CacheInfo struct {
	CbmMask    string `json:"cbm_mask,omitempty"`
	MinCbmBits uint64 `json:"min_cbm_bits,omitempty"`
	NumClosids uint64 `json:"num_closids,omitempty"`
}

type MemBwInfo struct {
	BandwidthGran uint64 `json:"bandwidth_gran,omitempty"`
	DelayLinear   uint64 `json:"delay_linear,omitempty"`
	MinBandwidth  uint64 `json:"min_bandwidth,omitempty"`
	NumClosids    uint64 `json:"num_closids,omitempty"`
}

type IntelRdt struct {
	// The read-only L3 cache information
	L3CacheInfo *L3CacheInfo `json:"l3_cache_info,omitempty"`

	// The read-only L3 cache schema in root
	L3CacheSchemaRoot string `json:"l3_cache_schema_root,omitempty"`

	// The L3 cache schema in 'container_id' group
	L3CacheSchema string `json:"l3_cache_schema,omitempty"`

	// The read-only memory bandwidth information
	MemBwInfo *MemBwInfo `json:"mem_bw_info,omitempty"`

	// The read-only memory bandwidth schema in root
	MemBwSchemaRoot string `json:"mem_bw_schema_root,omitempty"`

	// The memory bandwidth schema in 'container_id' group
	MemBwSchema string `json:"mem_bw_schema,omitempty"`
}

type NetworkInterface struct {
	// Name is the name of the network interface.
	Name string

	RxBytes   uint64
	RxPackets uint64
	RxErrors  uint64
	RxDropped uint64
	TxBytes   uint64
	TxPackets uint64
	TxErrors  uint64
	TxDropped uint64
}
