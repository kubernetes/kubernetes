package cgroups

type ThrottlingData struct {
	// Number of periods with throttling active
	Periods uint64 `json:"periods,omitempty"`
	// Number of periods when the container hit its throttling limit.
	ThrottledPeriods uint64 `json:"throttled_periods,omitempty"`
	// Aggregate time the container was throttled for in nanoseconds.
	ThrottledTime uint64 `json:"throttled_time,omitempty"`
}

// CpuUsage denotes the usage of a CPU.
// All CPU stats are aggregate since container inception.
type CpuUsage struct {
	// Total CPU time consumed.
	// Units: nanoseconds.
	TotalUsage uint64 `json:"total_usage,omitempty"`
	// Total CPU time consumed per core.
	// Units: nanoseconds.
	PercpuUsage []uint64 `json:"percpu_usage,omitempty"`
	// CPU time consumed per core in kernel mode
	// Units: nanoseconds.
	PercpuUsageInKernelmode []uint64 `json:"percpu_usage_in_kernelmode"`
	// CPU time consumed per core in user mode
	// Units: nanoseconds.
	PercpuUsageInUsermode []uint64 `json:"percpu_usage_in_usermode"`
	// Time spent by tasks of the cgroup in kernel mode.
	// Units: nanoseconds.
	UsageInKernelmode uint64 `json:"usage_in_kernelmode"`
	// Time spent by tasks of the cgroup in user mode.
	// Units: nanoseconds.
	UsageInUsermode uint64 `json:"usage_in_usermode"`
}

type CpuStats struct {
	CpuUsage       CpuUsage       `json:"cpu_usage,omitempty"`
	ThrottlingData ThrottlingData `json:"throttling_data,omitempty"`
}

type CPUSetStats struct {
	// List of the physical numbers of the CPUs on which processes
	// in that cpuset are allowed to execute
	CPUs []uint16 `json:"cpus,omitempty"`
	// cpu_exclusive flag
	CPUExclusive uint64 `json:"cpu_exclusive"`
	// List of memory nodes on which processes in that cpuset
	// are allowed to allocate memory
	Mems []uint16 `json:"mems,omitempty"`
	// mem_hardwall flag
	MemHardwall uint64 `json:"mem_hardwall"`
	// mem_exclusive flag
	MemExclusive uint64 `json:"mem_exclusive"`
	// memory_migrate flag
	MemoryMigrate uint64 `json:"memory_migrate"`
	// memory_spread page flag
	MemorySpreadPage uint64 `json:"memory_spread_page"`
	// memory_spread slab flag
	MemorySpreadSlab uint64 `json:"memory_spread_slab"`
	// memory_pressure
	MemoryPressure uint64 `json:"memory_pressure"`
	// sched_load balance flag
	SchedLoadBalance uint64 `json:"sched_load_balance"`
	// sched_relax_domain_level
	SchedRelaxDomainLevel int64 `json:"sched_relax_domain_level"`
}

type MemoryData struct {
	Usage    uint64 `json:"usage,omitempty"`
	MaxUsage uint64 `json:"max_usage,omitempty"`
	Failcnt  uint64 `json:"failcnt"`
	Limit    uint64 `json:"limit"`
}

type MemoryStats struct {
	// memory used for cache
	Cache uint64 `json:"cache,omitempty"`
	// usage of memory
	Usage MemoryData `json:"usage,omitempty"`
	// usage of memory + swap
	SwapUsage MemoryData `json:"swap_usage,omitempty"`
	// usage of kernel memory
	KernelUsage MemoryData `json:"kernel_usage,omitempty"`
	// usage of kernel TCP memory
	KernelTCPUsage MemoryData `json:"kernel_tcp_usage,omitempty"`
	// usage of memory pages by NUMA node
	// see chapter 5.6 of memory controller documentation
	PageUsageByNUMA PageUsageByNUMA `json:"page_usage_by_numa,omitempty"`
	// if true, memory usage is accounted for throughout a hierarchy of cgroups.
	UseHierarchy bool `json:"use_hierarchy"`

	Stats map[string]uint64 `json:"stats,omitempty"`
}

type PageUsageByNUMA struct {
	// Embedding is used as types can't be recursive.
	PageUsageByNUMAInner
	Hierarchical PageUsageByNUMAInner `json:"hierarchical,omitempty"`
}

type PageUsageByNUMAInner struct {
	Total       PageStats `json:"total,omitempty"`
	File        PageStats `json:"file,omitempty"`
	Anon        PageStats `json:"anon,omitempty"`
	Unevictable PageStats `json:"unevictable,omitempty"`
}

type PageStats struct {
	Total uint64           `json:"total,omitempty"`
	Nodes map[uint8]uint64 `json:"nodes,omitempty"`
}

type PidsStats struct {
	// number of pids in the cgroup
	Current uint64 `json:"current,omitempty"`
	// active pids hard limit
	Limit uint64 `json:"limit,omitempty"`
}

type BlkioStatEntry struct {
	Major uint64 `json:"major,omitempty"`
	Minor uint64 `json:"minor,omitempty"`
	Op    string `json:"op,omitempty"`
	Value uint64 `json:"value,omitempty"`
}

type BlkioStats struct {
	// number of bytes transferred to and from the block device
	IoServiceBytesRecursive []BlkioStatEntry `json:"io_service_bytes_recursive,omitempty"`
	IoServicedRecursive     []BlkioStatEntry `json:"io_serviced_recursive,omitempty"`
	IoQueuedRecursive       []BlkioStatEntry `json:"io_queue_recursive,omitempty"`
	IoServiceTimeRecursive  []BlkioStatEntry `json:"io_service_time_recursive,omitempty"`
	IoWaitTimeRecursive     []BlkioStatEntry `json:"io_wait_time_recursive,omitempty"`
	IoMergedRecursive       []BlkioStatEntry `json:"io_merged_recursive,omitempty"`
	IoTimeRecursive         []BlkioStatEntry `json:"io_time_recursive,omitempty"`
	SectorsRecursive        []BlkioStatEntry `json:"sectors_recursive,omitempty"`
}

type HugetlbStats struct {
	// current res_counter usage for hugetlb
	Usage uint64 `json:"usage,omitempty"`
	// maximum usage ever recorded.
	MaxUsage uint64 `json:"max_usage,omitempty"`
	// number of times hugetlb usage allocation failure.
	Failcnt uint64 `json:"failcnt"`
}

type RdmaEntry struct {
	Device     string `json:"device,omitempty"`
	HcaHandles uint32 `json:"hca_handles,omitempty"`
	HcaObjects uint32 `json:"hca_objects,omitempty"`
}

type RdmaStats struct {
	RdmaLimit   []RdmaEntry `json:"rdma_limit,omitempty"`
	RdmaCurrent []RdmaEntry `json:"rdma_current,omitempty"`
}

type Stats struct {
	CpuStats    CpuStats    `json:"cpu_stats,omitempty"`
	CPUSetStats CPUSetStats `json:"cpuset_stats,omitempty"`
	MemoryStats MemoryStats `json:"memory_stats,omitempty"`
	PidsStats   PidsStats   `json:"pids_stats,omitempty"`
	BlkioStats  BlkioStats  `json:"blkio_stats,omitempty"`
	// the map is in the format "size of hugepage: stats of the hugepage"
	HugetlbStats map[string]HugetlbStats `json:"hugetlb_stats,omitempty"`
	RdmaStats    RdmaStats               `json:"rdma_stats,omitempty"`
}

func NewStats() *Stats {
	memoryStats := MemoryStats{Stats: make(map[string]uint64)}
	hugetlbStats := make(map[string]HugetlbStats)
	return &Stats{MemoryStats: memoryStats, HugetlbStats: hugetlbStats}
}
