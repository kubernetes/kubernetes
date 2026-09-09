package cgroups

type ThrottlingData struct {
	// Number of periods with throttling active
	Periods uint64 `json:"periods,omitzero"`
	// Number of periods when the container hit its throttling limit.
	ThrottledPeriods uint64 `json:"throttled_periods,omitzero"`
	// Aggregate time the container was throttled for in nanoseconds.
	ThrottledTime uint64 `json:"throttled_time,omitzero"`
}

type BurstData struct {
	// Number of periods bandwidth burst occurs
	BurstsPeriods uint64 `json:"bursts_periods,omitzero"`
	// Cumulative wall-time that any cpus has used above quota in respective periods
	// Units: nanoseconds.
	BurstTime uint64 `json:"burst_time,omitzero"`
}

// CpuUsage denotes the usage of a CPU.
// All CPU stats are aggregate since container inception.
type CpuUsage struct {
	// Total CPU time consumed.
	// Units: nanoseconds.
	TotalUsage uint64 `json:"total_usage,omitzero"`
	// Total CPU time consumed per core.
	// Units: nanoseconds.
	PercpuUsage []uint64 `json:"percpu_usage,omitzero"`
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

type PSIData struct {
	Avg10  float64 `json:"avg10"`
	Avg60  float64 `json:"avg60"`
	Avg300 float64 `json:"avg300"`
	Total  uint64  `json:"total"`
}

type PSIStats struct {
	Some PSIData `json:"some,omitzero"`
	Full PSIData `json:"full,omitzero"`
}

type CpuStats struct {
	CpuUsage       CpuUsage       `json:"cpu_usage,omitzero"`
	ThrottlingData ThrottlingData `json:"throttling_data,omitzero"`
	PSI            *PSIStats      `json:"psi,omitzero"`
	BurstData      BurstData      `json:"burst_data,omitzero"`
}

type CPUSetStats struct {
	// List of the physical numbers of the CPUs on which processes
	// in that cpuset are allowed to execute
	CPUs []uint16 `json:"cpus,omitzero"`
	// cpu_exclusive flag
	CPUExclusive uint64 `json:"cpu_exclusive"`
	// List of memory nodes on which processes in that cpuset
	// are allowed to allocate memory
	Mems []uint16 `json:"mems,omitzero"`
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
	Usage    uint64 `json:"usage,omitzero"`
	MaxUsage uint64 `json:"max_usage,omitzero"`
	Failcnt  uint64 `json:"failcnt"`
	Limit    uint64 `json:"limit"`
}

type MemoryStats struct {
	// memory used for cache
	Cache uint64 `json:"cache,omitzero"`
	// usage of memory
	Usage MemoryData `json:"usage,omitzero"`
	// usage of memory + swap
	SwapUsage MemoryData `json:"swap_usage,omitzero"`
	// usage of swap only
	SwapOnlyUsage MemoryData `json:"swap_only_usage,omitzero"`
	// usage of kernel memory
	KernelUsage MemoryData `json:"kernel_usage,omitzero"`
	// usage of kernel TCP memory
	KernelTCPUsage MemoryData `json:"kernel_tcp_usage,omitzero"`
	// usage of memory pages by NUMA node
	// see chapter 5.6 of memory controller documentation
	PageUsageByNUMA PageUsageByNUMA `json:"page_usage_by_numa,omitzero"`
	// if true, memory usage is accounted for throughout a hierarchy of cgroups.
	UseHierarchy bool `json:"use_hierarchy"`

	Stats map[string]uint64 `json:"stats,omitzero"`
	PSI   *PSIStats         `json:"psi,omitzero"`
}

type PageUsageByNUMA struct {
	// Embedding is used as types can't be recursive.
	PageUsageByNUMAInner
	Hierarchical PageUsageByNUMAInner `json:"hierarchical,omitzero"`
}

type PageUsageByNUMAInner struct {
	Total       PageStats `json:"total,omitzero"`
	File        PageStats `json:"file,omitzero"`
	Anon        PageStats `json:"anon,omitzero"`
	Unevictable PageStats `json:"unevictable,omitzero"`
}

type PageStats struct {
	Total uint64           `json:"total,omitzero"`
	Nodes map[uint8]uint64 `json:"nodes,omitzero"`
}

type PidsStats struct {
	// number of pids in the cgroup
	Current uint64 `json:"current,omitzero"`
	// active pids hard limit
	Limit uint64 `json:"limit,omitzero"`
}

type BlkioStatEntry struct {
	Major uint64 `json:"major,omitzero"`
	Minor uint64 `json:"minor,omitzero"`
	Op    string `json:"op,omitzero"`
	Value uint64 `json:"value,omitzero"`
}

type BlkioStats struct {
	// number of bytes transferred to and from the block device
	IoServiceBytesRecursive []BlkioStatEntry `json:"io_service_bytes_recursive,omitzero"`
	IoServicedRecursive     []BlkioStatEntry `json:"io_serviced_recursive,omitzero"`
	IoQueuedRecursive       []BlkioStatEntry `json:"io_queue_recursive,omitzero"`
	IoServiceTimeRecursive  []BlkioStatEntry `json:"io_service_time_recursive,omitzero"`
	IoWaitTimeRecursive     []BlkioStatEntry `json:"io_wait_time_recursive,omitzero"`
	IoMergedRecursive       []BlkioStatEntry `json:"io_merged_recursive,omitzero"`
	IoTimeRecursive         []BlkioStatEntry `json:"io_time_recursive,omitzero"`
	SectorsRecursive        []BlkioStatEntry `json:"sectors_recursive,omitzero"`
	PSI                     *PSIStats        `json:"psi,omitzero"`
	IoCostUsage             []BlkioStatEntry `json:"io_cost_usage,omitzero"`
	IoCostWait              []BlkioStatEntry `json:"io_cost_wait,omitzero"`
	IoCostIndebt            []BlkioStatEntry `json:"io_cost_indebt,omitzero"`
	IoCostIndelay           []BlkioStatEntry `json:"io_cost_indelay,omitzero"`
}

type HugetlbStats struct {
	// current res_counter usage for hugetlb
	Usage uint64 `json:"usage,omitzero"`
	// maximum usage ever recorded.
	MaxUsage uint64 `json:"max_usage,omitzero"`
	// number of times hugetlb usage allocation failure.
	Failcnt uint64 `json:"failcnt"`
}

type RdmaEntry struct {
	Device     string `json:"device,omitzero"`
	HcaHandles uint32 `json:"hca_handles,omitzero"`
	HcaObjects uint32 `json:"hca_objects,omitzero"`
}

type RdmaStats struct {
	RdmaLimit   []RdmaEntry `json:"rdma_limit,omitzero"`
	RdmaCurrent []RdmaEntry `json:"rdma_current,omitzero"`
}

type MiscStats struct {
	// current resource usage for a key in misc
	Usage uint64 `json:"usage,omitzero"`
	// number of times the resource usage was about to go over the max boundary
	Events uint64 `json:"events,omitzero"`
}

type Stats struct {
	CpuStats    CpuStats    `json:"cpu_stats,omitzero"`
	CPUSetStats CPUSetStats `json:"cpuset_stats,omitzero"`
	MemoryStats MemoryStats `json:"memory_stats,omitzero"`
	PidsStats   PidsStats   `json:"pids_stats,omitzero"`
	BlkioStats  BlkioStats  `json:"blkio_stats,omitzero"`
	// the map is in the format "size of hugepage: stats of the hugepage"
	HugetlbStats map[string]HugetlbStats `json:"hugetlb_stats,omitzero"`
	RdmaStats    RdmaStats               `json:"rdma_stats,omitzero"`
	// the map is in the format "misc resource name: stats of the key"
	MiscStats map[string]MiscStats `json:"misc_stats,omitzero"`
}

func NewStats() *Stats {
	memoryStats := MemoryStats{Stats: make(map[string]uint64)}
	hugetlbStats := make(map[string]HugetlbStats)
	miscStats := make(map[string]MiscStats)
	return &Stats{MemoryStats: memoryStats, HugetlbStats: hugetlbStats, MiscStats: miscStats}
}

// Controller represents a cgroup controller type for stats collection.
type Controller int

// Controller types for cgroup stats collection.
const (
	CPU Controller = 1 << iota
	Memory
	Pids
	IO
	HugeTLB
	RDMA
	Misc
	CPUSet // v1 only
)

// AllControllers is a bitmask of all available controllers.
const AllControllers = CPU | Memory | Pids | IO | HugeTLB | RDMA | Misc | CPUSet

// StatsOptions specifies which controllers to retrieve statistics for.
type StatsOptions struct {
	// Controllers is a bitmask of Controller values.
	// If 0, all available controllers are queried (default behavior).
	// Use Controller constants like: CPU | Memory | Pids
	Controllers Controller
}
