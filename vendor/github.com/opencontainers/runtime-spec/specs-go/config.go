package specs

import "os"

// Spec is the base configuration for the container.
type Spec struct {
	// Version of the Open Container Runtime Specification with which the bundle complies.
	Version string `json:"ociVersion"`
	// Process configures the container process.
	Process *Process `json:"process,omitempty"`
	// Root configures the container's root filesystem.
	Root *Root `json:"root,omitempty"`
	// Hostname configures the container's hostname.
	Hostname string `json:"hostname,omitempty"`
	// Mounts configures additional mounts (on top of Root).
	Mounts []Mount `json:"mounts,omitempty"`
	// Hooks configures callbacks for container lifecycle events.
	Hooks *Hooks `json:"hooks,omitempty" platform:"linux,solaris"`
	// Annotations contains arbitrary metadata for the container.
	Annotations map[string]string `json:"annotations,omitempty"`

	// Linux is platform-specific configuration for Linux based containers.
	Linux *Linux `json:"linux,omitempty" platform:"linux"`
	// Solaris is platform-specific configuration for Solaris based containers.
	Solaris *Solaris `json:"solaris,omitempty" platform:"solaris"`
	// Windows is platform-specific configuration for Windows based containers.
	Windows *Windows `json:"windows,omitempty" platform:"windows"`
}

// Process contains information to start a specific application inside the container.
type Process struct {
	// Terminal creates an interactive terminal for the container.
	Terminal bool `json:"terminal,omitempty"`
	// ConsoleSize specifies the size of the console.
	ConsoleSize *Box `json:"consoleSize,omitempty"`
	// User specifies user information for the process.
	User User `json:"user"`
	// Args specifies the binary and arguments for the application to execute.
	Args []string `json:"args"`
	// Env populates the process environment for the process.
	Env []string `json:"env,omitempty"`
	// Cwd is the current working directory for the process and must be
	// relative to the container's root.
	Cwd string `json:"cwd"`
	// Capabilities are Linux capabilities that are kept for the process.
	Capabilities *LinuxCapabilities `json:"capabilities,omitempty" platform:"linux"`
	// Rlimits specifies rlimit options to apply to the process.
	Rlimits []POSIXRlimit `json:"rlimits,omitempty" platform:"linux,solaris"`
	// NoNewPrivileges controls whether additional privileges could be gained by processes in the container.
	NoNewPrivileges bool `json:"noNewPrivileges,omitempty" platform:"linux"`
	// ApparmorProfile specifies the apparmor profile for the container.
	ApparmorProfile string `json:"apparmorProfile,omitempty" platform:"linux"`
	// Specify an oom_score_adj for the container.
	OOMScoreAdj *int `json:"oomScoreAdj,omitempty" platform:"linux"`
	// SelinuxLabel specifies the selinux context that the container process is run as.
	SelinuxLabel string `json:"selinuxLabel,omitempty" platform:"linux"`
}

// LinuxCapabilities specifies the whitelist of capabilities that are kept for a process.
// http://man7.org/linux/man-pages/man7/capabilities.7.html
type LinuxCapabilities struct {
	// Bounding is the set of capabilities checked by the kernel.
	Bounding []string `json:"bounding,omitempty" platform:"linux"`
	// Effective is the set of capabilities checked by the kernel.
	Effective []string `json:"effective,omitempty" platform:"linux"`
	// Inheritable is the capabilities preserved across execve.
	Inheritable []string `json:"inheritable,omitempty" platform:"linux"`
	// Permitted is the limiting superset for effective capabilities.
	Permitted []string `json:"permitted,omitempty" platform:"linux"`
	// Ambient is the ambient set of capabilities that are kept.
	Ambient []string `json:"ambient,omitempty" platform:"linux"`
}

// Box specifies dimensions of a rectangle. Used for specifying the size of a console.
type Box struct {
	// Height is the vertical dimension of a box.
	Height uint `json:"height"`
	// Width is the horizontal dimension of a box.
	Width uint `json:"width"`
}

// User specifies specific user (and group) information for the container process.
type User struct {
	// UID is the user id.
	UID uint32 `json:"uid" platform:"linux,solaris"`
	// GID is the group id.
	GID uint32 `json:"gid" platform:"linux,solaris"`
	// AdditionalGids are additional group ids set for the container's process.
	AdditionalGids []uint32 `json:"additionalGids,omitempty" platform:"linux,solaris"`
	// Username is the user name.
	Username string `json:"username,omitempty" platform:"windows"`
}

// Root contains information about the container's root filesystem on the host.
type Root struct {
	// Path is the absolute path to the container's root filesystem.
	Path string `json:"path"`
	// Readonly makes the root filesystem for the container readonly before the process is executed.
	Readonly bool `json:"readonly,omitempty"`
}

// Mount specifies a mount for a container.
type Mount struct {
	// Destination is the absolute path where the mount will be placed in the container.
	Destination string `json:"destination"`
	// Type specifies the mount kind.
	Type string `json:"type,omitempty" platform:"linux,solaris"`
	// Source specifies the source path of the mount.
	Source string `json:"source,omitempty"`
	// Options are fstab style mount options.
	Options []string `json:"options,omitempty"`
}

// Hook specifies a command that is run at a particular event in the lifecycle of a container
type Hook struct {
	Path    string   `json:"path"`
	Args    []string `json:"args,omitempty"`
	Env     []string `json:"env,omitempty"`
	Timeout *int     `json:"timeout,omitempty"`
}

// Hooks for container setup and teardown
type Hooks struct {
	// Prestart is a list of hooks to be run before the container process is executed.
	Prestart []Hook `json:"prestart,omitempty"`
	// Poststart is a list of hooks to be run after the container process is started.
	Poststart []Hook `json:"poststart,omitempty"`
	// Poststop is a list of hooks to be run after the container process exits.
	Poststop []Hook `json:"poststop,omitempty"`
}

// Linux contains platform-specific configuration for Linux based containers.
type Linux struct {
	// UIDMapping specifies user mappings for supporting user namespaces.
	UIDMappings []LinuxIDMapping `json:"uidMappings,omitempty"`
	// GIDMapping specifies group mappings for supporting user namespaces.
	GIDMappings []LinuxIDMapping `json:"gidMappings,omitempty"`
	// Sysctl are a set of key value pairs that are set for the container on start
	Sysctl map[string]string `json:"sysctl,omitempty"`
	// Resources contain cgroup information for handling resource constraints
	// for the container
	Resources *LinuxResources `json:"resources,omitempty"`
	// CgroupsPath specifies the path to cgroups that are created and/or joined by the container.
	// The path is expected to be relative to the cgroups mountpoint.
	// If resources are specified, the cgroups at CgroupsPath will be updated based on resources.
	CgroupsPath string `json:"cgroupsPath,omitempty"`
	// Namespaces contains the namespaces that are created and/or joined by the container
	Namespaces []LinuxNamespace `json:"namespaces,omitempty"`
	// Devices are a list of device nodes that are created for the container
	Devices []LinuxDevice `json:"devices,omitempty"`
	// Seccomp specifies the seccomp security settings for the container.
	Seccomp *LinuxSeccomp `json:"seccomp,omitempty"`
	// RootfsPropagation is the rootfs mount propagation mode for the container.
	RootfsPropagation string `json:"rootfsPropagation,omitempty"`
	// MaskedPaths masks over the provided paths inside the container.
	MaskedPaths []string `json:"maskedPaths,omitempty"`
	// ReadonlyPaths sets the provided paths as RO inside the container.
	ReadonlyPaths []string `json:"readonlyPaths,omitempty"`
	// MountLabel specifies the selinux context for the mounts in the container.
	MountLabel string `json:"mountLabel,omitempty"`
	// IntelRdt contains Intel Resource Director Technology (RDT) information
	// for handling resource constraints (e.g., L3 cache) for the container
	IntelRdt *LinuxIntelRdt `json:"intelRdt,omitempty"`
}

// LinuxNamespace is the configuration for a Linux namespace
type LinuxNamespace struct {
	// Type is the type of namespace
	Type LinuxNamespaceType `json:"type"`
	// Path is a path to an existing namespace persisted on disk that can be joined
	// and is of the same type
	Path string `json:"path,omitempty"`
}

// LinuxNamespaceType is one of the Linux namespaces
type LinuxNamespaceType string

const (
	// PIDNamespace for isolating process IDs
	PIDNamespace LinuxNamespaceType = "pid"
	// NetworkNamespace for isolating network devices, stacks, ports, etc
	NetworkNamespace = "network"
	// MountNamespace for isolating mount points
	MountNamespace = "mount"
	// IPCNamespace for isolating System V IPC, POSIX message queues
	IPCNamespace = "ipc"
	// UTSNamespace for isolating hostname and NIS domain name
	UTSNamespace = "uts"
	// UserNamespace for isolating user and group IDs
	UserNamespace = "user"
	// CgroupNamespace for isolating cgroup hierarchies
	CgroupNamespace = "cgroup"
)

// LinuxIDMapping specifies UID/GID mappings
type LinuxIDMapping struct {
	// HostID is the starting UID/GID on the host to be mapped to 'ContainerID'
	HostID uint32 `json:"hostID"`
	// ContainerID is the starting UID/GID in the container
	ContainerID uint32 `json:"containerID"`
	// Size is the number of IDs to be mapped
	Size uint32 `json:"size"`
}

// POSIXRlimit type and restrictions
type POSIXRlimit struct {
	// Type of the rlimit to set
	Type string `json:"type"`
	// Hard is the hard limit for the specified type
	Hard uint64 `json:"hard"`
	// Soft is the soft limit for the specified type
	Soft uint64 `json:"soft"`
}

// LinuxHugepageLimit structure corresponds to limiting kernel hugepages
type LinuxHugepageLimit struct {
	// Pagesize is the hugepage size
	Pagesize string `json:"pageSize"`
	// Limit is the limit of "hugepagesize" hugetlb usage
	Limit uint64 `json:"limit"`
}

// LinuxInterfacePriority for network interfaces
type LinuxInterfacePriority struct {
	// Name is the name of the network interface
	Name string `json:"name"`
	// Priority for the interface
	Priority uint32 `json:"priority"`
}

// linuxBlockIODevice holds major:minor format supported in blkio cgroup
type linuxBlockIODevice struct {
	// Major is the device's major number.
	Major int64 `json:"major"`
	// Minor is the device's minor number.
	Minor int64 `json:"minor"`
}

// LinuxWeightDevice struct holds a `major:minor weight` pair for weightDevice
type LinuxWeightDevice struct {
	linuxBlockIODevice
	// Weight is the bandwidth rate for the device.
	Weight *uint16 `json:"weight,omitempty"`
	// LeafWeight is the bandwidth rate for the device while competing with the cgroup's child cgroups, CFQ scheduler only
	LeafWeight *uint16 `json:"leafWeight,omitempty"`
}

// LinuxThrottleDevice struct holds a `major:minor rate_per_second` pair
type LinuxThrottleDevice struct {
	linuxBlockIODevice
	// Rate is the IO rate limit per cgroup per device
	Rate uint64 `json:"rate"`
}

// LinuxBlockIO for Linux cgroup 'blkio' resource management
type LinuxBlockIO struct {
	// Specifies per cgroup weight
	Weight *uint16 `json:"weight,omitempty"`
	// Specifies tasks' weight in the given cgroup while competing with the cgroup's child cgroups, CFQ scheduler only
	LeafWeight *uint16 `json:"leafWeight,omitempty"`
	// Weight per cgroup per device, can override BlkioWeight
	WeightDevice []LinuxWeightDevice `json:"weightDevice,omitempty"`
	// IO read rate limit per cgroup per device, bytes per second
	ThrottleReadBpsDevice []LinuxThrottleDevice `json:"throttleReadBpsDevice,omitempty"`
	// IO write rate limit per cgroup per device, bytes per second
	ThrottleWriteBpsDevice []LinuxThrottleDevice `json:"throttleWriteBpsDevice,omitempty"`
	// IO read rate limit per cgroup per device, IO per second
	ThrottleReadIOPSDevice []LinuxThrottleDevice `json:"throttleReadIOPSDevice,omitempty"`
	// IO write rate limit per cgroup per device, IO per second
	ThrottleWriteIOPSDevice []LinuxThrottleDevice `json:"throttleWriteIOPSDevice,omitempty"`
}

// LinuxMemory for Linux cgroup 'memory' resource management
type LinuxMemory struct {
	// Memory limit (in bytes).
	Limit *int64 `json:"limit,omitempty"`
	// Memory reservation or soft_limit (in bytes).
	Reservation *int64 `json:"reservation,omitempty"`
	// Total memory limit (memory + swap).
	Swap *int64 `json:"swap,omitempty"`
	// Kernel memory limit (in bytes).
	Kernel *int64 `json:"kernel,omitempty"`
	// Kernel memory limit for tcp (in bytes)
	KernelTCP *int64 `json:"kernelTCP,omitempty"`
	// How aggressive the kernel will swap memory pages.
	Swappiness *uint64 `json:"swappiness,omitempty"`
	// DisableOOMKiller disables the OOM killer for out of memory conditions
	DisableOOMKiller *bool `json:"disableOOMKiller,omitempty"`
}

// LinuxCPU for Linux cgroup 'cpu' resource management
type LinuxCPU struct {
	// CPU shares (relative weight (ratio) vs. other cgroups with cpu shares).
	Shares *uint64 `json:"shares,omitempty"`
	// CPU hardcap limit (in usecs). Allowed cpu time in a given period.
	Quota *int64 `json:"quota,omitempty"`
	// CPU period to be used for hardcapping (in usecs).
	Period *uint64 `json:"period,omitempty"`
	// How much time realtime scheduling may use (in usecs).
	RealtimeRuntime *int64 `json:"realtimeRuntime,omitempty"`
	// CPU period to be used for realtime scheduling (in usecs).
	RealtimePeriod *uint64 `json:"realtimePeriod,omitempty"`
	// CPUs to use within the cpuset. Default is to use any CPU available.
	Cpus string `json:"cpus,omitempty"`
	// List of memory nodes in the cpuset. Default is to use any available memory node.
	Mems string `json:"mems,omitempty"`
}

// LinuxPids for Linux cgroup 'pids' resource management (Linux 4.3)
type LinuxPids struct {
	// Maximum number of PIDs. Default is "no limit".
	Limit int64 `json:"limit"`
}

// LinuxNetwork identification and priority configuration
type LinuxNetwork struct {
	// Set class identifier for container's network packets
	ClassID *uint32 `json:"classID,omitempty"`
	// Set priority of network traffic for container
	Priorities []LinuxInterfacePriority `json:"priorities,omitempty"`
}

// LinuxResources has container runtime resource constraints
type LinuxResources struct {
	// Devices configures the device whitelist.
	Devices []LinuxDeviceCgroup `json:"devices,omitempty"`
	// Memory restriction configuration
	Memory *LinuxMemory `json:"memory,omitempty"`
	// CPU resource restriction configuration
	CPU *LinuxCPU `json:"cpu,omitempty"`
	// Task resource restriction configuration.
	Pids *LinuxPids `json:"pids,omitempty"`
	// BlockIO restriction configuration
	BlockIO *LinuxBlockIO `json:"blockIO,omitempty"`
	// Hugetlb limit (in bytes)
	HugepageLimits []LinuxHugepageLimit `json:"hugepageLimits,omitempty"`
	// Network restriction configuration
	Network *LinuxNetwork `json:"network,omitempty"`
}

// LinuxDevice represents the mknod information for a Linux special device file
type LinuxDevice struct {
	// Path to the device.
	Path string `json:"path"`
	// Device type, block, char, etc.
	Type string `json:"type"`
	// Major is the device's major number.
	Major int64 `json:"major"`
	// Minor is the device's minor number.
	Minor int64 `json:"minor"`
	// FileMode permission bits for the device.
	FileMode *os.FileMode `json:"fileMode,omitempty"`
	// UID of the device.
	UID *uint32 `json:"uid,omitempty"`
	// Gid of the device.
	GID *uint32 `json:"gid,omitempty"`
}

// LinuxDeviceCgroup represents a device rule for the whitelist controller
type LinuxDeviceCgroup struct {
	// Allow or deny
	Allow bool `json:"allow"`
	// Device type, block, char, etc.
	Type string `json:"type,omitempty"`
	// Major is the device's major number.
	Major *int64 `json:"major,omitempty"`
	// Minor is the device's minor number.
	Minor *int64 `json:"minor,omitempty"`
	// Cgroup access permissions format, rwm.
	Access string `json:"access,omitempty"`
}

// Solaris contains platform-specific configuration for Solaris application containers.
type Solaris struct {
	// SMF FMRI which should go "online" before we start the container process.
	Milestone string `json:"milestone,omitempty"`
	// Maximum set of privileges any process in this container can obtain.
	LimitPriv string `json:"limitpriv,omitempty"`
	// The maximum amount of shared memory allowed for this container.
	MaxShmMemory string `json:"maxShmMemory,omitempty"`
	// Specification for automatic creation of network resources for this container.
	Anet []SolarisAnet `json:"anet,omitempty"`
	// Set limit on the amount of CPU time that can be used by container.
	CappedCPU *SolarisCappedCPU `json:"cappedCPU,omitempty"`
	// The physical and swap caps on the memory that can be used by this container.
	CappedMemory *SolarisCappedMemory `json:"cappedMemory,omitempty"`
}

// SolarisCappedCPU allows users to set limit on the amount of CPU time that can be used by container.
type SolarisCappedCPU struct {
	Ncpus string `json:"ncpus,omitempty"`
}

// SolarisCappedMemory allows users to set the physical and swap caps on the memory that can be used by this container.
type SolarisCappedMemory struct {
	Physical string `json:"physical,omitempty"`
	Swap     string `json:"swap,omitempty"`
}

// SolarisAnet provides the specification for automatic creation of network resources for this container.
type SolarisAnet struct {
	// Specify a name for the automatically created VNIC datalink.
	Linkname string `json:"linkname,omitempty"`
	// Specify the link over which the VNIC will be created.
	Lowerlink string `json:"lowerLink,omitempty"`
	// The set of IP addresses that the container can use.
	Allowedaddr string `json:"allowedAddress,omitempty"`
	// Specifies whether allowedAddress limitation is to be applied to the VNIC.
	Configallowedaddr string `json:"configureAllowedAddress,omitempty"`
	// The value of the optional default router.
	Defrouter string `json:"defrouter,omitempty"`
	// Enable one or more types of link protection.
	Linkprotection string `json:"linkProtection,omitempty"`
	// Set the VNIC's macAddress
	Macaddress string `json:"macAddress,omitempty"`
}

// Windows defines the runtime configuration for Windows based containers, including Hyper-V containers.
type Windows struct {
	// LayerFolders contains a list of absolute paths to directories containing image layers.
	LayerFolders []string `json:"layerFolders"`
	// Resources contains information for handling resource constraints for the container.
	Resources *WindowsResources `json:"resources,omitempty"`
	// CredentialSpec contains a JSON object describing a group Managed Service Account (gMSA) specification.
	CredentialSpec interface{} `json:"credentialSpec,omitempty"`
	// Servicing indicates if the container is being started in a mode to apply a Windows Update servicing operation.
	Servicing bool `json:"servicing,omitempty"`
	// IgnoreFlushesDuringBoot indicates if the container is being started in a mode where disk writes are not flushed during its boot process.
	IgnoreFlushesDuringBoot bool `json:"ignoreFlushesDuringBoot,omitempty"`
	// HyperV contains information for running a container with Hyper-V isolation.
	HyperV *WindowsHyperV `json:"hyperv,omitempty"`
	// Network restriction configuration.
	Network *WindowsNetwork `json:"network,omitempty"`
}

// WindowsResources has container runtime resource constraints for containers running on Windows.
type WindowsResources struct {
	// Memory restriction configuration.
	Memory *WindowsMemoryResources `json:"memory,omitempty"`
	// CPU resource restriction configuration.
	CPU *WindowsCPUResources `json:"cpu,omitempty"`
	// Storage restriction configuration.
	Storage *WindowsStorageResources `json:"storage,omitempty"`
}

// WindowsMemoryResources contains memory resource management settings.
type WindowsMemoryResources struct {
	// Memory limit in bytes.
	Limit *uint64 `json:"limit,omitempty"`
}

// WindowsCPUResources contains CPU resource management settings.
type WindowsCPUResources struct {
	// Number of CPUs available to the container.
	Count *uint64 `json:"count,omitempty"`
	// CPU shares (relative weight to other containers with cpu shares).
	Shares *uint16 `json:"shares,omitempty"`
	// Specifies the portion of processor cycles that this container can use as a percentage times 100.
	Maximum *uint16 `json:"maximum,omitempty"`
}

// WindowsStorageResources contains storage resource management settings.
type WindowsStorageResources struct {
	// Specifies maximum Iops for the system drive.
	Iops *uint64 `json:"iops,omitempty"`
	// Specifies maximum bytes per second for the system drive.
	Bps *uint64 `json:"bps,omitempty"`
	// Sandbox size specifies the minimum size of the system drive in bytes.
	SandboxSize *uint64 `json:"sandboxSize,omitempty"`
}

// WindowsNetwork contains network settings for Windows containers.
type WindowsNetwork struct {
	// List of HNS endpoints that the container should connect to.
	EndpointList []string `json:"endpointList,omitempty"`
	// Specifies if unqualified DNS name resolution is allowed.
	AllowUnqualifiedDNSQuery bool `json:"allowUnqualifiedDNSQuery,omitempty"`
	// Comma separated list of DNS suffixes to use for name resolution.
	DNSSearchList []string `json:"DNSSearchList,omitempty"`
	// Name (ID) of the container that we will share with the network stack.
	NetworkSharedContainerName string `json:"networkSharedContainerName,omitempty"`
}

// WindowsHyperV contains information for configuring a container to run with Hyper-V isolation.
type WindowsHyperV struct {
	// UtilityVMPath is an optional path to the image used for the Utility VM.
	UtilityVMPath string `json:"utilityVMPath,omitempty"`
}

// LinuxSeccomp represents syscall restrictions
type LinuxSeccomp struct {
	DefaultAction LinuxSeccompAction `json:"defaultAction"`
	Architectures []Arch             `json:"architectures,omitempty"`
	Syscalls      []LinuxSyscall     `json:"syscalls,omitempty"`
}

// Arch used for additional architectures
type Arch string

// Additional architectures permitted to be used for system calls
// By default only the native architecture of the kernel is permitted
const (
	ArchX86         Arch = "SCMP_ARCH_X86"
	ArchX86_64      Arch = "SCMP_ARCH_X86_64"
	ArchX32         Arch = "SCMP_ARCH_X32"
	ArchARM         Arch = "SCMP_ARCH_ARM"
	ArchAARCH64     Arch = "SCMP_ARCH_AARCH64"
	ArchMIPS        Arch = "SCMP_ARCH_MIPS"
	ArchMIPS64      Arch = "SCMP_ARCH_MIPS64"
	ArchMIPS64N32   Arch = "SCMP_ARCH_MIPS64N32"
	ArchMIPSEL      Arch = "SCMP_ARCH_MIPSEL"
	ArchMIPSEL64    Arch = "SCMP_ARCH_MIPSEL64"
	ArchMIPSEL64N32 Arch = "SCMP_ARCH_MIPSEL64N32"
	ArchPPC         Arch = "SCMP_ARCH_PPC"
	ArchPPC64       Arch = "SCMP_ARCH_PPC64"
	ArchPPC64LE     Arch = "SCMP_ARCH_PPC64LE"
	ArchS390        Arch = "SCMP_ARCH_S390"
	ArchS390X       Arch = "SCMP_ARCH_S390X"
	ArchPARISC      Arch = "SCMP_ARCH_PARISC"
	ArchPARISC64    Arch = "SCMP_ARCH_PARISC64"
)

// LinuxSeccompAction taken upon Seccomp rule match
type LinuxSeccompAction string

// Define actions for Seccomp rules
const (
	ActKill  LinuxSeccompAction = "SCMP_ACT_KILL"
	ActTrap  LinuxSeccompAction = "SCMP_ACT_TRAP"
	ActErrno LinuxSeccompAction = "SCMP_ACT_ERRNO"
	ActTrace LinuxSeccompAction = "SCMP_ACT_TRACE"
	ActAllow LinuxSeccompAction = "SCMP_ACT_ALLOW"
)

// LinuxSeccompOperator used to match syscall arguments in Seccomp
type LinuxSeccompOperator string

// Define operators for syscall arguments in Seccomp
const (
	OpNotEqual     LinuxSeccompOperator = "SCMP_CMP_NE"
	OpLessThan     LinuxSeccompOperator = "SCMP_CMP_LT"
	OpLessEqual    LinuxSeccompOperator = "SCMP_CMP_LE"
	OpEqualTo      LinuxSeccompOperator = "SCMP_CMP_EQ"
	OpGreaterEqual LinuxSeccompOperator = "SCMP_CMP_GE"
	OpGreaterThan  LinuxSeccompOperator = "SCMP_CMP_GT"
	OpMaskedEqual  LinuxSeccompOperator = "SCMP_CMP_MASKED_EQ"
)

// LinuxSeccompArg used for matching specific syscall arguments in Seccomp
type LinuxSeccompArg struct {
	Index    uint                 `json:"index"`
	Value    uint64               `json:"value"`
	ValueTwo uint64               `json:"valueTwo,omitempty"`
	Op       LinuxSeccompOperator `json:"op"`
}

// LinuxSyscall is used to match a syscall in Seccomp
type LinuxSyscall struct {
	Names  []string           `json:"names"`
	Action LinuxSeccompAction `json:"action"`
	Args   []LinuxSeccompArg  `json:"args,omitempty"`
}

// LinuxIntelRdt has container runtime resource constraints
// for Intel RDT/CAT which introduced in Linux 4.10 kernel
type LinuxIntelRdt struct {
	// The schema for L3 cache id and capacity bitmask (CBM)
	// Format: "L3:<cache_id0>=<cbm0>;<cache_id1>=<cbm1>;..."
	L3CacheSchema string `json:"l3CacheSchema,omitempty"`
}
