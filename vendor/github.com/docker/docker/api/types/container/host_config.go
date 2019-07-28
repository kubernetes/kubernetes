package container // import "github.com/docker/docker/api/types/container"

import (
	"strings"

	"github.com/docker/docker/api/types/blkiodev"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/api/types/strslice"
	"github.com/docker/go-connections/nat"
	"github.com/docker/go-units"
)

// Isolation represents the isolation technology of a container. The supported
// values are platform specific
type Isolation string

// IsDefault indicates the default isolation technology of a container. On Linux this
// is the native driver. On Windows, this is a Windows Server Container.
func (i Isolation) IsDefault() bool {
	return strings.ToLower(string(i)) == "default" || string(i) == ""
}

// IsHyperV indicates the use of a Hyper-V partition for isolation
func (i Isolation) IsHyperV() bool {
	return strings.ToLower(string(i)) == "hyperv"
}

// IsProcess indicates the use of process isolation
func (i Isolation) IsProcess() bool {
	return strings.ToLower(string(i)) == "process"
}

const (
	// IsolationEmpty is unspecified (same behavior as default)
	IsolationEmpty = Isolation("")
	// IsolationDefault is the default isolation mode on current daemon
	IsolationDefault = Isolation("default")
	// IsolationProcess is process isolation mode
	IsolationProcess = Isolation("process")
	// IsolationHyperV is HyperV isolation mode
	IsolationHyperV = Isolation("hyperv")
)

// IpcMode represents the container ipc stack.
type IpcMode string

// IsPrivate indicates whether the container uses its own private ipc namespace which can not be shared.
func (n IpcMode) IsPrivate() bool {
	return n == "private"
}

// IsHost indicates whether the container shares the host's ipc namespace.
func (n IpcMode) IsHost() bool {
	return n == "host"
}

// IsShareable indicates whether the container's ipc namespace can be shared with another container.
func (n IpcMode) IsShareable() bool {
	return n == "shareable"
}

// IsContainer indicates whether the container uses another container's ipc namespace.
func (n IpcMode) IsContainer() bool {
	parts := strings.SplitN(string(n), ":", 2)
	return len(parts) > 1 && parts[0] == "container"
}

// IsNone indicates whether container IpcMode is set to "none".
func (n IpcMode) IsNone() bool {
	return n == "none"
}

// IsEmpty indicates whether container IpcMode is empty
func (n IpcMode) IsEmpty() bool {
	return n == ""
}

// Valid indicates whether the ipc mode is valid.
func (n IpcMode) Valid() bool {
	return n.IsEmpty() || n.IsNone() || n.IsPrivate() || n.IsHost() || n.IsShareable() || n.IsContainer()
}

// Container returns the name of the container ipc stack is going to be used.
func (n IpcMode) Container() string {
	parts := strings.SplitN(string(n), ":", 2)
	if len(parts) > 1 && parts[0] == "container" {
		return parts[1]
	}
	return ""
}

// NetworkMode represents the container network stack.
type NetworkMode string

// IsNone indicates whether container isn't using a network stack.
func (n NetworkMode) IsNone() bool {
	return n == "none"
}

// IsDefault indicates whether container uses the default network stack.
func (n NetworkMode) IsDefault() bool {
	return n == "default"
}

// IsPrivate indicates whether container uses its private network stack.
func (n NetworkMode) IsPrivate() bool {
	return !(n.IsHost() || n.IsContainer())
}

// IsContainer indicates whether container uses a container network stack.
func (n NetworkMode) IsContainer() bool {
	parts := strings.SplitN(string(n), ":", 2)
	return len(parts) > 1 && parts[0] == "container"
}

// ConnectedContainer is the id of the container which network this container is connected to.
func (n NetworkMode) ConnectedContainer() string {
	parts := strings.SplitN(string(n), ":", 2)
	if len(parts) > 1 {
		return parts[1]
	}
	return ""
}

//UserDefined indicates user-created network
func (n NetworkMode) UserDefined() string {
	if n.IsUserDefined() {
		return string(n)
	}
	return ""
}

// UsernsMode represents userns mode in the container.
type UsernsMode string

// IsHost indicates whether the container uses the host's userns.
func (n UsernsMode) IsHost() bool {
	return n == "host"
}

// IsPrivate indicates whether the container uses the a private userns.
func (n UsernsMode) IsPrivate() bool {
	return !(n.IsHost())
}

// Valid indicates whether the userns is valid.
func (n UsernsMode) Valid() bool {
	parts := strings.Split(string(n), ":")
	switch mode := parts[0]; mode {
	case "", "host":
	default:
		return false
	}
	return true
}

// CgroupSpec represents the cgroup to use for the container.
type CgroupSpec string

// IsContainer indicates whether the container is using another container cgroup
func (c CgroupSpec) IsContainer() bool {
	parts := strings.SplitN(string(c), ":", 2)
	return len(parts) > 1 && parts[0] == "container"
}

// Valid indicates whether the cgroup spec is valid.
func (c CgroupSpec) Valid() bool {
	return c.IsContainer() || c == ""
}

// Container returns the name of the container whose cgroup will be used.
func (c CgroupSpec) Container() string {
	parts := strings.SplitN(string(c), ":", 2)
	if len(parts) > 1 {
		return parts[1]
	}
	return ""
}

// UTSMode represents the UTS namespace of the container.
type UTSMode string

// IsPrivate indicates whether the container uses its private UTS namespace.
func (n UTSMode) IsPrivate() bool {
	return !(n.IsHost())
}

// IsHost indicates whether the container uses the host's UTS namespace.
func (n UTSMode) IsHost() bool {
	return n == "host"
}

// Valid indicates whether the UTS namespace is valid.
func (n UTSMode) Valid() bool {
	parts := strings.Split(string(n), ":")
	switch mode := parts[0]; mode {
	case "", "host":
	default:
		return false
	}
	return true
}

// PidMode represents the pid namespace of the container.
type PidMode string

// IsPrivate indicates whether the container uses its own new pid namespace.
func (n PidMode) IsPrivate() bool {
	return !(n.IsHost() || n.IsContainer())
}

// IsHost indicates whether the container uses the host's pid namespace.
func (n PidMode) IsHost() bool {
	return n == "host"
}

// IsContainer indicates whether the container uses a container's pid namespace.
func (n PidMode) IsContainer() bool {
	parts := strings.SplitN(string(n), ":", 2)
	return len(parts) > 1 && parts[0] == "container"
}

// Valid indicates whether the pid namespace is valid.
func (n PidMode) Valid() bool {
	parts := strings.Split(string(n), ":")
	switch mode := parts[0]; mode {
	case "", "host":
	case "container":
		if len(parts) != 2 || parts[1] == "" {
			return false
		}
	default:
		return false
	}
	return true
}

// Container returns the name of the container whose pid namespace is going to be used.
func (n PidMode) Container() string {
	parts := strings.SplitN(string(n), ":", 2)
	if len(parts) > 1 {
		return parts[1]
	}
	return ""
}

// DeviceRequest represents a request for devices from a device driver.
// Used by GPU device drivers.
type DeviceRequest struct {
	Driver       string            // Name of device driver
	Count        int               // Number of devices to request (-1 = All)
	DeviceIDs    []string          // List of device IDs as recognizable by the device driver
	Capabilities [][]string        // An OR list of AND lists of device capabilities (e.g. "gpu")
	Options      map[string]string // Options to pass onto the device driver
}

// DeviceMapping represents the device mapping between the host and the container.
type DeviceMapping struct {
	PathOnHost        string
	PathInContainer   string
	CgroupPermissions string
}

// RestartPolicy represents the restart policies of the container.
type RestartPolicy struct {
	Name              string
	MaximumRetryCount int
}

// IsNone indicates whether the container has the "no" restart policy.
// This means the container will not automatically restart when exiting.
func (rp *RestartPolicy) IsNone() bool {
	return rp.Name == "no" || rp.Name == ""
}

// IsAlways indicates whether the container has the "always" restart policy.
// This means the container will automatically restart regardless of the exit status.
func (rp *RestartPolicy) IsAlways() bool {
	return rp.Name == "always"
}

// IsOnFailure indicates whether the container has the "on-failure" restart policy.
// This means the container will automatically restart of exiting with a non-zero exit status.
func (rp *RestartPolicy) IsOnFailure() bool {
	return rp.Name == "on-failure"
}

// IsUnlessStopped indicates whether the container has the
// "unless-stopped" restart policy. This means the container will
// automatically restart unless user has put it to stopped state.
func (rp *RestartPolicy) IsUnlessStopped() bool {
	return rp.Name == "unless-stopped"
}

// IsSame compares two RestartPolicy to see if they are the same
func (rp *RestartPolicy) IsSame(tp *RestartPolicy) bool {
	return rp.Name == tp.Name && rp.MaximumRetryCount == tp.MaximumRetryCount
}

// LogMode is a type to define the available modes for logging
// These modes affect how logs are handled when log messages start piling up.
type LogMode string

// Available logging modes
const (
	LogModeUnset            = ""
	LogModeBlocking LogMode = "blocking"
	LogModeNonBlock LogMode = "non-blocking"
)

// LogConfig represents the logging configuration of the container.
type LogConfig struct {
	Type   string
	Config map[string]string
}

// Resources contains container's resources (cgroups config, ulimits...)
type Resources struct {
	// Applicable to all platforms
	CPUShares int64 `json:"CpuShares"` // CPU shares (relative weight vs. other containers)
	Memory    int64 // Memory limit (in bytes)
	NanoCPUs  int64 `json:"NanoCpus"` // CPU quota in units of 10<sup>-9</sup> CPUs.

	// Applicable to UNIX platforms
	CgroupParent         string // Parent cgroup.
	BlkioWeight          uint16 // Block IO weight (relative weight vs. other containers)
	BlkioWeightDevice    []*blkiodev.WeightDevice
	BlkioDeviceReadBps   []*blkiodev.ThrottleDevice
	BlkioDeviceWriteBps  []*blkiodev.ThrottleDevice
	BlkioDeviceReadIOps  []*blkiodev.ThrottleDevice
	BlkioDeviceWriteIOps []*blkiodev.ThrottleDevice
	CPUPeriod            int64           `json:"CpuPeriod"`          // CPU CFS (Completely Fair Scheduler) period
	CPUQuota             int64           `json:"CpuQuota"`           // CPU CFS (Completely Fair Scheduler) quota
	CPURealtimePeriod    int64           `json:"CpuRealtimePeriod"`  // CPU real-time period
	CPURealtimeRuntime   int64           `json:"CpuRealtimeRuntime"` // CPU real-time runtime
	CpusetCpus           string          // CpusetCpus 0-2, 0,1
	CpusetMems           string          // CpusetMems 0-2, 0,1
	Devices              []DeviceMapping // List of devices to map inside the container
	DeviceCgroupRules    []string        // List of rule to be added to the device cgroup
	DeviceRequests       []DeviceRequest // List of device requests for device drivers
	DiskQuota            int64           // Disk limit (in bytes)
	KernelMemory         int64           // Kernel memory limit (in bytes)
	KernelMemoryTCP      int64           // Hard limit for kernel TCP buffer memory (in bytes)
	MemoryReservation    int64           // Memory soft limit (in bytes)
	MemorySwap           int64           // Total memory usage (memory + swap); set `-1` to enable unlimited swap
	MemorySwappiness     *int64          // Tuning container memory swappiness behaviour
	OomKillDisable       *bool           // Whether to disable OOM Killer or not
	PidsLimit            *int64          // Setting PIDs limit for a container; Set `0` or `-1` for unlimited, or `null` to not change.
	Ulimits              []*units.Ulimit // List of ulimits to be set in the container

	// Applicable to Windows
	CPUCount           int64  `json:"CpuCount"`   // CPU count
	CPUPercent         int64  `json:"CpuPercent"` // CPU percent
	IOMaximumIOps      uint64 // Maximum IOps for the container system drive
	IOMaximumBandwidth uint64 // Maximum IO in bytes per second for the container system drive
}

// UpdateConfig holds the mutable attributes of a Container.
// Those attributes can be updated at runtime.
type UpdateConfig struct {
	// Contains container's resources (cgroups, ulimits)
	Resources
	RestartPolicy RestartPolicy
}

// HostConfig the non-portable Config structure of a container.
// Here, "non-portable" means "dependent of the host we are running on".
// Portable information *should* appear in Config.
type HostConfig struct {
	// Applicable to all platforms
	Binds           []string      // List of volume bindings for this container
	ContainerIDFile string        // File (path) where the containerId is written
	LogConfig       LogConfig     // Configuration of the logs for this container
	NetworkMode     NetworkMode   // Network mode to use for the container
	PortBindings    nat.PortMap   // Port mapping between the exposed port (container) and the host
	RestartPolicy   RestartPolicy // Restart policy to be used for the container
	AutoRemove      bool          // Automatically remove container when it exits
	VolumeDriver    string        // Name of the volume driver used to mount volumes
	VolumesFrom     []string      // List of volumes to take from other container

	// Applicable to UNIX platforms
	CapAdd          strslice.StrSlice // List of kernel capabilities to add to the container
	CapDrop         strslice.StrSlice // List of kernel capabilities to remove from the container
	Capabilities    []string          `json:"Capabilities"` // List of kernel capabilities to be available for container (this overrides the default set)
	DNS             []string          `json:"Dns"`          // List of DNS server to lookup
	DNSOptions      []string          `json:"DnsOptions"`   // List of DNSOption to look for
	DNSSearch       []string          `json:"DnsSearch"`    // List of DNSSearch to look for
	ExtraHosts      []string          // List of extra hosts
	GroupAdd        []string          // List of additional groups that the container process will run as
	IpcMode         IpcMode           // IPC namespace to use for the container
	Cgroup          CgroupSpec        // Cgroup to use for the container
	Links           []string          // List of links (in the name:alias form)
	OomScoreAdj     int               // Container preference for OOM-killing
	PidMode         PidMode           // PID namespace to use for the container
	Privileged      bool              // Is the container in privileged mode
	PublishAllPorts bool              // Should docker publish all exposed port for the container
	ReadonlyRootfs  bool              // Is the container root filesystem in read-only
	SecurityOpt     []string          // List of string values to customize labels for MLS systems, such as SELinux.
	StorageOpt      map[string]string `json:",omitempty"` // Storage driver options per container.
	Tmpfs           map[string]string `json:",omitempty"` // List of tmpfs (mounts) used for the container
	UTSMode         UTSMode           // UTS namespace to use for the container
	UsernsMode      UsernsMode        // The user namespace to use for the container
	ShmSize         int64             // Total shm memory usage
	Sysctls         map[string]string `json:",omitempty"` // List of Namespaced sysctls used for the container
	Runtime         string            `json:",omitempty"` // Runtime to use with this container

	// Applicable to Windows
	ConsoleSize [2]uint   // Initial console size (height,width)
	Isolation   Isolation // Isolation technology of the container (e.g. default, hyperv)

	// Contains container's resources (cgroups, ulimits)
	Resources

	// Mounts specs used by the container
	Mounts []mount.Mount `json:",omitempty"`

	// MaskedPaths is the list of paths to be masked inside the container (this overrides the default set of paths)
	MaskedPaths []string

	// ReadonlyPaths is the list of paths to be set as read-only inside the container (this overrides the default set of paths)
	ReadonlyPaths []string

	// Run a custom init inside the container, if null, use the daemon's configured settings
	Init *bool `json:",omitempty"`
}
