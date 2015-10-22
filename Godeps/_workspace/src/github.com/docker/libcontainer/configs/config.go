package configs

type Rlimit struct {
	Type int    `json:"type"`
	Hard uint64 `json:"hard"`
	Soft uint64 `json:"soft"`
}

// IDMap represents UID/GID Mappings for User Namespaces.
type IDMap struct {
	ContainerID int `json:"container_id"`
	HostID      int `json:"host_id"`
	Size        int `json:"size"`
}

type Seccomp struct {
	Syscalls []*Syscall `json:"syscalls"`
}

type Action int

const (
	Kill Action = iota - 3
	Trap
	Allow
)

type Operator int

const (
	EqualTo Operator = iota
	NotEqualTo
	GreatherThan
	LessThan
	MaskEqualTo
)

type Arg struct {
	Index int      `json:"index"`
	Value uint32   `json:"value"`
	Op    Operator `json:"op"`
}

type Syscall struct {
	Value  int    `json:"value"`
	Action Action `json:"action"`
	Args   []*Arg `json:"args"`
}

// TODO Windows. Many of these fields should be factored out into those parts
// which are common across platforms, and those which are platform specific.

// Config defines configuration options for executing a process inside a contained environment.
type Config struct {
	// NoPivotRoot will use MS_MOVE and a chroot to jail the process into the container's rootfs
	// This is a common option when the container is running in ramdisk
	NoPivotRoot bool `json:"no_pivot_root"`

	// ParentDeathSignal specifies the signal that is sent to the container's process in the case
	// that the parent process dies.
	ParentDeathSignal int `json:"parent_death_signal"`

	// PivotDir allows a custom directory inside the container's root filesystem to be used as pivot, when NoPivotRoot is not set.
	// When a custom PivotDir not set, a temporary dir inside the root filesystem will be used. The pivot dir needs to be writeable.
	// This is required when using read only root filesystems. In these cases, a read/writeable path can be (bind) mounted somewhere inside the root filesystem to act as pivot.
	PivotDir string `json:"pivot_dir"`

	// Path to a directory containing the container's root filesystem.
	Rootfs string `json:"rootfs"`

	// Readonlyfs will remount the container's rootfs as readonly where only externally mounted
	// bind mounts are writtable.
	Readonlyfs bool `json:"readonlyfs"`

	// Privatefs will mount the container's rootfs as private where mount points from the parent will not propogate
	Privatefs bool `json:"privatefs"`

	// Mounts specify additional source and destination paths that will be mounted inside the container's
	// rootfs and mount namespace if specified
	Mounts []*Mount `json:"mounts"`

	// The device nodes that should be automatically created within the container upon container start.  Note, make sure that the node is marked as allowed in the cgroup as well!
	Devices []*Device `json:"devices"`

	MountLabel string `json:"mount_label"`

	// Hostname optionally sets the container's hostname if provided
	Hostname string `json:"hostname"`

	// Namespaces specifies the container's namespaces that it should setup when cloning the init process
	// If a namespace is not provided that namespace is shared from the container's parent process
	Namespaces Namespaces `json:"namespaces"`

	// Capabilities specify the capabilities to keep when executing the process inside the container
	// All capbilities not specified will be dropped from the processes capability mask
	Capabilities []string `json:"capabilities"`

	// Networks specifies the container's network setup to be created
	Networks []*Network `json:"networks"`

	// Routes can be specified to create entries in the route table as the container is started
	Routes []*Route `json:"routes"`

	// Cgroups specifies specific cgroup settings for the various subsystems that the container is
	// placed into to limit the resources the container has available
	Cgroups *Cgroup `json:"cgroups"`

	// AppArmorProfile specifies the profile to apply to the process running in the container and is
	// change at the time the process is execed
	AppArmorProfile string `json:"apparmor_profile"`

	// ProcessLabel specifies the label to apply to the process running in the container.  It is
	// commonly used by selinux
	ProcessLabel string `json:"process_label"`

	// Rlimits specifies the resource limits, such as max open files, to set in the container
	// If Rlimits are not set, the container will inherit rlimits from the parent process
	Rlimits []Rlimit `json:"rlimits"`

	// AdditionalGroups specifies the gids that should be added to supplementary groups
	// in addition to those that the user belongs to.
	AdditionalGroups []string `json:"additional_groups"`

	// UidMappings is an array of User ID mappings for User Namespaces
	UidMappings []IDMap `json:"uid_mappings"`

	// GidMappings is an array of Group ID mappings for User Namespaces
	GidMappings []IDMap `json:"gid_mappings"`

	// MaskPaths specifies paths within the container's rootfs to mask over with a bind
	// mount pointing to /dev/null as to prevent reads of the file.
	MaskPaths []string `json:"mask_paths"`

	// ReadonlyPaths specifies paths within the container's rootfs to remount as read-only
	// so that these files prevent any writes.
	ReadonlyPaths []string `json:"readonly_paths"`

	// SystemProperties is a map of properties and their values. It is the equivalent of using
	// sysctl -w my.property.name value in Linux.
	SystemProperties map[string]string `json:"system_properties"`

	// Seccomp allows actions to be taken whenever a syscall is made within the container.
	// By default, all syscalls are allowed with actions to allow, trap, kill, or return an errno
	// can be specified on a per syscall basis.
	Seccomp *Seccomp `json:"seccomp"`
}
