package configs

import (
	"bytes"
	"encoding/json"
	"os/exec"
)

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

// Seccomp represents syscall restrictions
// By default, only the native architecture of the kernel is allowed to be used
// for syscalls. Additional architectures can be added by specifying them in
// Architectures.
type Seccomp struct {
	DefaultAction Action     `json:"default_action"`
	Architectures []string   `json:"architectures"`
	Syscalls      []*Syscall `json:"syscalls"`
}

// An action to be taken upon rule match in Seccomp
type Action int

const (
	Kill Action = iota + 1
	Errno
	Trap
	Allow
	Trace
)

// A comparison operator to be used when matching syscall arguments in Seccomp
type Operator int

const (
	EqualTo Operator = iota + 1
	NotEqualTo
	GreaterThan
	GreaterThanOrEqualTo
	LessThan
	LessThanOrEqualTo
	MaskEqualTo
)

// A rule to match a specific syscall argument in Seccomp
type Arg struct {
	Index    uint     `json:"index"`
	Value    uint64   `json:"value"`
	ValueTwo uint64   `json:"value_two"`
	Op       Operator `json:"op"`
}

// An rule to match a syscall in Seccomp
type Syscall struct {
	Name   string `json:"name"`
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

	// Specifies the mount propagation flags to be applied to /.
	RootPropagation int `json:"rootPropagation"`

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

	// OomScoreAdj specifies the adjustment to be made by the kernel when calculating oom scores
	// for a process. Valid values are between the range [-1000, '1000'], where processes with
	// higher scores are preferred for being killed.
	// More information about kernel oom score calculation here: https://lwn.net/Articles/317814/
	OomScoreAdj int `json:"oom_score_adj"`

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

	// Sysctl is a map of properties and their values. It is the equivalent of using
	// sysctl -w my.property.name value in Linux.
	Sysctl map[string]string `json:"sysctl"`

	// Seccomp allows actions to be taken whenever a syscall is made within the container.
	// A number of rules are given, each having an action to be taken if a syscall matches it.
	// A default action to be taken if no rules match is also given.
	Seccomp *Seccomp `json:"seccomp"`

	// Hooks are a collection of actions to perform at various container lifecycle events.
	// Hooks are not able to be marshaled to json but they are also not needed to.
	Hooks *Hooks `json:"-"`

	// Version is the version of opencontainer specification that is supported.
	Version string `json:"version"`
}

type Hooks struct {
	// Prestart commands are executed after the container namespaces are created,
	// but before the user supplied command is executed from init.
	Prestart []Hook

	// Poststart commands are executed after the container init process starts.
	Poststart []Hook

	// Poststop commands are executed after the container init process exits.
	Poststop []Hook
}

// HookState is the payload provided to a hook on execution.
type HookState struct {
	Version string `json:"version"`
	ID      string `json:"id"`
	Pid     int    `json:"pid"`
	Root    string `json:"root"`
}

type Hook interface {
	// Run executes the hook with the provided state.
	Run(HookState) error
}

// NewFunctionHooks will call the provided function when the hook is run.
func NewFunctionHook(f func(HookState) error) FuncHook {
	return FuncHook{
		run: f,
	}
}

type FuncHook struct {
	run func(HookState) error
}

func (f FuncHook) Run(s HookState) error {
	return f.run(s)
}

type Command struct {
	Path string   `json:"path"`
	Args []string `json:"args"`
	Env  []string `json:"env"`
	Dir  string   `json:"dir"`
}

// NewCommandHooks will execute the provided command when the hook is run.
func NewCommandHook(cmd Command) CommandHook {
	return CommandHook{
		Command: cmd,
	}
}

type CommandHook struct {
	Command
}

func (c Command) Run(s HookState) error {
	b, err := json.Marshal(s)
	if err != nil {
		return err
	}
	cmd := exec.Cmd{
		Path:  c.Path,
		Args:  c.Args,
		Env:   c.Env,
		Stdin: bytes.NewReader(b),
	}
	return cmd.Run()
}
