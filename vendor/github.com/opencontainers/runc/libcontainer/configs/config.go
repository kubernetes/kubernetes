package configs

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"time"

	"github.com/opencontainers/runtime-spec/specs-go"

	"github.com/sirupsen/logrus"
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

// Action is taken upon rule match in Seccomp
type Action int

const (
	Kill Action = iota + 1
	Errno
	Trap
	Allow
	Trace
)

// Operator is a comparison operator to be used when matching syscall arguments in Seccomp
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

// Arg is a rule to match a specific syscall argument in Seccomp
type Arg struct {
	Index    uint     `json:"index"`
	Value    uint64   `json:"value"`
	ValueTwo uint64   `json:"value_two"`
	Op       Operator `json:"op"`
}

// Syscall is a rule to match a syscall in Seccomp
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
	// All capabilities not specified will be dropped from the processes capability mask
	Capabilities *Capabilities `json:"capabilities"`

	// Networks specifies the container's network setup to be created
	Networks []*Network `json:"networks"`

	// Routes can be specified to create entries in the route table as the container is started
	Routes []*Route `json:"routes"`

	// Cgroups specifies specific cgroup settings for the various subsystems that the container is
	// placed into to limit the resources the container has available
	Cgroups *Cgroup `json:"cgroups"`

	// AppArmorProfile specifies the profile to apply to the process running in the container and is
	// change at the time the process is execed
	AppArmorProfile string `json:"apparmor_profile,omitempty"`

	// ProcessLabel specifies the label to apply to the process running in the container.  It is
	// commonly used by selinux
	ProcessLabel string `json:"process_label,omitempty"`

	// Rlimits specifies the resource limits, such as max open files, to set in the container
	// If Rlimits are not set, the container will inherit rlimits from the parent process
	Rlimits []Rlimit `json:"rlimits,omitempty"`

	// OomScoreAdj specifies the adjustment to be made by the kernel when calculating oom scores
	// for a process. Valid values are between the range [-1000, '1000'], where processes with
	// higher scores are preferred for being killed.
	// More information about kernel oom score calculation here: https://lwn.net/Articles/317814/
	OomScoreAdj int `json:"oom_score_adj"`

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

	// NoNewPrivileges controls whether processes in the container can gain additional privileges.
	NoNewPrivileges bool `json:"no_new_privileges,omitempty"`

	// Hooks are a collection of actions to perform at various container lifecycle events.
	// CommandHooks are serialized to JSON, but other hooks are not.
	Hooks *Hooks

	// Version is the version of opencontainer specification that is supported.
	Version string `json:"version"`

	// Labels are user defined metadata that is stored in the config and populated on the state
	Labels []string `json:"labels"`

	// NoNewKeyring will not allocated a new session keyring for the container.  It will use the
	// callers keyring in this case.
	NoNewKeyring bool `json:"no_new_keyring"`

	// Rootless specifies whether the container is a rootless container.
	Rootless bool `json:"rootless"`
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

type Capabilities struct {
	// Bounding is the set of capabilities checked by the kernel.
	Bounding []string
	// Effective is the set of capabilities checked by the kernel.
	Effective []string
	// Inheritable is the capabilities preserved across execve.
	Inheritable []string
	// Permitted is the limiting superset for effective capabilities.
	Permitted []string
	// Ambient is the ambient set of capabilities that are kept.
	Ambient []string
}

func (hooks *Hooks) UnmarshalJSON(b []byte) error {
	var state struct {
		Prestart  []CommandHook
		Poststart []CommandHook
		Poststop  []CommandHook
	}

	if err := json.Unmarshal(b, &state); err != nil {
		return err
	}

	deserialize := func(shooks []CommandHook) (hooks []Hook) {
		for _, shook := range shooks {
			hooks = append(hooks, shook)
		}

		return hooks
	}

	hooks.Prestart = deserialize(state.Prestart)
	hooks.Poststart = deserialize(state.Poststart)
	hooks.Poststop = deserialize(state.Poststop)
	return nil
}

func (hooks Hooks) MarshalJSON() ([]byte, error) {
	serialize := func(hooks []Hook) (serializableHooks []CommandHook) {
		for _, hook := range hooks {
			switch chook := hook.(type) {
			case CommandHook:
				serializableHooks = append(serializableHooks, chook)
			default:
				logrus.Warnf("cannot serialize hook of type %T, skipping", hook)
			}
		}

		return serializableHooks
	}

	return json.Marshal(map[string]interface{}{
		"prestart":  serialize(hooks.Prestart),
		"poststart": serialize(hooks.Poststart),
		"poststop":  serialize(hooks.Poststop),
	})
}

// HookState is the payload provided to a hook on execution.
type HookState specs.State

type Hook interface {
	// Run executes the hook with the provided state.
	Run(HookState) error
}

// NewFunctionHook will call the provided function when the hook is run.
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
	Path    string         `json:"path"`
	Args    []string       `json:"args"`
	Env     []string       `json:"env"`
	Dir     string         `json:"dir"`
	Timeout *time.Duration `json:"timeout"`
}

// NewCommandHook will execute the provided command when the hook is run.
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
	var stdout, stderr bytes.Buffer
	cmd := exec.Cmd{
		Path:   c.Path,
		Args:   c.Args,
		Env:    c.Env,
		Stdin:  bytes.NewReader(b),
		Stdout: &stdout,
		Stderr: &stderr,
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	errC := make(chan error, 1)
	go func() {
		err := cmd.Wait()
		if err != nil {
			err = fmt.Errorf("error running hook: %v, stdout: %s, stderr: %s", err, stdout.String(), stderr.String())
		}
		errC <- err
	}()
	var timerCh <-chan time.Time
	if c.Timeout != nil {
		timer := time.NewTimer(*c.Timeout)
		defer timer.Stop()
		timerCh = timer.C
	}
	select {
	case err := <-errC:
		return err
	case <-timerCh:
		cmd.Process.Kill()
		cmd.Wait()
		return fmt.Errorf("hook ran past specified timeout of %.1fs", c.Timeout.Seconds())
	}
}
