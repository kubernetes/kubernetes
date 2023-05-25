package configs

import "golang.org/x/sys/unix"

const (
	// EXT_COPYUP is a directive to copy up the contents of a directory when
	// a tmpfs is mounted over it.
	EXT_COPYUP = 1 << iota //nolint:golint // ignore "don't use ALL_CAPS" warning
)

type Mount struct {
	// Source path for the mount.
	Source string `json:"source"`

	// Destination path for the mount inside the container.
	Destination string `json:"destination"`

	// Device the mount is for.
	Device string `json:"device"`

	// Mount flags.
	Flags int `json:"flags"`

	// Propagation Flags
	PropagationFlags []int `json:"propagation_flags"`

	// Mount data applied to the mount.
	Data string `json:"data"`

	// Relabel source if set, "z" indicates shared, "Z" indicates unshared.
	Relabel string `json:"relabel"`

	// RecAttr represents mount properties to be applied recursively (AT_RECURSIVE), see mount_setattr(2).
	RecAttr *unix.MountAttr `json:"rec_attr"`

	// Extensions are additional flags that are specific to runc.
	Extensions int `json:"extensions"`

	// Optional Command to be run before Source is mounted.
	PremountCmds []Command `json:"premount_cmds"`

	// Optional Command to be run after Source is mounted.
	PostmountCmds []Command `json:"postmount_cmds"`
}

func (m *Mount) IsBind() bool {
	return m.Flags&unix.MS_BIND != 0
}
