package configs

import "golang.org/x/sys/unix"

type MountIDMapping struct {
	// Recursive indicates if the mapping needs to be recursive.
	Recursive bool `json:"recursive"`

	// UserNSPath is a path to a user namespace that indicates the necessary
	// id-mappings for MOUNT_ATTR_IDMAP. If set to non-"", UIDMappings and
	// GIDMappings must be set to nil.
	UserNSPath string `json:"userns_path,omitempty"`

	// UIDMappings is the uid mapping set for this mount, to be used with
	// MOUNT_ATTR_IDMAP.
	UIDMappings []IDMap `json:"uid_mappings,omitempty"`

	// GIDMappings is the gid mapping set for this mount, to be used with
	// MOUNT_ATTR_IDMAP.
	GIDMappings []IDMap `json:"gid_mappings,omitempty"`
}

type Mount struct {
	// Source path for the mount.
	Source string `json:"source"`

	// Destination path for the mount inside the container.
	Destination string `json:"destination"`

	// Device the mount is for.
	Device string `json:"device"`

	// Mount flags.
	Flags int `json:"flags"`

	// Mount flags that were explicitly cleared in the configuration (meaning
	// the user explicitly requested that these flags *not* be set).
	ClearedFlags int `json:"cleared_flags"`

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

	// Mapping is the MOUNT_ATTR_IDMAP configuration for the mount. If non-nil,
	// the mount is configured to use MOUNT_ATTR_IDMAP-style id mappings.
	IDMapping *MountIDMapping `json:"id_mapping,omitempty"`
}

func (m *Mount) IsBind() bool {
	return m.Flags&unix.MS_BIND != 0
}

func (m *Mount) IsIDMapped() bool {
	return m.IDMapping != nil
}
