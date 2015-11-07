package configs

type Mount struct {
	// Source path for the mount.
	Source string `json:"source"`

	// Destination path for the mount inside the container.
	Destination string `json:"destination"`

	// Device the mount is for.
	Device string `json:"device"`

	// Mount flags.
	Flags int `json:"flags"`

	// Mount data applied to the mount.
	Data string `json:"data"`

	// Relabel source if set, "z" indicates shared, "Z" indicates unshared.
	Relabel string `json:"relabel"`

	// Optional Command to be run before Source is mounted.
	PremountCmds []Command `json:"premount_cmds"`

	// Optional Command to be run after Source is mounted.
	PostmountCmds []Command `json:"postmount_cmds"`
}

type Command struct {
	Path string   `json:"path"`
	Args []string `json:"args"`
	Env  []string `json:"env"`
	Dir  string   `json:"dir"`
}
