// +build darwin freebsd solaris

package containerd

const (
	// DefaultSnapshotter will set the default snapshotter for the platform.
	// This will be based on the client compilation target, so take that into
	// account when choosing this value.
	DefaultSnapshotter = "naive"
)
