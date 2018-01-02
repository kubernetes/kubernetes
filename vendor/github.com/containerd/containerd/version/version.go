package version

var (
	// Package is filled at linking time
	Package = "github.com/containerd/containerd"

	// Version holds the complete version number. Filled in at linking time.
	Version = "1.0.0-beta.2+unknown"

	// Revision is filled with the VCS (e.g. git) revision being used to build
	// the program at linking time.
	Revision = ""
)
