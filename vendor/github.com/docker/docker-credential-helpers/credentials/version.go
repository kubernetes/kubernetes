package credentials

var (
	// Name is filled at linking time
	Name = ""

	// Package is filled at linking time
	Package = "github.com/docker/docker-credential-helpers"

	// Version holds the complete version number. Filled in at linking time.
	Version = "v0.0.0+unknown"

	// Revision is filled with the VCS (e.g. git) revision being used to build
	// the program at linking time.
	Revision = ""
)
