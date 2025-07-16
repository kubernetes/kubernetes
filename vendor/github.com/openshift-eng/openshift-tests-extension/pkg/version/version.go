package version

var (
	// CommitFromGit is a constant representing the source version that
	// generated this build. It should be set during build via -ldflags.
	CommitFromGit string
	// BuildDate in ISO8601 format, output of $(date -u +'%Y-%m-%dT%H:%M:%SZ')
	BuildDate string
	// GitTreeState has the state of git tree, either "clean" or "dirty"
	GitTreeState string
)
