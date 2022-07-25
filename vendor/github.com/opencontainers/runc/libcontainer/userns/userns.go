package userns

// RunningInUserNS detects whether we are currently running in a user namespace.
// Originally copied from github.com/lxc/lxd/shared/util.go
var RunningInUserNS = runningInUserNS
