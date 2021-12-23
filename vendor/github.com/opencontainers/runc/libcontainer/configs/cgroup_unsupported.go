//go:build !linux
// +build !linux

package configs

// Cgroup holds properties of a cgroup on Linux
// TODO Windows: This can ultimately be entirely factored out on Windows as
// cgroups are a Unix-specific construct.
type Cgroup struct{}
