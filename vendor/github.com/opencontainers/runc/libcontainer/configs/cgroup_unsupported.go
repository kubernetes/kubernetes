// +build !linux

package configs

// TODO Windows: This can ultimately be entirely factored out on Windows as
// cgroups are a Unix-specific construct.
type Cgroup struct {
}
