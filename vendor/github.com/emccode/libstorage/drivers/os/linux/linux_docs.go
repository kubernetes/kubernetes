// +build linux

/*
Package linux is the OS driver for linux. In order to reduce external
dependencies, this package borrows the following packages:

  - github.com/docker/docker/pkg/mount
  - github.com/opencontainers/runc/libcontainer/label
*/
package linux
