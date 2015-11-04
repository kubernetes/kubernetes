// Package netns allows ultra-simple network namespace handling. NsHandles
// can be retrieved and set. Note that the current namespace is thread
// local so actions that set and reset namespaces should use LockOSThread
// to make sure the namespace doesn't change due to a goroutine switch.
// It is best to close NsHandles when you are done with them. This can be
// accomplished via a `defer ns.Close()` on the handle. Changing namespaces
// requires elevated privileges, so in most cases this code needs to be run
// as root.
package netns

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

const (
	// These constants belong in the syscall library but have not been
	// added yet.
	CLONE_NEWUTS  = 0x04000000 /* New utsname group? */
	CLONE_NEWIPC  = 0x08000000 /* New ipcs */
	CLONE_NEWUSER = 0x10000000 /* New user namespace */
	CLONE_NEWPID  = 0x20000000 /* New pid namespace */
	CLONE_NEWNET  = 0x40000000 /* New network namespace */
	CLONE_IO      = 0x80000000 /* Get io context */
)

// Setns sets namespace using syscall. Note that this should be a method
// in syscall but it has not been added.
func Setns(ns NsHandle, nstype int) (err error) {
	_, _, e1 := syscall.Syscall(SYS_SETNS, uintptr(ns), uintptr(nstype), 0)
	if e1 != 0 {
		err = e1
	}
	return
}

// NsHandle is a handle to a network namespace. It can be cast directly
// to an int and used as a file descriptor.
type NsHandle int

// Equal determines if two network handles refer to the same network
// namespace. This is done by comparing the device and inode that the
// file descripors point to.
func (ns NsHandle) Equal(other NsHandle) bool {
	var s1, s2 syscall.Stat_t
	if err := syscall.Fstat(int(ns), &s1); err != nil {
		return false
	}
	if err := syscall.Fstat(int(other), &s2); err != nil {
		return false
	}
	return (s1.Dev == s2.Dev) && (s1.Ino == s2.Ino)
}

// String shows the file descriptor number and its dev and inode.
func (ns NsHandle) String() string {
	var s syscall.Stat_t
	if err := syscall.Fstat(int(ns), &s); err != nil {
		return fmt.Sprintf("NS(%d: unknown)", ns)
	}
	return fmt.Sprintf("NS(%d: %d, %d)", ns, s.Dev, s.Ino)
}

// IsOpen returns true if Close() has not been called.
func (ns NsHandle) IsOpen() bool {
	return ns != -1
}

// Close closes the NsHandle and resets its file descriptor to -1.
// It is not safe to use an NsHandle after Close() is called.
func (ns *NsHandle) Close() error {
	if err := syscall.Close(int(*ns)); err != nil {
		return err
	}
	(*ns) = -1
	return nil
}

// Set sets the current network namespace to the namespace represented
// by NsHandle.
func Set(ns NsHandle) (err error) {
	return Setns(ns, CLONE_NEWNET)
}

// New creates a new network namespace and returns a handle to it.
func New() (ns NsHandle, err error) {
	if err := syscall.Unshare(CLONE_NEWNET); err != nil {
		return -1, err
	}
	return Get()
}

// Get gets a handle to the current threads network namespace.
func Get() (NsHandle, error) {
	return GetFromPid(os.Getpid())
}

// GetFromName gets a handle to a named network namespace such as one
// created by `ip netns add`.
func GetFromName(name string) (NsHandle, error) {
	fd, err := syscall.Open(fmt.Sprintf("/var/run/netns/%s", name), syscall.O_RDONLY, 0)
	if err != nil {
		return -1, err
	}
	return NsHandle(fd), nil
}

// GetFromName gets a handle to the network namespace of a given pid.
func GetFromPid(pid int) (NsHandle, error) {
	fd, err := syscall.Open(fmt.Sprintf("/proc/%d/ns/net", pid), syscall.O_RDONLY, 0)
	if err != nil {
		return -1, err
	}
	return NsHandle(fd), nil
}

// GetFromName gets a handle to the network namespace of a docker container.
// Id is prefixed matched against the running docker containers, so a short
// identifier can be used as long as it isn't ambiguous.
func GetFromDocker(id string) (NsHandle, error) {
	pid, err := getPidForContainer(id)
	if err != nil {
		return -1, err
	}
	return GetFromPid(pid)
}

// borrowed from docker/utils/utils.go
func findCgroupMountpoint(cgroupType string) (string, error) {
	output, err := ioutil.ReadFile("/proc/mounts")
	if err != nil {
		return "", err
	}

	// /proc/mounts has 6 fields per line, one mount per line, e.g.
	// cgroup /sys/fs/cgroup/devices cgroup rw,relatime,devices 0 0
	for _, line := range strings.Split(string(output), "\n") {
		parts := strings.Split(line, " ")
		if len(parts) == 6 && parts[2] == "cgroup" {
			for _, opt := range strings.Split(parts[3], ",") {
				if opt == cgroupType {
					return parts[1], nil
				}
			}
		}
	}

	return "", fmt.Errorf("cgroup mountpoint not found for %s", cgroupType)
}

// Returns the relative path to the cgroup docker is running in.
// borrowed from docker/utils/utils.go
// modified to get the docker pid instead of using /proc/self
func getThisCgroup(cgroupType string) (string, error) {
	dockerpid, err := ioutil.ReadFile("/var/run/docker.pid")
	if err != nil {
		return "", err
	}
	result := strings.Split(string(dockerpid), "\n")
	if len(result) == 0 || len(result[0]) == 0 {
		return "", fmt.Errorf("docker pid not found in /var/run/docker.pid")
	}
	pid, err := strconv.Atoi(result[0])

	output, err := ioutil.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(string(output), "\n") {
		parts := strings.Split(line, ":")
		// any type used by docker should work
		if parts[1] == cgroupType {
			return parts[2], nil
		}
	}
	return "", fmt.Errorf("cgroup '%s' not found in /proc/%d/cgroup", cgroupType, pid)
}

// Returns the first pid in a container.
// borrowed from docker/utils/utils.go
// modified to only return the first pid
// modified to glob with id
func getPidForContainer(id string) (int, error) {
	pid := 0

	// memory is chosen randomly, any cgroup used by docker works
	cgroupType := "memory"

	cgroupRoot, err := findCgroupMountpoint(cgroupType)
	if err != nil {
		return pid, err
	}

	cgroupThis, err := getThisCgroup(cgroupType)
	if err != nil {
		return pid, err
	}

	id += "*"

	filename := filepath.Join(cgroupRoot, cgroupThis, id, "tasks")
	filenames, _ := filepath.Glob(filename)
	if len(filenames) > 1 {
		return pid, fmt.Errorf("Ambiguous id supplied: %v", filenames)
	} else if len(filenames) == 1 {
		filename = filenames[0]
	}
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		// With more recent lxc versions use, cgroup will be in lxc/
		filename = filepath.Join(cgroupRoot, cgroupThis, "lxc", id, "tasks")
		filenames, _ = filepath.Glob(filename)
		if len(filenames) > 1 {
			return pid, fmt.Errorf("Ambiguous id supplied: %v", filenames)
		} else if len(filenames) == 1 {
			filename = filenames[0]
		} else {
			return pid, fmt.Errorf("Unable to find container: %v", id[:len(id)-1])
		}
	}

	output, err := ioutil.ReadFile(filename)
	if err != nil {
		return pid, err
	}

	result := strings.Split(string(output), "\n")
	if len(result) == 0 || len(result[0]) == 0 {
		return pid, fmt.Errorf("No pid found for container")
	}

	pid, err = strconv.Atoi(result[0])
	if err != nil {
		return pid, fmt.Errorf("Invalid pid '%s': %s", result[0], err)
	}

	return pid, nil
}
