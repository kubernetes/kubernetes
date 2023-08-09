// +build linux

package netns

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"golang.org/x/sys/unix"
)

// Deprecated: use syscall pkg instead (go >= 1.5 needed).
const (
	CLONE_NEWUTS  = 0x04000000   /* New utsname group? */
	CLONE_NEWIPC  = 0x08000000   /* New ipcs */
	CLONE_NEWUSER = 0x10000000   /* New user namespace */
	CLONE_NEWPID  = 0x20000000   /* New pid namespace */
	CLONE_NEWNET  = 0x40000000   /* New network namespace */
	CLONE_IO      = 0x80000000   /* Get io context */
	bindMountPath = "/run/netns" /* Bind mount path for named netns */
)

// Setns sets namespace using syscall. Note that this should be a method
// in syscall but it has not been added.
func Setns(ns NsHandle, nstype int) (err error) {
	return unix.Setns(int(ns), nstype)
}

// Set sets the current network namespace to the namespace represented
// by NsHandle.
func Set(ns NsHandle) (err error) {
	return Setns(ns, CLONE_NEWNET)
}

// New creates a new network namespace, sets it as current and returns
// a handle to it.
func New() (ns NsHandle, err error) {
	if err := unix.Unshare(CLONE_NEWNET); err != nil {
		return -1, err
	}
	return Get()
}

// NewNamed creates a new named network namespace and returns a handle to it
func NewNamed(name string) (NsHandle, error) {
	if _, err := os.Stat(bindMountPath); os.IsNotExist(err) {
		err = os.MkdirAll(bindMountPath, 0755)
		if err != nil {
			return None(), err
		}
	}

	newNs, err := New()
	if err != nil {
		return None(), err
	}

	namedPath := path.Join(bindMountPath, name)

	f, err := os.OpenFile(namedPath, os.O_CREATE|os.O_EXCL, 0444)
	if err != nil {
		return None(), err
	}
	f.Close()

	nsPath := fmt.Sprintf("/proc/%d/task/%d/ns/net", os.Getpid(), syscall.Gettid())
	err = syscall.Mount(nsPath, namedPath, "bind", syscall.MS_BIND, "")
	if err != nil {
		return None(), err
	}

	return newNs, nil
}

// DeleteNamed deletes a named network namespace
func DeleteNamed(name string) error {
	namedPath := path.Join(bindMountPath, name)

	err := syscall.Unmount(namedPath, syscall.MNT_DETACH)
	if err != nil {
		return err
	}

	return os.Remove(namedPath)
}

// Get gets a handle to the current threads network namespace.
func Get() (NsHandle, error) {
	return GetFromThread(os.Getpid(), unix.Gettid())
}

// GetFromPath gets a handle to a network namespace
// identified by the path
func GetFromPath(path string) (NsHandle, error) {
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_CLOEXEC, 0)
	if err != nil {
		return -1, err
	}
	return NsHandle(fd), nil
}

// GetFromName gets a handle to a named network namespace such as one
// created by `ip netns add`.
func GetFromName(name string) (NsHandle, error) {
	return GetFromPath(fmt.Sprintf("/var/run/netns/%s", name))
}

// GetFromPid gets a handle to the network namespace of a given pid.
func GetFromPid(pid int) (NsHandle, error) {
	return GetFromPath(fmt.Sprintf("/proc/%d/ns/net", pid))
}

// GetFromThread gets a handle to the network namespace of a given pid and tid.
func GetFromThread(pid, tid int) (NsHandle, error) {
	return GetFromPath(fmt.Sprintf("/proc/%d/task/%d/ns/net", pid, tid))
}

// GetFromDocker gets a handle to the network namespace of a docker container.
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
	if err != nil {
		return "", err
	}
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
// modified to search for newer docker containers
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

	attempts := []string{
		filepath.Join(cgroupRoot, cgroupThis, id, "tasks"),
		// With more recent lxc versions use, cgroup will be in lxc/
		filepath.Join(cgroupRoot, cgroupThis, "lxc", id, "tasks"),
		// With more recent docker, cgroup will be in docker/
		filepath.Join(cgroupRoot, cgroupThis, "docker", id, "tasks"),
		// Even more recent docker versions under systemd use docker-<id>.scope/
		filepath.Join(cgroupRoot, "system.slice", "docker-"+id+".scope", "tasks"),
		// Even more recent docker versions under cgroup/systemd/docker/<id>/
		filepath.Join(cgroupRoot, "..", "systemd", "docker", id, "tasks"),
		// Kubernetes with docker and CNI is even more different
		filepath.Join(cgroupRoot, "..", "systemd", "kubepods", "*", "pod*", id, "tasks"),
		// Another flavor of containers location in recent kubernetes 1.11+
		filepath.Join(cgroupRoot, cgroupThis, "kubepods.slice", "kubepods-besteffort.slice", "*", "docker-"+id+".scope", "tasks"),
		// When runs inside of a container with recent kubernetes 1.11+
		filepath.Join(cgroupRoot, "kubepods.slice", "kubepods-besteffort.slice", "*", "docker-"+id+".scope", "tasks"),
	}

	var filename string
	for _, attempt := range attempts {
		filenames, _ := filepath.Glob(attempt)
		if len(filenames) > 1 {
			return pid, fmt.Errorf("Ambiguous id supplied: %v", filenames)
		} else if len(filenames) == 1 {
			filename = filenames[0]
			break
		}
	}

	if filename == "" {
		return pid, fmt.Errorf("Unable to find container: %v", id[:len(id)-1])
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
