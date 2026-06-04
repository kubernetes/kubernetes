package netns

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

// Deprecated: use golang.org/x/sys/unix pkg instead.
const (
	CLONE_NEWUTS  = unix.CLONE_NEWUTS  /* New utsname group? */
	CLONE_NEWIPC  = unix.CLONE_NEWIPC  /* New ipcs */
	CLONE_NEWUSER = unix.CLONE_NEWUSER /* New user namespace */
	CLONE_NEWPID  = unix.CLONE_NEWPID  /* New pid namespace */
	CLONE_NEWNET  = unix.CLONE_NEWNET  /* New network namespace */
	CLONE_IO      = unix.CLONE_IO      /* Get io context */
)

const bindMountPath = "/run/netns" /* Bind mount path for named netns */

// Setns sets namespace using golang.org/x/sys/unix.Setns.
//
// Deprecated: Use golang.org/x/sys/unix.Setns instead.
func Setns(ns NsHandle, nstype int) error {
	return unix.Setns(int(ns), nstype)
}

// Set sets the current network namespace to the namespace represented
// by NsHandle.
func Set(ns NsHandle) error {
	return unix.Setns(int(ns), unix.CLONE_NEWNET)
}

// New creates a new network namespace, sets it as current and returns
// a handle to it.
func New() (NsHandle, error) {
	if err := unix.Unshare(unix.CLONE_NEWNET); err != nil {
		return -1, err
	}
	return Get()
}

// NewNamed creates a new named network namespace, sets it as current,
// and returns a handle to it
func NewNamed(name string) (NsHandle, error) {
	if _, err := os.Stat(bindMountPath); os.IsNotExist(err) {
		err = os.MkdirAll(bindMountPath, 0o755)
		if err != nil {
			return None(), err
		}
	}

	newNs, err := New()
	if err != nil {
		return None(), err
	}

	namedPath := path.Join(bindMountPath, name)

	f, err := os.OpenFile(namedPath, os.O_CREATE|os.O_EXCL, 0o444)
	if err != nil {
		newNs.Close()
		return None(), err
	}
	f.Close()

	nsPath := fmt.Sprintf("/proc/%d/task/%d/ns/net", os.Getpid(), unix.Gettid())
	err = unix.Mount(nsPath, namedPath, "bind", unix.MS_BIND, "")
	if err != nil {
		newNs.Close()
		return None(), err
	}

	return newNs, nil
}

// DeleteNamed deletes a named network namespace
func DeleteNamed(name string) error {
	namedPath := path.Join(bindMountPath, name)

	err := unix.Unmount(namedPath, unix.MNT_DETACH)
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
	return GetFromPath(filepath.Join(bindMountPath, name))
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
func findCgroupMountpoint(cgroupType string) (int, string, error) {
	output, err := os.ReadFile("/proc/mounts")
	if err != nil {
		return -1, "", err
	}

	// /proc/mounts has 6 fields per line, one mount per line, e.g.
	// cgroup /sys/fs/cgroup/devices cgroup rw,relatime,devices 0 0
	for _, line := range strings.Split(string(output), "\n") {
		parts := strings.Split(line, " ")
		if len(parts) == 6 {
			switch parts[2] {
			case "cgroup2":
				return 2, parts[1], nil
			case "cgroup":
				for _, opt := range strings.Split(parts[3], ",") {
					if opt == cgroupType {
						return 1, parts[1], nil
					}
				}
			}
		}
	}

	return -1, "", fmt.Errorf("cgroup mountpoint not found for %s", cgroupType)
}

// Returns the relative path to the cgroup docker is running in.
// borrowed from docker/utils/utils.go
// modified to get the docker pid instead of using /proc/self
func getDockerCgroup(cgroupVer int, cgroupType string) (string, error) {
	dockerpid, err := os.ReadFile("/var/run/docker.pid")
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
	output, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(string(output), "\n") {
		parts := strings.Split(line, ":")
		// any type used by docker should work
		if (cgroupVer == 1 && parts[1] == cgroupType) ||
			(cgroupVer == 2 && parts[1] == "") {
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
// modified to look for cgroups v2
func getPidForContainer(id string) (int, error) {
	pid := 0

	// memory is chosen randomly, any cgroup used by docker works
	cgroupType := "memory"

	cgroupVer, cgroupRoot, err := findCgroupMountpoint(cgroupType)
	if err != nil {
		return pid, err
	}

	cgroupDocker, err := getDockerCgroup(cgroupVer, cgroupType)
	if err != nil {
		return pid, err
	}

	id += "*"

	var pidFile string
	switch cgroupVer {
	case 1:
		pidFile = "tasks"
	case 2:
		pidFile = "cgroup.procs"
	default:
		return -1, fmt.Errorf("Invalid cgroup version '%d'", cgroupVer)
	}

	attempts := []string{
		filepath.Join(cgroupRoot, cgroupDocker, id, pidFile),
		// With more recent lxc versions use, cgroup will be in lxc/
		filepath.Join(cgroupRoot, cgroupDocker, "lxc", id, pidFile),
		// With more recent docker, cgroup will be in docker/
		filepath.Join(cgroupRoot, cgroupDocker, "docker", id, pidFile),
		// Even more recent docker versions under systemd use docker-<id>.scope/
		filepath.Join(cgroupRoot, "system.slice", "docker-"+id+".scope", pidFile),
		// Even more recent docker versions under cgroup/systemd/docker/<id>/
		filepath.Join(cgroupRoot, "..", "systemd", "docker", id, pidFile),
		// Kubernetes with docker and CNI is even more different. Works for BestEffort and Burstable QoS
		filepath.Join(cgroupRoot, "..", "systemd", "kubepods", "*", "pod*", id, pidFile),
		// Same as above but for Guaranteed QoS
		filepath.Join(cgroupRoot, "..", "systemd", "kubepods", "pod*", id, pidFile),
		// Another flavor of containers location in recent kubernetes 1.11+. Works for BestEffort and Burstable QoS
		filepath.Join(cgroupRoot, cgroupDocker, "kubepods.slice", "*.slice", "*", "docker-"+id+".scope", pidFile),
		// Same as above but for Guaranteed QoS
		filepath.Join(cgroupRoot, cgroupDocker, "kubepods.slice", "*", "docker-"+id+".scope", pidFile),
		// When runs inside of a container with recent kubernetes 1.11+. Works for BestEffort and Burstable QoS
		filepath.Join(cgroupRoot, "kubepods.slice", "*.slice", "*", "docker-"+id+".scope", pidFile),
		// Same as above but for Guaranteed QoS
		filepath.Join(cgroupRoot, "kubepods.slice", "*", "docker-"+id+".scope", pidFile),
		// Support for nerdctl
		filepath.Join(cgroupRoot, "system.slice", "nerdctl-"+id+".scope", pidFile),
		// Support for finch
		filepath.Join(cgroupRoot, "..", "systemd", "finch", id, pidFile),
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

	output, err := os.ReadFile(filename)
	if err != nil {
		return pid, err
	}

	result := strings.Split(string(output), "\n")
	if len(result) == 0 || len(result[0]) == 0 {
		return pid, fmt.Errorf("No pid found for container")
	}

	pid, err = strconv.Atoi(result[0])
	if err != nil {
		return pid, fmt.Errorf("Invalid pid '%s': %w", result[0], err)
	}

	return pid, nil
}
