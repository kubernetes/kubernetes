// +build linux

/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package nsenter

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/utils/exec"

	"k8s.io/klog"
)

const (
	// DefaultHostRootFsPath is path to host's filesystem mounted into container
	// with kubelet.
	DefaultHostRootFsPath = "/rootfs"
	// mountNsPath is the default mount namespace of the host
	mountNsPath = "/proc/1/ns/mnt"
	// nsenterPath is the default nsenter command
	nsenterPath = "nsenter"
)

// Nsenter is part of experimental support for running the kubelet
// in a container.
//
// Nsenter requires:
//
// 1.  Docker >= 1.6 due to the dependency on the slave propagation mode
//     of the bind-mount of the kubelet root directory in the container.
//     Docker 1.5 used a private propagation mode for bind-mounts, so mounts
//     performed in the host's mount namespace do not propagate out to the
//     bind-mount in this docker version.
// 2.  The host's root filesystem must be available at /rootfs
// 3.  The nsenter binary must be on the Kubelet process' PATH in the container's
//     filesystem.
// 4.  The Kubelet process must have CAP_SYS_ADMIN (required by nsenter); at
//     the present, this effectively means that the kubelet is running in a
//     privileged container.
// 5.  The volume path used by the Kubelet must be the same inside and outside
//     the container and be writable by the container (to initialize volume)
//     contents. TODO: remove this requirement.
// 6.  The host image must have "mount", "findmnt", "umount", "stat", "touch",
//     "mkdir", "ls", "sh" and "chmod" binaries in /bin, /usr/sbin, or /usr/bin
// 7.  The host image should have systemd-run in /bin, /usr/sbin, or /usr/bin if
//     systemd is installed/enabled in the operating system.
// For more information about mount propagation modes, see:
//   https://www.kernel.org/doc/Documentation/filesystems/sharedsubtree.txt
type Nsenter struct {
	// a map of commands to their paths on the host filesystem
	paths map[string]string

	// Path to the host filesystem, typically "/rootfs". Used only for testing.
	hostRootFsPath string

	// Exec implementation, used only for testing
	executor exec.Interface
}

// NewNsenter constructs a new instance of Nsenter
func NewNsenter(hostRootFsPath string, executor exec.Interface) (*Nsenter, error) {
	ne := &Nsenter{
		hostRootFsPath: hostRootFsPath,
		executor:       executor,
	}
	if err := ne.initPaths(); err != nil {
		return nil, err
	}
	return ne, nil
}

func (ne *Nsenter) initPaths() error {
	ne.paths = map[string]string{}
	binaries := []string{
		"mount",
		"findmnt",
		"umount",
		"systemd-run",
		"stat",
		"touch",
		"mkdir",
		"sh",
		"chmod",
		"realpath",
	}
	// search for the required commands in other locations besides /usr/bin
	for _, binary := range binaries {
		// check for binary under the following directories
		for _, path := range []string{"/", "/bin", "/usr/sbin", "/usr/bin"} {
			binPath := filepath.Join(path, binary)
			if _, err := os.Stat(filepath.Join(ne.hostRootFsPath, binPath)); err != nil {
				continue
			}
			ne.paths[binary] = binPath
			break
		}
		// systemd-run is optional, bailout if we don't find any of the other binaries
		if ne.paths[binary] == "" && binary != "systemd-run" {
			return fmt.Errorf("unable to find %v", binary)
		}
	}
	return nil
}

// Exec executes nsenter commands in hostProcMountNsPath mount namespace
func (ne *Nsenter) Exec(cmd string, args []string) exec.Cmd {
	hostProcMountNsPath := filepath.Join(ne.hostRootFsPath, mountNsPath)
	fullArgs := append([]string{fmt.Sprintf("--mount=%s", hostProcMountNsPath), "--"},
		append([]string{ne.AbsHostPath(cmd)}, args...)...)
	klog.V(5).Infof("Running nsenter command: %v %v", nsenterPath, fullArgs)
	return ne.executor.Command(nsenterPath, fullArgs...)
}

// AbsHostPath returns the absolute runnable path for a specified command
func (ne *Nsenter) AbsHostPath(command string) string {
	path, ok := ne.paths[command]
	if !ok {
		return command
	}
	return path
}

// SupportsSystemd checks whether command systemd-run exists
func (ne *Nsenter) SupportsSystemd() (string, bool) {
	systemdRunPath, ok := ne.paths["systemd-run"]
	return systemdRunPath, ok && systemdRunPath != ""
}

// EvalSymlinks returns the path name on the host after evaluating symlinks on the
// host.
// mustExist makes EvalSymlinks to return error when the path does not
// exist. When it's false, it evaluates symlinks of the existing part and
// blindly adds the non-existing part:
// pathname: /mnt/volume/non/existing/directory
//     /mnt/volume exists
//                non/existing/directory does not exist
// -> It resolves symlinks in /mnt/volume to say /mnt/foo and returns
//    /mnt/foo/non/existing/directory.
//
// BEWARE! EvalSymlinks is not able to detect symlink looks with mustExist=false!
// If /tmp/link is symlink to /tmp/link, EvalSymlinks(/tmp/link/foo) returns /tmp/link/foo.
func (ne *Nsenter) EvalSymlinks(pathname string, mustExist bool) (string, error) {
	var args []string
	if mustExist {
		// "realpath -e: all components of the path must exist"
		args = []string{"-e", pathname}
	} else {
		// "realpath -m: no path components need exist or be a directory"
		args = []string{"-m", pathname}
	}
	outBytes, err := ne.Exec("realpath", args).CombinedOutput()
	if err != nil {
		klog.Infof("failed to resolve symbolic links on %s: %v", pathname, err)
		return "", err
	}
	return strings.TrimSpace(string(outBytes)), nil
}

// KubeletPath returns the path name that can be accessed by containerized
// kubelet. It is recommended to resolve symlinks on the host by EvalSymlinks
// before calling this function
func (ne *Nsenter) KubeletPath(pathname string) string {
	return filepath.Join(ne.hostRootFsPath, pathname)
}

// NewFakeNsenter returns a Nsenter that does not run "nsenter --mount=... --",
// but runs everything in the same mount namespace as the unit test binary.
// rootfsPath is supposed to be a symlink, e.g. /tmp/xyz/rootfs -> /.
// This fake Nsenter is enough for most operations, e.g. to resolve symlinks,
// but it's not enough to call /bin/mount - unit tests don't run as root.
func NewFakeNsenter(rootfsPath string) (*Nsenter, error) {
	executor := &fakeExec{
		rootfsPath: rootfsPath,
	}
	// prepare /rootfs/bin, usr/bin and usr/sbin
	bin := filepath.Join(rootfsPath, "bin")
	if err := os.Symlink("/bin", bin); err != nil {
		return nil, err
	}

	usr := filepath.Join(rootfsPath, "usr")
	if err := os.Mkdir(usr, 0755); err != nil {
		return nil, err
	}
	usrbin := filepath.Join(usr, "bin")
	if err := os.Symlink("/usr/bin", usrbin); err != nil {
		return nil, err
	}
	usrsbin := filepath.Join(usr, "sbin")
	if err := os.Symlink("/usr/sbin", usrsbin); err != nil {
		return nil, err
	}

	return NewNsenter(rootfsPath, executor)
}

type fakeExec struct {
	rootfsPath string
}

func (f fakeExec) Command(cmd string, args ...string) exec.Cmd {
	// This will intentionaly panic if Nsenter does not provide enough arguments.
	realCmd := args[2]
	realArgs := args[3:]
	return exec.New().Command(realCmd, realArgs...)
}

func (fakeExec) LookPath(file string) (string, error) {
	return "", errors.New("not implemented")
}

func (fakeExec) CommandContext(ctx context.Context, cmd string, args ...string) exec.Cmd {
	return nil
}

var _ exec.Interface = fakeExec{}
