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
	"fmt"
	"os"
	"path/filepath"

	"k8s.io/utils/exec"

	"github.com/golang/glog"
)

const (
	hostRootFsPath = "/rootfs"
	// hostProcMountNsPath is the default mount namespace for rootfs
	hostProcMountNsPath = "/rootfs/proc/1/ns/mnt"
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
// 7.  The host image should have systemd-run in /bin, /usr/sbin, or /usr/bin
// For more information about mount propagation modes, see:
//   https://www.kernel.org/doc/Documentation/filesystems/sharedsubtree.txt
type Nsenter struct {
	// a map of commands to their paths on the host filesystem
	paths map[string]string
}

// NewNsenter constructs a new instance of Nsenter
func NewNsenter() *Nsenter {
	ne := &Nsenter{
		paths: map[string]string{
			"mount":       "",
			"findmnt":     "",
			"umount":      "",
			"systemd-run": "",
			"stat":        "",
			"touch":       "",
			"mkdir":       "",
			"ls":          "",
			"sh":          "",
			"chmod":       "",
		},
	}
	// search for the required commands in other locations besides /usr/bin
	for binary := range ne.paths {
		// default to root
		ne.paths[binary] = filepath.Join("/", binary)
		for _, path := range []string{"/bin", "/usr/sbin", "/usr/bin"} {
			binPath := filepath.Join(path, binary)
			if _, err := os.Stat(filepath.Join(hostRootFsPath, binPath)); err != nil {
				continue
			}
			ne.paths[binary] = binPath
			break
		}
		// TODO: error, so that the kubelet can stop if the paths don't exist
		// (don't forget that systemd-run is optional)
	}
	return ne
}

// Exec executes nsenter commands in hostProcMountNsPath mount namespace
func (ne *Nsenter) Exec(cmd string, args []string) exec.Cmd {
	fullArgs := append([]string{fmt.Sprintf("--mount=%s", hostProcMountNsPath), "--"},
		append([]string{ne.AbsHostPath(cmd)}, args...)...)
	glog.V(5).Infof("Running nsenter command: %v %v", nsenterPath, fullArgs)
	exec := exec.New()
	return exec.Command(nsenterPath, fullArgs...)
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
	systemdRunPath, hasSystemd := ne.paths["systemd-run"]
	return systemdRunPath, hasSystemd
}
