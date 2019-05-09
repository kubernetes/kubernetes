// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"fmt"
	"os"
	"path"

	"github.com/prometheus/procfs/nfs"
	"github.com/prometheus/procfs/xfs"
)

// FS represents the pseudo-filesystem proc, which provides an interface to
// kernel data structures.
type FS string

// DefaultMountPoint is the common mount point of the proc filesystem.
const DefaultMountPoint = "/proc"

// NewFS returns a new FS mounted under the given mountPoint. It will error
// if the mount point can't be read.
func NewFS(mountPoint string) (FS, error) {
	info, err := os.Stat(mountPoint)
	if err != nil {
		return "", fmt.Errorf("could not read %s: %s", mountPoint, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("mount point %s is not a directory", mountPoint)
	}

	return FS(mountPoint), nil
}

// Path returns the path of the given subsystem relative to the procfs root.
func (fs FS) Path(p ...string) string {
	return path.Join(append([]string{string(fs)}, p...)...)
}

// XFSStats retrieves XFS filesystem runtime statistics.
func (fs FS) XFSStats() (*xfs.Stats, error) {
	f, err := os.Open(fs.Path("fs/xfs/stat"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return xfs.ParseStats(f)
}

// NFSClientRPCStats retrieves NFS client RPC statistics.
func (fs FS) NFSClientRPCStats() (*nfs.ClientRPCStats, error) {
	f, err := os.Open(fs.Path("net/rpc/nfs"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return nfs.ParseClientRPCStats(f)
}

// NFSdServerRPCStats retrieves NFS daemon RPC statistics.
func (fs FS) NFSdServerRPCStats() (*nfs.ServerRPCStats, error) {
	f, err := os.Open(fs.Path("net/rpc/nfsd"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return nfs.ParseServerRPCStats(f)
}
