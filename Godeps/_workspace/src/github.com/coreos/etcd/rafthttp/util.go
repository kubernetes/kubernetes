// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rafthttp

import (
	"encoding/binary"
	"fmt"
	"io"
	"net/http"

	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/version"
	"github.com/coreos/go-semver/semver"
)

func writeEntryTo(w io.Writer, ent *raftpb.Entry) error {
	size := ent.Size()
	if err := binary.Write(w, binary.BigEndian, uint64(size)); err != nil {
		return err
	}
	b, err := ent.Marshal()
	if err != nil {
		return err
	}
	_, err = w.Write(b)
	return err
}

func readEntryFrom(r io.Reader, ent *raftpb.Entry) error {
	var l uint64
	if err := binary.Read(r, binary.BigEndian, &l); err != nil {
		return err
	}
	buf := make([]byte, int(l))
	if _, err := io.ReadFull(r, buf); err != nil {
		return err
	}
	return ent.Unmarshal(buf)
}

// compareMajorMinorVersion returns an integer comparing two versions based on
// their major and minor version. The result will be 0 if a==b, -1 if a < b,
// and 1 if a > b.
func compareMajorMinorVersion(a, b *semver.Version) int {
	na := &semver.Version{Major: a.Major, Minor: a.Minor}
	nb := &semver.Version{Major: b.Major, Minor: b.Minor}
	switch {
	case na.LessThan(*nb):
		return -1
	case nb.LessThan(*na):
		return 1
	default:
		return 0
	}
}

// serverVersion returns the server version from the given header.
func serverVersion(h http.Header) *semver.Version {
	verStr := h.Get("X-Server-Version")
	// backward compatibility with etcd 2.0
	if verStr == "" {
		verStr = "2.0.0"
	}
	return semver.Must(semver.NewVersion(verStr))
}

// serverVersion returns the min cluster version from the given header.
func minClusterVersion(h http.Header) *semver.Version {
	verStr := h.Get("X-Min-Cluster-Version")
	// backward compatibility with etcd 2.0
	if verStr == "" {
		verStr = "2.0.0"
	}
	return semver.Must(semver.NewVersion(verStr))
}

// checkVersionCompability checks whether the given version is compatible
// with the local version.
func checkVersionCompability(name string, server, minCluster *semver.Version) error {
	localServer := semver.Must(semver.NewVersion(version.Version))
	localMinCluster := semver.Must(semver.NewVersion(version.MinClusterVersion))
	if compareMajorMinorVersion(server, localMinCluster) == -1 {
		return fmt.Errorf("remote version is too low: remote[%s]=%s, local=%s", name, server, localServer)
	}
	if compareMajorMinorVersion(minCluster, localServer) == 1 {
		return fmt.Errorf("local version is too low: remote[%s]=%s, local=%s", name, server, localServer)
	}
	return nil
}
