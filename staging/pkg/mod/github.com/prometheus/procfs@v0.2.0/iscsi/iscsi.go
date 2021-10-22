// Copyright 2019 The Prometheus Authors
//
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

package iscsi

import (
	"path/filepath"
	"strings"

	"github.com/prometheus/procfs/internal/fs"
)

// iscsi target started with /sys/kernel/config/target/iscsi/iqn*
// configfs + target/iscsi/iqn*
// iqnGlob is representing all the possible IQN
const iqnGlob = "target/iscsi/iqn*"

// targetCore static path /sys/kernel/config/target/core for node_exporter
// reading runtime status
const targetCore = "target/core"

// devicePath static path /sys/devices/rbd/[0-9]* for rbd devices to
// read at runtime status
const devicePath = "devices/rbd"

// FS represents the pseudo-filesystem configfs, which provides an interface to
// iscsi kernel data structures in
// sysfs	as /sys
// configfs as /sys/kernel/config
type FS struct {
	sysfs    *fs.FS
	configfs *fs.FS
}

// NewFS returns a new configfs mounted under the given mount point. It will
// error and return empty FS if the mount point can't be read. For the ease of
// use, an empty string parameter configfsMountPoint will call internal fs for
// the default sys path as /sys/kernel/config
func NewFS(sysfsPath string, configfsMountPoint string) (FS, error) {
	if strings.TrimSpace(sysfsPath) == "" {
		sysfsPath = fs.DefaultSysMountPoint
	}
	sysfs, err := fs.NewFS(sysfsPath)
	if err != nil {
		return FS{}, err
	}
	if strings.TrimSpace(configfsMountPoint) == "" {
		configfsMountPoint = fs.DefaultConfigfsMountPoint
	}
	configfs, err := fs.NewFS(configfsMountPoint)
	if err != nil {
		return FS{}, err
	}
	return FS{&sysfs, &configfs}, nil
}

// helper function to get configfs path
func (fs FS) Path(p ...string) string {
	return fs.configfs.Path(p...)
}

// ISCSIStats getting iscsi runtime information
func (fs FS) ISCSIStats() ([]*Stats, error) {
	matches, err := filepath.Glob(fs.configfs.Path(iqnGlob))
	if err != nil {
		return nil, err
	}

	stats := make([]*Stats, 0, len(matches))
	for _, iqnPath := range matches {
		// stats
		s, err := GetStats(iqnPath)
		if err != nil {
			return nil, err
		}
		stats = append(stats, s)
	}

	return stats, nil
}

// TPGT struct for sys target portal group tag info
type TPGT struct {
	Name     string // name of the tpgt group
	TpgtPath string // file path of tpgt
	IsEnable bool   // is the tpgt enable
	Luns     []LUN  // the Luns that tpgt has
}

// LUN struct for sys logical unit number info
type LUN struct {
	Name       string // name of the lun
	LunPath    string // file path of the lun
	Backstore  string // backstore of the lun
	ObjectName string // place holder for object
	TypeNumber string // place holder for number of the device
}

// FILEIO struct for backstore info
type FILEIO struct {
	Name       string // name of the fileio
	Fnumber    string // number related to the backstore
	ObjectName string // place holder for object in iscsi object
	Filename   string // link to the actual file being export
}

// IBLOCK struct for backstore info
type IBLOCK struct {
	Name       string // name of the iblock
	Bnumber    string // number related to the backstore
	ObjectName string // place holder for object in iscsi object
	Iblock     string // link to the actual block being export
}

// RBD struct for backstore info
type RBD struct {
	Name    string // name of the rbd
	Rnumber string // number related to the backstore
	Pool    string // place holder for the rbd pool
	Image   string // place holder for the rbd image
}

// RDMCP struct for backstore info
type RDMCP struct {
	Name       string // name of the rdm_cp
	ObjectName string // place holder for object name
}

// Stats struct for all targets info
type Stats struct {
	Name     string
	Tpgt     []TPGT
	RootPath string
}
