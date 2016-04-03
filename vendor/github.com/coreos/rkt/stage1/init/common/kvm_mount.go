// Copyright 2015 The rkt Authors
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

// Package common and the file kvm_mount.go provide functions for creating mount units for managing
// inner(kind=empty) and external(kind=host) volumes.
// note: used only for kvm flavor (lkvm based)
//
// Idea.
// For example when we have two volumes:
// 1) --volume=hostdata,kind=host,source=/host/some_data_to_share
// 2) --volume=temporary,kind=empty
// then in stage1/rootfs rkt creates two folders (in rootfs of guest)
// 1) /mnt/hostdata - which is mounted through 9p host thanks to
//					lkvm --9p=/host/some_data_to_share,hostdata flag shared to quest
// 2) /mnt/temporary - is created as empty directory in guest
//
// both of them are then bind mounted to /opt/stage2/<application/<mountPoint.path>
// for every application, that has mountPoints specified in ACI json
// - host mounting is realized by podToSystemdHostMountUnits (for whole pod),
//   which creates mount.units (9p) required and ordered before all applications
//   service units
// - bind mounting is realized by appToSystemdMountUnits (for each app),
//   which creates mount.units (bind) required and ordered before particular application
// note: systemd mount units require /usr/bin/mount
package common

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/unit"
	"github.com/coreos/rkt/common"
	"github.com/hashicorp/errwrap"
)

const (
	// location within stage1 rootfs where shared volumes will be put
	// (or empty directories for kind=empty)
	stage1MntDir = "/mnt/"
)

// makeHashFromVolumeName returns string of 16 bytes length from the volume name,
// P9 file system needs that the tag be less than 31 bytes.
func makeHashFromVolumeName(v string) (ret string) {
	h := md5.New()
	io.WriteString(h, v)
	ret = hex.EncodeToString(h.Sum(nil))
	return
}

// serviceUnitName returns a systemd service unit name for the given app name.
// note: it was shamefully copy-pasted from stage1/init/path.go
// TODO: extract common functions from path.go
func serviceUnitName(appName types.ACName) string {
	return appName.String() + ".service"
}

// installNewMountUnit creates and installs a new mount unit in the default
// systemd location (/usr/lib/systemd/system) inside the pod stage1 filesystem.
// root is pod's absolute stage1 path (from Pod.Root).
// beforeAndrequiredBy creates a systemd unit dependency (can be space separated
// for multi).
// It returns the name of the generated unit.
func installNewMountUnit(root, what, where, fsType, options, beforeAndrequiredBy, unitsDir string) (string, error) {
	opts := []*unit.UnitOption{
		unit.NewUnitOption("Unit", "Description", fmt.Sprintf("Mount unit for %s", where)),
		unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
		unit.NewUnitOption("Unit", "Before", beforeAndrequiredBy),
		unit.NewUnitOption("Mount", "What", what),
		unit.NewUnitOption("Mount", "Where", where),
		unit.NewUnitOption("Mount", "Type", fsType),
		unit.NewUnitOption("Mount", "Options", options),
		unit.NewUnitOption("Install", "RequiredBy", beforeAndrequiredBy),
	}

	unitsPath := filepath.Join(root, unitsDir)
	unitName := unit.UnitNamePathEscape(where + ".mount")

	if err := writeUnit(opts, filepath.Join(unitsPath, unitName)); err != nil {
		return "", err
	}
	diag.Printf("mount unit created: %q in %q (what=%q, where=%q)", unitName, unitsPath, what, where)

	return unitName, nil
}

func writeUnit(opts []*unit.UnitOption, unitPath string) error {
	unitBytes, err := ioutil.ReadAll(unit.Serialize(opts))
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("failed to serialize mount unit file to bytes %q", unitPath), err)
	}

	err = ioutil.WriteFile(unitPath, unitBytes, 0644)
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("failed to create mount unit file %q", unitPath), err)
	}

	return nil
}

// PodToSystemdHostMountUnits create host shared remote file system
// mounts (using e.g. 9p) according to https://www.kernel.org/doc/Documentation/filesystems/9p.txt.
// Additionally it creates required directories in stage1MntDir and then prepares
// bind mount unit for each app.
// "root" parameter is stage1 root filesystem path.
// appNames are used to create before/required dependency between mount unit and
// app service units.
func PodToSystemdHostMountUnits(root string, volumes []types.Volume, appNames []types.ACName, unitsDir string) error {
	// pod volumes need to mount p9 qemu mount_tags
	for _, vol := range volumes { // only host shared volumes
		name := vol.Name.String()
		mountTag := makeHashFromVolumeName(name)

		// serviceNames for ordering and requirements dependency for apps
		var serviceNames []string
		for _, appName := range appNames {
			serviceNames = append(serviceNames, serviceUnitName(appName))
		}

		// for host kind we create a mount unit to mount host shared folder
		if vol.Kind == "host" {
			// /var/lib/.../pod/run/rootfs/mnt/{mountTag}
			mountPoint := filepath.Join(root, stage1MntDir, mountTag)
			err := os.MkdirAll(mountPoint, 0700)
			if err != nil {
				return err
			}

			_, err = installNewMountUnit(root,
				mountTag, // what (source) in 9p it is a channel tag which equals to volume mountTag
				filepath.Join(stage1MntDir, name), // where - destination
				"9p",                            // fsType
				"trans=virtio",                  // 9p specific options
				strings.Join(serviceNames, " "), // space separated list of services for unit dependency
				unitsDir,
			)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// AppToSystemdMountUnits prepare bind mount unit for empty or host kind mounting
// between stage1 rootfs and chrooted filesystem for application
func AppToSystemdMountUnits(root string, appName types.ACName, volumes []types.Volume, ra *schema.RuntimeApp, unitsDir string) error {
	app := ra.App

	vols := make(map[types.ACName]types.Volume)
	for _, v := range volumes {
		vols[v.Name] = v
	}

	mounts := GenerateMounts(ra, vols)
	for _, m := range mounts {
		vol := vols[m.Volume]

		// source relative to stage1 rootfs to relative pod root
		hashedVolName := makeHashFromVolumeName(vol.Name.String())

		whatPath := filepath.Join(stage1MntDir, hashedVolName)

		whatFullPath := filepath.Join(root, whatPath)

		// set volume permissions
		if err := PrepareMountpoints(whatFullPath, &vol); err != nil {
			return err
		}

		// destination relative to stage1 rootfs and relative to pod root
		wherePath := filepath.Join(common.RelAppRootfsPath(appName), m.Path)
		whereFullPath := filepath.Join(root, wherePath)

		// assertion to make sure that "what" exists (created earlier by PodToSystemdHostMountUnits)
		diag.Printf("checking required source path: %q", whatFullPath)
		if _, err := os.Stat(whatFullPath); os.IsNotExist(err) {
			return fmt.Errorf("bug: missing source for volume %v", vol.Name)
		}

		// optionally prepare app directory
		diag.Printf("optionally preparing destination path: %q", whereFullPath)
		err := os.MkdirAll(whereFullPath, 0700)
		if err != nil {
			return errwrap.Wrap(fmt.Errorf("failed to prepare dir for mount %v", m.Volume), err)
		}

		// install new mount unit for bind mount /mnt/volumeName -> /opt/stage2/{app-id}/rootfs/{{mountPoint.Path}}
		mu, err := installNewMountUnit(
			root,      // where put a mount unit
			whatPath,  // what - stage1 rootfs /mnt/VolumeName
			wherePath, // where - inside chroot app filesystem
			"bind",    // fstype
			"bind",    // options
			serviceUnitName(appName),
			unitsDir,
		)
		if err != nil {
			return errwrap.Wrap(fmt.Errorf("cannot install new mount unit for app %q", appName.String()), err)
		}

		// TODO(iaguis) when we update util-linux to 2.27, this code can go
		// away and we can bind-mount RO with one unit file.
		// http://ftp.kernel.org/pub/linux/utils/util-linux/v2.27/v2.27-ReleaseNotes
		if IsMountReadOnly(vol, app.MountPoints) {
			opts := []*unit.UnitOption{
				unit.NewUnitOption("Unit", "Description", fmt.Sprintf("Remount read-only unit for %s", wherePath)),
				unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
				unit.NewUnitOption("Unit", "After", mu),
				unit.NewUnitOption("Unit", "Wants", mu),
				unit.NewUnitOption("Service", "ExecStart", fmt.Sprintf("/usr/bin/mount -o remount,ro %s", wherePath)),
				unit.NewUnitOption("Install", "RequiredBy", mu),
			}

			remountUnitPath := filepath.Join(root, unitsDir, unit.UnitNamePathEscape(wherePath+"-remount.service"))
			if err := writeUnit(opts, remountUnitPath); err != nil {
				return err
			}
		}
	}
	return nil
}

// VolumesToKvmDiskArgs prepares argument list to be passed to lkvm to configure
// shared volumes (only for "host" kind).
// Example return is ["--9p,src/folder,9ptag"].
func VolumesToKvmDiskArgs(volumes []types.Volume) []string {
	var args []string

	for _, vol := range volumes {
		// tag/channel name for virtio
		mountTag := makeHashFromVolumeName(vol.Name.String())
		if vol.Kind == "host" {
			// eg. --9p=/home/jon/srcdir,tag
			arg := "--9p=" + vol.Source + "," + mountTag
			diag.Printf("--disk argument: %#v", arg)
			args = append(args, arg)
		}
	}

	return args
}
