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

package common

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"syscall"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/fileutil"
	"github.com/coreos/rkt/pkg/fs"
	"github.com/coreos/rkt/pkg/user"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"
)

/*
 * Some common stage1 mount tasks
 *
 * TODO(cdc) De-duplicate code from stage0/gc.go
 */

// Mount extends schema.Mount with additional rkt specific fields.
type Mount struct {
	schema.Mount

	Volume         types.Volume
	DockerImplicit bool
	ReadOnly       bool
}

// ConvertedFromDocker determines if an app's image has been converted
// from docker. This is needed because implicit docker empty volumes have
// different behavior from AppC
func ConvertedFromDocker(im *schema.ImageManifest) bool {
	if im == nil { // nil sometimes sneaks in here due to unit tests
		return false
	}
	ann := im.Annotations
	_, ok := ann.Get("appc.io/docker/repository")
	return ok
}

// Source computes the real volume source for a volume.
// Volumes of type 'empty' use a workdir relative to podRoot
func (m *Mount) Source(podRoot string) string {
	switch m.Volume.Kind {
	case "host":
		return m.Volume.Source
	case "empty":
		return filepath.Join(common.SharedVolumesPath(podRoot), m.Volume.Name.String())
	}
	return "" // We validate in GenerateMounts that it's valid
}

// GenerateMounts maps MountPoint paths to volumes, returning a list of mounts,
// each with a parameter indicating if it's an implicit empty volume from a
// Docker image.
func GenerateMounts(ra *schema.RuntimeApp, podVolumes []types.Volume, convertedFromDocker bool) ([]Mount, error) {
	app := ra.App

	var genMnts []Mount

	vols := make(map[types.ACName]types.Volume)
	for _, v := range podVolumes {
		vols[v.Name] = v
	}

	// RuntimeApps have mounts, whereas Apps have mountPoints. mountPoints are partially for
	// Docker compat; since apps can declare mountpoints. However, if we just run with rkt run,
	// then we'll only have a Mount and no corresponding MountPoint.
	// Furthermore, Mounts can have embedded volumes in the case of the CRI.
	// So, we generate a pile of Mounts and their corresponding Volume

	// Map of hostpath -> Mount
	mnts := make(map[string]schema.Mount)

	// Check runtimeApp's Mounts
	for _, m := range ra.Mounts {
		mnts[m.Path] = m

		vol := m.AppVolume // Mounts can supply a volume
		if vol == nil {
			vv, ok := vols[m.Volume]
			if !ok {
				return nil, fmt.Errorf("could not find volume %s", m.Volume)
			}
			vol = &vv
		}

		// Find a corresponding MountPoint, which is optional
		ro := false
		for _, mp := range ra.App.MountPoints {
			if mp.Name == m.Volume {
				ro = mp.ReadOnly
				break
			}
		}
		if vol.ReadOnly != nil {
			ro = *vol.ReadOnly
		}

		switch vol.Kind {
		case "host":
		case "empty":
		default:
			return nil, fmt.Errorf("Volume %s has invalid kind %s", vol.Name, vol.Kind)
		}
		genMnts = append(genMnts,
			Mount{
				Mount:          m,
				DockerImplicit: false,
				ReadOnly:       ro,
				Volume:         *vol,
			})
	}

	// Now, match up MountPoints with Mounts or Volumes
	// If there's no Mount and no Volume, generate an empty volume
	for _, mp := range app.MountPoints {
		// there's already a Mount for this MountPoint, stop
		if _, ok := mnts[mp.Path]; ok {
			continue
		}

		// No Mount, try to match based on volume name
		vol, ok := vols[mp.Name]
		// there is no volume for this mount point, creating an "empty" volume
		// implicitly
		if !ok {
			defaultMode := "0755"
			defaultUID := 0
			defaultGID := 0
			uniqName := ra.Name + "-" + mp.Name
			emptyVol := types.Volume{
				Name: uniqName,
				Kind: "empty",
				Mode: &defaultMode,
				UID:  &defaultUID,
				GID:  &defaultGID,
			}

			log.Printf("warning: no volume specified for mount point %q, implicitly creating an \"empty\" volume. This volume will be removed when the pod is garbage-collected.", mp.Name)
			if convertedFromDocker {
				log.Printf("Docker converted image, initializing implicit volume with data contained at the mount point %q.", mp.Name)
			}

			vols[uniqName] = emptyVol
			genMnts = append(genMnts,
				Mount{
					Mount: schema.Mount{
						Volume: uniqName,
						Path:   mp.Path,
					},
					Volume:         emptyVol,
					ReadOnly:       mp.ReadOnly,
					DockerImplicit: convertedFromDocker,
				})
		} else {
			ro := mp.ReadOnly
			if vol.ReadOnly != nil {
				ro = *vol.ReadOnly
			}
			genMnts = append(genMnts,
				Mount{
					Mount: schema.Mount{
						Volume: vol.Name,
						Path:   mp.Path,
					},
					Volume:         vol,
					ReadOnly:       ro,
					DockerImplicit: false,
				})
		}
	}

	return genMnts, nil
}

// PrepareMountpoints creates and sets permissions for empty volumes.
// If the mountpoint comes from a Docker image and it is an implicit empty
// volume, we copy files from the image to the volume, see
// https://docs.docker.com/engine/userguide/containers/dockervolumes/#data-volumes
func PrepareMountpoints(volPath string, targetPath string, vol *types.Volume, dockerImplicit bool) error {
	if vol.Kind != "empty" {
		return nil
	}

	diag.Printf("creating an empty volume folder for sharing: %q", volPath)
	m, err := strconv.ParseUint(*vol.Mode, 8, 32)
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("invalid mode %q for volume %q", *vol.Mode, vol.Name), err)
	}
	mode := os.FileMode(m)
	Uid := *vol.UID
	Gid := *vol.GID

	if dockerImplicit {
		fi, err := os.Stat(targetPath)
		if err == nil {
			// the directory exists in the image, let's set the same
			// permissions and copy files from there to the empty volume
			mode = fi.Mode()
			Uid = int(fi.Sys().(*syscall.Stat_t).Uid)
			Gid = int(fi.Sys().(*syscall.Stat_t).Gid)

			if err := fileutil.CopyTree(targetPath, volPath, user.NewBlankUidRange()); err != nil {
				return errwrap.Wrap(fmt.Errorf("error copying image files to empty volume %q", volPath), err)
			}
		}
	}

	if err := os.MkdirAll(volPath, 0770); err != nil {
		return errwrap.Wrap(fmt.Errorf("error creating %q", volPath), err)
	}
	if err := os.Chown(volPath, Uid, Gid); err != nil {
		return errwrap.Wrap(fmt.Errorf("could not change owner of %q", volPath), err)
	}
	if err := os.Chmod(volPath, mode); err != nil {
		return errwrap.Wrap(fmt.Errorf("could not change permissions of %q", volPath), err)
	}

	return nil
}

// BindMount, well, bind mounts a source in to a destination. This will
// do some bookkeeping:
// * evaluate all symlinks
// * ensure the source exists
// * recursively create the destination
func BindMount(mnt fs.MountUnmounter, source, destination string, readOnly bool) error {
	absSource, err := filepath.EvalSymlinks(source)
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("Could not resolve symlink for source %v", source), err)
	}

	if err := EnsureTargetExists(absSource, destination); err != nil {
		return errwrap.Wrap(fmt.Errorf("Could not create destination mount point: %v", destination), err)
	} else if err := mnt.Mount(absSource, destination, "bind", syscall.MS_BIND, ""); err != nil {
		return errwrap.Wrap(fmt.Errorf("Could not bind mount %v to %v", absSource, destination), err)
	}
	if readOnly {
		err := mnt.Mount(source, destination, "bind", syscall.MS_REMOUNT|syscall.MS_RDONLY|syscall.MS_BIND, "")

		// If we failed to remount ro, unmount
		if err != nil {
			mnt.Unmount(destination, 0) // if this fails, oh well
			return errwrap.Wrap(fmt.Errorf("Could not remount %v read-only", destination), err)
		}
	}
	return nil
}

// EnsureTargetExists will recursively create a given mountpoint. If directories
// are created, their permissions are initialized to common.SharedVolumePerm
func EnsureTargetExists(source, destination string) error {
	fileInfo, err := os.Stat(source)
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("could not stat source location: %v", source), err)
	}

	targetPathParent, _ := filepath.Split(destination)
	if err := os.MkdirAll(targetPathParent, common.SharedVolumePerm); err != nil {
		return errwrap.Wrap(fmt.Errorf("could not create parent directory: %v", targetPathParent), err)
	}

	if fileInfo.IsDir() {
		if err := os.Mkdir(destination, common.SharedVolumePerm); err != nil && !os.IsExist(err) {
			return errwrap.Wrap(errors.New("could not create destination directory "+destination), err)
		}
	} else {
		if file, err := os.OpenFile(destination, os.O_CREATE, common.SharedVolumePerm); err != nil {
			return errwrap.Wrap(errors.New("could not create destination file"), err)
		} else {
			file.Close()
		}
	}
	return nil
}
