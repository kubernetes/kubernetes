// Copyright 2014 The rkt Authors
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

//+build linux

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/appc/spec/schema"
	"github.com/coreos/go-systemd/util"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/networking"
	"github.com/coreos/rkt/pkg/mountinfo"
	stage1commontypes "github.com/coreos/rkt/stage1/common/types"
	stage1initcommon "github.com/coreos/rkt/stage1/init/common"
	"github.com/coreos/rkt/stage1/init/kvm"
	"github.com/hashicorp/errwrap"
)

const journalDir = "/var/log/journal"

// Supported hypervisors
var hypervisors = [...]string{"lkvm", "qemu"}

// KvmNetworkingToSystemd generates systemd unit files for a pod according to network configuration
func KvmNetworkingToSystemd(p *stage1commontypes.Pod, n *networking.Networking) error {
	podRoot := common.Stage1RootfsPath(p.Root)

	// networking
	netDescriptions := kvm.GetNetworkDescriptions(n)
	if err := kvm.GenerateNetworkInterfaceUnits(filepath.Join(podRoot, stage1initcommon.UnitsDir), netDescriptions); err != nil {
		return errwrap.Wrap(errors.New("failed to transform networking to units"), err)
	}

	return nil
}

func mountSharedVolumes(p *stage1commontypes.Pod, ra *schema.RuntimeApp) error {
	appName := ra.Name

	sharedVolPath, err := common.CreateSharedVolumesPath(p.Root)
	if err != nil {
		return err
	}

	imageManifest := p.Images[appName.String()]
	mounts, err := stage1initcommon.GenerateMounts(ra, p.Manifest.Volumes, stage1initcommon.ConvertedFromDocker(imageManifest))
	if err != nil {
		return err
	}
	for _, m := range mounts {
		absRoot, err := filepath.Abs(p.Root) // Absolute path to the pod's rootfs.
		if err != nil {
			return errwrap.Wrap(errors.New("could not get pod's root absolute path"), err)
		}

		absAppRootfs := common.AppRootfsPath(absRoot, appName)
		if err != nil {
			return fmt.Errorf(`could not evaluate absolute path for application rootfs in app: %v`, appName)
		}

		mntPath, err := stage1initcommon.EvaluateSymlinksInsideApp(absAppRootfs, m.Mount.Path)
		if err != nil {
			return errwrap.Wrap(fmt.Errorf("could not evaluate path %v", m.Mount.Path), err)
		}
		absDestination := filepath.Join(absAppRootfs, mntPath)
		shPath := filepath.Join(sharedVolPath, m.Volume.Name.String())
		if err := stage1initcommon.PrepareMountpoints(shPath, absDestination, &m.Volume, m.DockerImplicit); err != nil {
			return err
		}

		source := m.Source(p.Root)
		if cleanedSource, err := filepath.EvalSymlinks(source); err != nil {
			return errwrap.Wrap(fmt.Errorf("could not resolve symlink for source: %v", source), err)
		} else if err := ensureDestinationExists(cleanedSource, absDestination); err != nil {
			return errwrap.Wrap(fmt.Errorf("could not create destination mount point: %v", absDestination), err)
		} else if err := doBindMount(cleanedSource, absDestination, m.ReadOnly, m.Volume.Recursive); err != nil {
			return errwrap.Wrap(fmt.Errorf("could not bind mount path %v (s: %v, d: %v)", m.Mount.Path, source, absDestination), err)
		}
	}
	return nil
}

func doBindMount(source, destination string, readOnly bool, recursive *bool) error {
	var flags uintptr = syscall.MS_BIND

	// Enable recursive by default and remove it if explicitly requested
	recursiveBool := recursive == nil || *recursive == true
	if recursiveBool {
		flags |= syscall.MS_REC
	}

	if err := syscall.Mount(source, destination, "bind", flags, ""); err != nil {
		return errwrap.Wrap(fmt.Errorf("error mounting %s", destination), err)
	}

	// Linux can't bind-mount with readonly in a single operation, so remount +ro
	if readOnly {
		if err := syscall.Mount("", destination, "none", syscall.MS_REMOUNT|syscall.MS_RDONLY|syscall.MS_BIND, ""); err != nil {
			return errwrap.Wrap(fmt.Errorf("error remounting read-only %s", destination), err)
		}
	}

	if readOnly && recursiveBool {
		// Sub-mounts are still read-write, so find them and remount them read-only

		mnts, err := mountinfo.ParseMounts(0)
		if err != nil {
			return errwrap.Wrap(fmt.Errorf("error getting mounts under %q from mountinfo", source), err)
		}
		mnts = mnts.Filter(mountinfo.HasPrefix(source + "/"))

		for _, mnt := range mnts {
			innerAbsPath := destination + strings.Replace(mnt.MountPoint, source, "", -1)
			if err := syscall.Mount("", innerAbsPath, "none", syscall.MS_REMOUNT|syscall.MS_RDONLY|syscall.MS_BIND, ""); err != nil {
				return errwrap.Wrap(fmt.Errorf("error remounting child mount %s read-only", innerAbsPath), err)
			}
		}
	}

	return nil
}

func ensureDestinationExists(source, destination string) error {
	fileInfo, err := os.Stat(source)
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("could not stat source location: %v", source), err)
	}

	targetPathParent, _ := filepath.Split(destination)
	if err := os.MkdirAll(targetPathParent, common.SharedVolumePerm); err != nil {
		return errwrap.Wrap(fmt.Errorf("could not create parent directory: %v", targetPathParent), err)
	}

	if fileInfo.IsDir() {
		if err := os.Mkdir(destination, common.SharedVolumePerm); !os.IsExist(err) {
			return err
		}
	} else {
		if file, err := os.OpenFile(destination, os.O_CREATE, common.SharedVolumePerm); err != nil {
			return err
		} else {
			file.Close()
		}
	}
	return nil
}

func prepareMountsForApp(p *stage1commontypes.Pod, ra *schema.RuntimeApp) error {
	// bind mount all shared volumes (we don't use mechanism for bind-mounting given by nspawn)
	if err := mountSharedVolumes(p, ra); err != nil {
		return errwrap.Wrap(errors.New("failed to prepare mount point"), err)
	}

	return nil
}

func KvmPrepareMounts(p *stage1commontypes.Pod) error {
	for i := range p.Manifest.Apps {
		ra := &p.Manifest.Apps[i]
		if err := prepareMountsForApp(p, ra); err != nil {
			return errwrap.Wrap(fmt.Errorf("failed prepare mounts for app %q", ra.Name), err)
		}
	}

	return nil
}

func KvmCheckHypervisor(s1Root string) (string, error) {
	for _, hv := range hypervisors {
		if _, err := os.Stat(filepath.Join(s1Root, hv)); err == nil {
			return hv, nil
		}
	}
	return "", fmt.Errorf("unrecognized hypervisor")
}

func linkJournal(s1Root, machineID string) error {
	if !util.IsRunningSystemd() {
		return nil
	}

	absS1Root, err := filepath.Abs(s1Root)
	if err != nil {
		return err
	}

	// /var/log/journal doesn't exist on the host, don't do anything
	if _, err := os.Stat(journalDir); os.IsNotExist(err) {
		return nil
	}

	machineJournalDir := filepath.Join(journalDir, machineID)
	podJournalDir := filepath.Join(absS1Root, machineJournalDir)

	hostMachineID, err := util.GetMachineID()
	if err != nil {
		return err
	}

	// unlikely, machine ID is random (== pod UUID)
	if hostMachineID == machineID {
		return fmt.Errorf("host and pod machine IDs are equal (%s)", machineID)
	}

	fi, err := os.Lstat(machineJournalDir)
	switch {
	case os.IsNotExist(err):
		// good, we'll create the symlink
	case err != nil:
		return err
	// unlikely, machine ID is random (== pod UUID)
	default:
		if fi.IsDir() {
			if err := os.Remove(machineJournalDir); err != nil {
				return err
			}
		}

		link, err := os.Readlink(machineJournalDir)
		if err != nil {
			return err
		}

		if link == podJournalDir {
			return nil
		} else {
			if err := os.Remove(machineJournalDir); err != nil {
				return err
			}
		}
	}

	if err := os.Symlink(podJournalDir, machineJournalDir); err != nil {
		return err
	}

	return nil
}
