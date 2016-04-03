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
	"path/filepath"

	"github.com/appc/spec/schema/types"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/networking"
	stage1commontypes "github.com/coreos/rkt/stage1/common/types"
	stage1initcommon "github.com/coreos/rkt/stage1/init/common"
	"github.com/coreos/rkt/stage1/init/kvm"
	"github.com/hashicorp/errwrap"
)

// KvmPodToSystemd generates systemd unit files for a pod according to the manifest and network configuration
func KvmPodToSystemd(p *stage1commontypes.Pod, n *networking.Networking) error {
	podRoot := common.Stage1RootfsPath(p.Root)

	// networking
	netDescriptions := kvm.GetNetworkDescriptions(n)
	if err := kvm.GenerateNetworkInterfaceUnits(filepath.Join(podRoot, stage1initcommon.UnitsDir), netDescriptions); err != nil {
		return errwrap.Wrap(errors.New("failed to transform networking to units"), err)
	}

	// volumes
	// prepare all applications names to become dependency for mount units
	// all host-shared folder has to become available before applications starts
	appNames := []types.ACName{}
	for _, runtimeApp := range p.Manifest.Apps {
		appNames = append(appNames, runtimeApp.Name)
	}
	// mount host volumes through some remote file system e.g. 9p to /mnt/volumeName location
	// order is important here: PodToSystemHostMountUnits prepares folders that are checked by each appToSystemdMountUnits later
	if err := stage1initcommon.PodToSystemdHostMountUnits(podRoot, p.Manifest.Volumes, appNames, stage1initcommon.UnitsDir); err != nil {
		return errwrap.Wrap(errors.New("failed to transform pod volumes into mount units"), err)
	}

	return nil
}
