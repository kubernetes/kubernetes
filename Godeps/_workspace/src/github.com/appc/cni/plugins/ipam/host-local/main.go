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

package main

import (
	"errors"

	"github.com/appc/cni/plugins/ipam/host-local/backend/disk"

	"github.com/appc/cni/pkg/skel"
	"github.com/appc/cni/pkg/types"
)

func main() {
	skel.PluginMain(cmdAdd, cmdDel)
}

func cmdAdd(args *skel.CmdArgs) error {
	ipamConf, err := LoadIPAMConfig(args.StdinData)
	if err != nil {
		return err
	}

	store, err := disk.New(ipamConf.Name)
	if err != nil {
		return err
	}
	defer store.Close()

	allocator, err := NewIPAllocator(ipamConf, store)
	if err != nil {
		return err
	}

	var ipConf *types.IPConfig

	switch ipamConf.Type {
	case "host-local":
		ipConf, err = allocator.Get(args.ContainerID)
	case "host-local-ptp":
		ipConf, err = allocator.GetPtP(args.ContainerID)
	default:
		return errors.New("Unsupported IPAM plugin type")
	}

	if err != nil {
		return err
	}

	r := &types.Result{
		IP4: ipConf,
	}
	return r.Print()
}

func cmdDel(args *skel.CmdArgs) error {
	ipamConf, err := LoadIPAMConfig(args.StdinData)
	if err != nil {
		return err
	}

	store, err := disk.New(ipamConf.Name)
	if err != nil {
		return err
	}
	defer store.Close()

	allocator, err := NewIPAllocator(ipamConf, store)
	if err != nil {
		return err
	}

	return allocator.Release(args.ContainerID)
}
