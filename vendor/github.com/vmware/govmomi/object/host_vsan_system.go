/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package object

import (
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

type HostVsanSystem struct {
	Common
}

func NewHostVsanSystem(c *vim25.Client, ref types.ManagedObjectReference) *HostVsanSystem {
	return &HostVsanSystem{
		Common: NewCommon(c, ref),
	}
}

func (s HostVsanSystem) Update(ctx context.Context, config types.VsanHostConfigInfo) (*Task, error) {
	req := types.UpdateVsan_Task{
		This:   s.Reference(),
		Config: config,
	}

	res, err := methods.UpdateVsan_Task(ctx, s.Client(), &req)
	if err != nil {
		return nil, err
	}

	return NewTask(s.Client(), res.Returnval), nil
}

// updateVnic in support of the HostVirtualNicManager.{SelectVnic,DeselectVnic} methods
func (s HostVsanSystem) updateVnic(ctx context.Context, device string, enable bool) error {
	var vsan mo.HostVsanSystem

	err := s.Properties(ctx, s.Reference(), []string{"config.networkInfo.port"}, &vsan)
	if err != nil {
		return err
	}

	info := vsan.Config

	var port []types.VsanHostConfigInfoNetworkInfoPortConfig

	for _, p := range info.NetworkInfo.Port {
		if p.Device == device {
			continue
		}

		port = append(port, p)
	}

	if enable {
		port = append(port, types.VsanHostConfigInfoNetworkInfoPortConfig{
			Device: device,
		})
	}

	info.NetworkInfo.Port = port

	task, err := s.Update(ctx, info)
	if err != nil {
		return err
	}

	_, err = task.WaitForResult(ctx, nil)
	return err
}
