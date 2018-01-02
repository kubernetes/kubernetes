/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package esxcli

import (
	"context"
	"strings"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type hostInfo struct {
	*Executor
	wids map[string]string
}

type GuestInfo struct {
	c     *vim25.Client
	hosts map[string]*hostInfo
}

func NewGuestInfo(c *vim25.Client) *GuestInfo {
	return &GuestInfo{
		c:     c,
		hosts: make(map[string]*hostInfo),
	}
}

func (g *GuestInfo) hostInfo(ref *types.ManagedObjectReference) (*hostInfo, error) {
	// cache exectuor and uuid -> worldid map
	if h, ok := g.hosts[ref.Value]; ok {
		return h, nil
	}

	host := object.NewHostSystem(g.c, *ref)

	e, err := NewExecutor(g.c, host)
	if err != nil {
		return nil, err
	}

	res, err := e.Run([]string{"vm", "process", "list"})
	if err != nil {
		return nil, err
	}

	ids := make(map[string]string, len(res.Values))

	for _, process := range res.Values {
		// Normalize uuid, esxcli and mo.VirtualMachine have different formats
		uuid := strings.Replace(process["UUID"][0], " ", "", -1)
		uuid = strings.Replace(uuid, "-", "", -1)

		ids[uuid] = process["WorldID"][0]
	}

	h := &hostInfo{e, ids}
	g.hosts[ref.Value] = h

	return h, nil
}

// IpAddress attempts to find the guest IP address using esxcli.
// ESX hosts must be configured with the /Net/GuestIPHack enabled.
// For example:
// $ govc host.esxcli -- system settings advanced set -o /Net/GuestIPHack -i 1
func (g *GuestInfo) IpAddress(vm *object.VirtualMachine) (string, error) {
	ctx := context.TODO()
	const any = "0.0.0.0"
	var mvm mo.VirtualMachine

	pc := property.DefaultCollector(g.c)
	err := pc.RetrieveOne(ctx, vm.Reference(), []string{"runtime.host", "config.uuid"}, &mvm)
	if err != nil {
		return "", err
	}

	h, err := g.hostInfo(mvm.Runtime.Host)
	if err != nil {
		return "", err
	}

	// Normalize uuid, esxcli and mo.VirtualMachine have different formats
	uuid := strings.Replace(mvm.Config.Uuid, "-", "", -1)

	if wid, ok := h.wids[uuid]; ok {
		res, err := h.Run([]string{"network", "vm", "port", "list", "--world-id", wid})
		if err != nil {
			return "", err
		}

		for _, val := range res.Values {
			if ip, ok := val["IPAddress"]; ok {
				if ip[0] != any {
					return ip[0], nil
				}
			}
		}
	}

	return any, nil
}
