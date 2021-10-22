/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"context"
	"testing"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/simulator/esx"
)

func TestHostFirewallSystem(t *testing.T) {
	ctx := context.Background()

	m := ESX()
	m.Datastore = 0
	m.Machine = 0

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	c := m.Service.client

	host := object.NewHostSystem(c, esx.HostSystem.Reference())

	hfs, _ := host.ConfigManager().FirewallSystem(ctx)

	err = hfs.DisableRuleset(ctx, "enoent")
	if err == nil {
		t.Error("expected error")
	}

	err = hfs.EnableRuleset(ctx, "enoent")
	if err == nil {
		t.Error("expected error")
	}

	err = hfs.DisableRuleset(ctx, "sshServer")
	if err != nil {
		t.Error(err)
	}

	err = hfs.EnableRuleset(ctx, "sshServer")
	if err != nil {
		t.Error(err)
	}

	_, err = hfs.Info(ctx)
	if err != nil {
		t.Fatal(err)
	}
}
