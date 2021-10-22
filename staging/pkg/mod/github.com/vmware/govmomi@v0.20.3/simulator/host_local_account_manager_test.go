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

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

func TestHostLocalAccountManager(t *testing.T) {
	ctx := context.Background()
	m := ESX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	ts := m.Service.NewServer()
	defer ts.Close()

	c, err := govmomi.NewClient(ctx, ts.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	ref := types.ManagedObjectReference{Type: "HostLocalAccountManager", Value: "ha-localacctmgr"}

	createUserReq := &types.CreateUser{
		This: ref,
		User: &types.HostAccountSpec{
			Id: "userid",
		},
	}

	_, err = methods.CreateUser(ctx, c.Client, createUserReq)
	if err != nil {
		t.Fatal(err)
	}

	_, err = methods.CreateUser(ctx, c.Client, createUserReq)
	if err == nil {
		t.Fatal("expect err; got nil")
	}

	updateUserReq := &types.UpdateUser{
		This: ref,
		User: &types.HostAccountSpec{
			Id: "userid",
		},
	}

	_, err = methods.UpdateUser(ctx, c.Client, updateUserReq)
	if err != nil {
		t.Fatal(err)
	}

	removeUserReq := &types.RemoveUser{
		This:     ref,
		UserName: "userid",
	}

	_, err = methods.RemoveUser(ctx, c.Client, removeUserReq)
	if err != nil {
		t.Fatal(err)
	}

	_, err = methods.RemoveUser(ctx, c.Client, removeUserReq)
	if err == nil {
		t.Fatal("expect err; got nil")
	}
}
