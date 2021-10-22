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
	"testing"

	"github.com/vmware/govmomi/simulator/vpx"
	"github.com/vmware/govmomi/vim25/types"
)

func TestAuthorizationManager(t *testing.T) {
	for i := 0; i < 2; i++ {
		model := VPX()

		_ = New(NewServiceInstance(model.ServiceContent, model.RootFolder)) // 2nd pass panics w/o copying RoleList

		authz := Map.Get(*vpx.ServiceContent.AuthorizationManager).(*AuthorizationManager)
		authz.RemoveAuthorizationRole(&types.RemoveAuthorizationRole{
			RoleId: -2, // ReadOnly
		})
	}
}
