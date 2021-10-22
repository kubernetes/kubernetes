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

package license

import (
	"context"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type AssignmentManager struct {
	object.Common
}

func (m AssignmentManager) QueryAssigned(ctx context.Context, id string) ([]types.LicenseAssignmentManagerLicenseAssignment, error) {
	req := types.QueryAssignedLicenses{
		This:     m.Reference(),
		EntityId: id,
	}

	res, err := methods.QueryAssignedLicenses(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (m AssignmentManager) Remove(ctx context.Context, id string) error {
	req := types.RemoveAssignedLicense{
		This:     m.Reference(),
		EntityId: id,
	}

	_, err := methods.RemoveAssignedLicense(ctx, m.Client(), &req)

	return err
}

func (m AssignmentManager) Update(ctx context.Context, id string, key string, name string) (*types.LicenseManagerLicenseInfo, error) {
	req := types.UpdateAssignedLicense{
		This:              m.Reference(),
		Entity:            id,
		LicenseKey:        key,
		EntityDisplayName: name,
	}

	res, err := methods.UpdateAssignedLicense(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}
