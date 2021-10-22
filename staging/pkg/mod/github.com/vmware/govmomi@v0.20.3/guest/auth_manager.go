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

package guest

import (
	"context"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type AuthManager struct {
	types.ManagedObjectReference

	vm types.ManagedObjectReference

	c *vim25.Client
}

func (m AuthManager) Reference() types.ManagedObjectReference {
	return m.ManagedObjectReference
}

func (m AuthManager) AcquireCredentials(ctx context.Context, requestedAuth types.BaseGuestAuthentication, sessionID int64) (types.BaseGuestAuthentication, error) {
	req := types.AcquireCredentialsInGuest{
		This:          m.Reference(),
		Vm:            m.vm,
		RequestedAuth: requestedAuth,
		SessionID:     sessionID,
	}

	res, err := methods.AcquireCredentialsInGuest(ctx, m.c, &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

func (m AuthManager) ReleaseCredentials(ctx context.Context, auth types.BaseGuestAuthentication) error {
	req := types.ReleaseCredentialsInGuest{
		This: m.Reference(),
		Vm:   m.vm,
		Auth: auth,
	}

	_, err := methods.ReleaseCredentialsInGuest(ctx, m.c, &req)

	return err
}

func (m AuthManager) ValidateCredentials(ctx context.Context, auth types.BaseGuestAuthentication) error {
	req := types.ValidateCredentialsInGuest{
		This: m.Reference(),
		Vm:   m.vm,
		Auth: auth,
	}

	_, err := methods.ValidateCredentialsInGuest(ctx, m.c, &req)
	if err != nil {
		return err
	}

	return nil
}
