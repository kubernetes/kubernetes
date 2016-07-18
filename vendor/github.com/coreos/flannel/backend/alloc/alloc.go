// Copyright 2015 flannel authors
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

package alloc

import (
	"fmt"

	"golang.org/x/net/context"
	"github.com/coreos/flannel/backend"
	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/subnet"
)

func init() {
	backend.Register("alloc", New)
}

type AllocBackend struct {
	sm       subnet.Manager
	extIface *backend.ExternalInterface
}

func New(sm subnet.Manager, extIface *backend.ExternalInterface) (backend.Backend, error) {
	be := AllocBackend{
		sm:       sm,
		extIface: extIface,
	}
	return &be, nil
}

func (_ *AllocBackend) Run(ctx context.Context) {
	<-ctx.Done()
}

func (be *AllocBackend) RegisterNetwork(ctx context.Context, network string, config *subnet.Config) (backend.Network, error) {
	attrs := subnet.LeaseAttrs{
		PublicIP: ip.FromIP(be.extIface.ExtAddr),
	}

	l, err := be.sm.AcquireLease(ctx, network, &attrs)
	switch err {
	case nil:
		return &backend.SimpleNetwork{
			SubnetLease: l,
			ExtIface:    be.extIface,
		}, nil

	case context.Canceled, context.DeadlineExceeded:
		return nil, err

	default:
		return nil, fmt.Errorf("failed to acquire lease: %v", err)
	}
}
