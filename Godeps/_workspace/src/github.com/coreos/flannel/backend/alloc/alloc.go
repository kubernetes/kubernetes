package alloc

import (
	"fmt"
	"net"

	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"
	"github.com/coreos/flannel/backend"
	"github.com/coreos/flannel/pkg/ip"
	"github.com/coreos/flannel/subnet"
)

type AllocBackend struct {
	sm       subnet.Manager
	publicIP ip.IP4
	mtu      int
	lease    *subnet.Lease
}

func New(sm subnet.Manager, extIface *net.Interface, extIaddr net.IP, extEaddr net.IP) (backend.Backend, error) {
	be := AllocBackend{
		sm:      sm,
		publicIP: ip.FromIP(extEaddr),
		mtu:      extIface.MTU,
	}
	return &be, nil
}

func (m *AllocBackend) RegisterNetwork(ctx context.Context, network string, config *subnet.Config) (*backend.SubnetDef, error) {
	attrs := subnet.LeaseAttrs{
		PublicIP: m.publicIP,
	}

	l, err := m.sm.AcquireLease(ctx, network, &attrs)
	switch err {
	case nil:
		m.lease = l
		return &backend.SubnetDef{
			Lease: l,
			MTU:   m.mtu,
		}, nil

	case context.Canceled, context.DeadlineExceeded:
		return nil, err

	default:
		return nil, fmt.Errorf("failed to acquire lease: %v", err)
	}
}

func (m *AllocBackend) Run(ctx context.Context) {
}
