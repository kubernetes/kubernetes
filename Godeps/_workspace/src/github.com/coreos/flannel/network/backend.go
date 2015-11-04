package network

import (
	"fmt"
	"net"
	"strings"

	"github.com/coreos/flannel/backend"
	"github.com/coreos/flannel/backend/alloc"
	"github.com/coreos/flannel/backend/awsvpc"
	"github.com/coreos/flannel/backend/gce"
	"github.com/coreos/flannel/backend/hostgw"
	"github.com/coreos/flannel/backend/udp"
	"github.com/coreos/flannel/backend/vxlan"
	"github.com/coreos/flannel/subnet"
)

type beNewFunc func(sm subnet.Manager, extIface *net.Interface, extIaddr net.IP, extEaddr net.IP) (backend.Backend, error)

var backendMap = map[string]beNewFunc {
	"udp":     udp.New,
	"alloc":   alloc.New,
	"host-gw": hostgw.New,
	"vxlan":   vxlan.New,
	"aws-vpc": awsvpc.New,
	"gce":     gce.New,
}

func newBackend(sm subnet.Manager, backendType string, extIface *net.Interface, extIaddr net.IP, extEaddr net.IP) (backend.Backend, error) {
	betype := strings.ToLower(backendType)
	befunc, ok := backendMap[betype]
	if !ok {
		return nil, fmt.Errorf("unknown backend type")
	}
	return befunc(sm, extIface, extIaddr, extEaddr)
}
