// Package agent provides a logical endpoint for Consul agents in the
// network.  agent data originates from Serf gossip and is primarily used to
// communicate Consul server information.  Gossiped information that ends up
// in Server contains the necessary metadata required for servers.Manager to
// select which server an RPC request should be routed to.
package agent

import (
	"fmt"
	"net"
	"strconv"

	"github.com/hashicorp/serf/serf"
)

// Key is used in maps and for equality tests.  A key is based on endpoints.
type Key struct {
	name string
}

// Equal compares two Key objects
func (k *Key) Equal(x *Key) bool {
	return k.name == x.name
}

// Server is used to return details of a consul server
type Server struct {
	Name       string
	Datacenter string
	Port       int
	Bootstrap  bool
	Expect     int
	Version    int
	Addr       net.Addr
}

// Key returns the corresponding Key
func (s *Server) Key() *Key {
	return &Key{
		name: s.Name,
	}
}

// String returns a string representation of Server
func (s *Server) String() string {
	var addrStr, networkStr string
	if s.Addr != nil {
		addrStr = s.Addr.String()
		networkStr = s.Addr.Network()
	}

	return fmt.Sprintf("%s (Addr: %s/%s) (DC: %s)", s.Name, networkStr, addrStr, s.Datacenter)
}

// IsConsulServer returns true if a serf member is a consul server
// agent. Returns a bool and a pointer to the Server.
func IsConsulServer(m serf.Member) (bool, *Server) {
	if m.Tags["role"] != "consul" {
		return false, nil
	}

	datacenter := m.Tags["dc"]
	_, bootstrap := m.Tags["bootstrap"]

	expect := 0
	expect_str, ok := m.Tags["expect"]
	var err error
	if ok {
		expect, err = strconv.Atoi(expect_str)
		if err != nil {
			return false, nil
		}
	}

	port_str := m.Tags["port"]
	port, err := strconv.Atoi(port_str)
	if err != nil {
		return false, nil
	}

	vsn_str := m.Tags["vsn"]
	vsn, err := strconv.Atoi(vsn_str)
	if err != nil {
		return false, nil
	}

	addr := &net.TCPAddr{IP: m.Addr, Port: port}

	parts := &Server{
		Name:       m.Name,
		Datacenter: datacenter,
		Port:       port,
		Bootstrap:  bootstrap,
		Expect:     expect,
		Addr:       addr,
		Version:    vsn,
	}
	return true, parts
}
