// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package socks provides a SOCKS version 5 client implementation.
//
// SOCKS protocol version 5 is defined in RFC 1928.
// Username/Password authentication for SOCKS version 5 is defined in
// RFC 1929.
package socks

import (
	"context"
	"errors"
	"io"
	"net"
	"strconv"
)

// A Command represents a SOCKS command.
type Command int

func (cmd Command) String() string {
	switch cmd {
	case CmdConnect:
		return "socks connect"
	case cmdBind:
		return "socks bind"
	default:
		return "socks " + strconv.Itoa(int(cmd))
	}
}

// An AuthMethod represents a SOCKS authentication method.
type AuthMethod int

// A Reply represents a SOCKS command reply code.
type Reply int

func (code Reply) String() string {
	switch code {
	case StatusSucceeded:
		return "succeeded"
	case 0x01:
		return "general SOCKS server failure"
	case 0x02:
		return "connection not allowed by ruleset"
	case 0x03:
		return "network unreachable"
	case 0x04:
		return "host unreachable"
	case 0x05:
		return "connection refused"
	case 0x06:
		return "TTL expired"
	case 0x07:
		return "command not supported"
	case 0x08:
		return "address type not supported"
	default:
		return "unknown code: " + strconv.Itoa(int(code))
	}
}

// Wire protocol constants.
const (
	Version5 = 0x05

	AddrTypeIPv4 = 0x01
	AddrTypeFQDN = 0x03
	AddrTypeIPv6 = 0x04

	CmdConnect Command = 0x01 // establishes an active-open forward proxy connection
	cmdBind    Command = 0x02 // establishes a passive-open forward proxy connection

	AuthMethodNotRequired         AuthMethod = 0x00 // no authentication required
	AuthMethodUsernamePassword    AuthMethod = 0x02 // use username/password
	AuthMethodNoAcceptableMethods AuthMethod = 0xff // no acceptable authentication methods

	StatusSucceeded Reply = 0x00
)

// An Addr represents a SOCKS-specific address.
// Either Name or IP is used exclusively.
type Addr struct {
	Name string // fully-qualified domain name
	IP   net.IP
	Port int
}

func (a *Addr) Network() string { return "socks" }

func (a *Addr) String() string {
	if a == nil {
		return "<nil>"
	}
	port := strconv.Itoa(a.Port)
	if a.IP == nil {
		return net.JoinHostPort(a.Name, port)
	}
	return net.JoinHostPort(a.IP.String(), port)
}

// A Conn represents a forward proxy connection.
type Conn struct {
	net.Conn

	boundAddr net.Addr
}

// BoundAddr returns the address assigned by the proxy server for
// connecting to the command target address from the proxy server.
func (c *Conn) BoundAddr() net.Addr {
	if c == nil {
		return nil
	}
	return c.boundAddr
}

// A Dialer holds SOCKS-specific options.
type Dialer struct {
	cmd          Command // either CmdConnect or cmdBind
	proxyNetwork string  // network between a proxy server and a client
	proxyAddress string  // proxy server address

	// ProxyDial specifies the optional dial function for
	// establishing the transport connection.
	ProxyDial func(context.Context, string, string) (net.Conn, error)

	// AuthMethods specifies the list of request authention
	// methods.
	// If empty, SOCKS client requests only AuthMethodNotRequired.
	AuthMethods []AuthMethod

	// Authenticate specifies the optional authentication
	// function. It must be non-nil when AuthMethods is not empty.
	// It must return an error when the authentication is failed.
	Authenticate func(context.Context, io.ReadWriter, AuthMethod) error
}

// DialContext connects to the provided address on the provided
// network.
//
// The returned error value may be a net.OpError. When the Op field of
// net.OpError contains "socks", the Source field contains a proxy
// server address and the Addr field contains a command target
// address.
//
// See func Dial of the net package of standard library for a
// description of the network and address parameters.
func (d *Dialer) DialContext(ctx context.Context, network, address string) (net.Conn, error) {
	if err := d.validateTarget(network, address); err != nil {
		proxy, dst, _ := d.pathAddrs(address)
		return nil, &net.OpError{Op: d.cmd.String(), Net: network, Source: proxy, Addr: dst, Err: err}
	}
	if ctx == nil {
		proxy, dst, _ := d.pathAddrs(address)
		return nil, &net.OpError{Op: d.cmd.String(), Net: network, Source: proxy, Addr: dst, Err: errors.New("nil context")}
	}
	var err error
	var c net.Conn
	if d.ProxyDial != nil {
		c, err = d.ProxyDial(ctx, d.proxyNetwork, d.proxyAddress)
	} else {
		var dd net.Dialer
		c, err = dd.DialContext(ctx, d.proxyNetwork, d.proxyAddress)
	}
	if err != nil {
		proxy, dst, _ := d.pathAddrs(address)
		return nil, &net.OpError{Op: d.cmd.String(), Net: network, Source: proxy, Addr: dst, Err: err}
	}
	a, err := d.connect(ctx, c, address)
	if err != nil {
		c.Close()
		proxy, dst, _ := d.pathAddrs(address)
		return nil, &net.OpError{Op: d.cmd.String(), Net: network, Source: proxy, Addr: dst, Err: err}
	}
	return &Conn{Conn: c, boundAddr: a}, nil
}

// DialWithConn initiates a connection from SOCKS server to the target
// network and address using the connection c that is already
// connected to the SOCKS server.
//
// It returns the connection's local address assigned by the SOCKS
// server.
func (d *Dialer) DialWithConn(ctx context.Context, c net.Conn, network, address string) (net.Addr, error) {
	if err := d.validateTarget(network, address); err != nil {
		proxy, dst, _ := d.pathAddrs(address)
		return nil, &net.OpError{Op: d.cmd.String(), Net: network, Source: proxy, Addr: dst, Err: err}
	}
	if ctx == nil {
		proxy, dst, _ := d.pathAddrs(address)
		return nil, &net.OpError{Op: d.cmd.String(), Net: network, Source: proxy, Addr: dst, Err: errors.New("nil context")}
	}
	a, err := d.connect(ctx, c, address)
	if err != nil {
		proxy, dst, _ := d.pathAddrs(address)
		return nil, &net.OpError{Op: d.cmd.String(), Net: network, Source: proxy, Addr: dst, Err: err}
	}
	return a, nil
}

// Dial connects to the provided address on the provided network.
//
// Unlike DialContext, it returns a raw transport connection instead
// of a forward proxy connection.
//
// Deprecated: Use DialContext or DialWithConn instead.
func (d *Dialer) Dial(network, address string) (net.Conn, error) {
	if err := d.validateTarget(network, address); err != nil {
		proxy, dst, _ := d.pathAddrs(address)
		return nil, &net.OpError{Op: d.cmd.String(), Net: network, Source: proxy, Addr: dst, Err: err}
	}
	var err error
	var c net.Conn
	if d.ProxyDial != nil {
		c, err = d.ProxyDial(context.Background(), d.proxyNetwork, d.proxyAddress)
	} else {
		c, err = net.Dial(d.proxyNetwork, d.proxyAddress)
	}
	if err != nil {
		proxy, dst, _ := d.pathAddrs(address)
		return nil, &net.OpError{Op: d.cmd.String(), Net: network, Source: proxy, Addr: dst, Err: err}
	}
	if _, err := d.DialWithConn(context.Background(), c, network, address); err != nil {
		return nil, err
	}
	return c, nil
}

func (d *Dialer) validateTarget(network, address string) error {
	switch network {
	case "tcp", "tcp6", "tcp4":
	default:
		return errors.New("network not implemented")
	}
	switch d.cmd {
	case CmdConnect, cmdBind:
	default:
		return errors.New("command not implemented")
	}
	return nil
}

func (d *Dialer) pathAddrs(address string) (proxy, dst net.Addr, err error) {
	for i, s := range []string{d.proxyAddress, address} {
		host, port, err := splitHostPort(s)
		if err != nil {
			return nil, nil, err
		}
		a := &Addr{Port: port}
		a.IP = net.ParseIP(host)
		if a.IP == nil {
			a.Name = host
		}
		if i == 0 {
			proxy = a
		} else {
			dst = a
		}
	}
	return
}

// NewDialer returns a new Dialer that dials through the provided
// proxy server's network and address.
func NewDialer(network, address string) *Dialer {
	return &Dialer{proxyNetwork: network, proxyAddress: address, cmd: CmdConnect}
}

const (
	authUsernamePasswordVersion = 0x01
	authStatusSucceeded         = 0x00
)

// UsernamePassword are the credentials for the username/password
// authentication method.
type UsernamePassword struct {
	Username string
	Password string
}

// Authenticate authenticates a pair of username and password with the
// proxy server.
func (up *UsernamePassword) Authenticate(ctx context.Context, rw io.ReadWriter, auth AuthMethod) error {
	switch auth {
	case AuthMethodNotRequired:
		return nil
	case AuthMethodUsernamePassword:
		if len(up.Username) == 0 || len(up.Username) > 255 || len(up.Password) == 0 || len(up.Password) > 255 {
			return errors.New("invalid username/password")
		}
		b := []byte{authUsernamePasswordVersion}
		b = append(b, byte(len(up.Username)))
		b = append(b, up.Username...)
		b = append(b, byte(len(up.Password)))
		b = append(b, up.Password...)
		// TODO(mikio): handle IO deadlines and cancelation if
		// necessary
		if _, err := rw.Write(b); err != nil {
			return err
		}
		if _, err := io.ReadFull(rw, b[:2]); err != nil {
			return err
		}
		if b[0] != authUsernamePasswordVersion {
			return errors.New("invalid username/password version")
		}
		if b[1] != authStatusSucceeded {
			return errors.New("username/password authentication failed")
		}
		return nil
	}
	return errors.New("unsupported authentication method " + strconv.Itoa(int(auth)))
}
