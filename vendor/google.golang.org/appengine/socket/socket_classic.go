// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build appengine

package socket

import (
	"fmt"
	"io"
	"net"
	"strconv"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"
	"google.golang.org/appengine/internal"

	pb "google.golang.org/appengine/internal/socket"
)

// Dial connects to the address addr on the network protocol.
// The address format is host:port, where host may be a hostname or an IP address.
// Known protocols are "tcp" and "udp".
// The returned connection satisfies net.Conn, and is valid while ctx is valid;
// if the connection is to be used after ctx becomes invalid, invoke SetContext
// with the new context.
func Dial(ctx context.Context, protocol, addr string) (*Conn, error) {
	return DialTimeout(ctx, protocol, addr, 0)
}

var ipFamilies = []pb.CreateSocketRequest_SocketFamily{
	pb.CreateSocketRequest_IPv4,
	pb.CreateSocketRequest_IPv6,
}

// DialTimeout is like Dial but takes a timeout.
// The timeout includes name resolution, if required.
func DialTimeout(ctx context.Context, protocol, addr string, timeout time.Duration) (*Conn, error) {
	dialCtx := ctx // Used for dialing and name resolution, but not stored in the *Conn.
	if timeout > 0 {
		var cancel context.CancelFunc
		dialCtx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	host, portStr, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, err
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return nil, fmt.Errorf("socket: bad port %q: %v", portStr, err)
	}

	var prot pb.CreateSocketRequest_SocketProtocol
	switch protocol {
	case "tcp":
		prot = pb.CreateSocketRequest_TCP
	case "udp":
		prot = pb.CreateSocketRequest_UDP
	default:
		return nil, fmt.Errorf("socket: unknown protocol %q", protocol)
	}

	packedAddrs, resolved, err := resolve(dialCtx, ipFamilies, host)
	if err != nil {
		return nil, fmt.Errorf("socket: failed resolving %q: %v", host, err)
	}
	if len(packedAddrs) == 0 {
		return nil, fmt.Errorf("no addresses for %q", host)
	}

	packedAddr := packedAddrs[0] // use first address
	fam := pb.CreateSocketRequest_IPv4
	if len(packedAddr) == net.IPv6len {
		fam = pb.CreateSocketRequest_IPv6
	}

	req := &pb.CreateSocketRequest{
		Family:   fam.Enum(),
		Protocol: prot.Enum(),
		RemoteIp: &pb.AddressPort{
			Port:          proto.Int32(int32(port)),
			PackedAddress: packedAddr,
		},
	}
	if resolved {
		req.RemoteIp.HostnameHint = &host
	}
	res := &pb.CreateSocketReply{}
	if err := internal.Call(dialCtx, "remote_socket", "CreateSocket", req, res); err != nil {
		return nil, err
	}

	return &Conn{
		ctx:    ctx,
		desc:   res.GetSocketDescriptor(),
		prot:   prot,
		local:  res.ProxyExternalIp,
		remote: req.RemoteIp,
	}, nil
}

// LookupIP returns the given host's IP addresses.
func LookupIP(ctx context.Context, host string) (addrs []net.IP, err error) {
	packedAddrs, _, err := resolve(ctx, ipFamilies, host)
	if err != nil {
		return nil, fmt.Errorf("socket: failed resolving %q: %v", host, err)
	}
	addrs = make([]net.IP, len(packedAddrs))
	for i, pa := range packedAddrs {
		addrs[i] = net.IP(pa)
	}
	return addrs, nil
}

func resolve(ctx context.Context, fams []pb.CreateSocketRequest_SocketFamily, host string) ([][]byte, bool, error) {
	// Check if it's an IP address.
	if ip := net.ParseIP(host); ip != nil {
		if ip := ip.To4(); ip != nil {
			return [][]byte{ip}, false, nil
		}
		return [][]byte{ip}, false, nil
	}

	req := &pb.ResolveRequest{
		Name:            &host,
		AddressFamilies: fams,
	}
	res := &pb.ResolveReply{}
	if err := internal.Call(ctx, "remote_socket", "Resolve", req, res); err != nil {
		// XXX: need to map to pb.ResolveReply_ErrorCode?
		return nil, false, err
	}
	return res.PackedAddress, true, nil
}

// withDeadline is like context.WithDeadline, except it ignores the zero deadline.
func withDeadline(parent context.Context, deadline time.Time) (context.Context, context.CancelFunc) {
	if deadline.IsZero() {
		return parent, func() {}
	}
	return context.WithDeadline(parent, deadline)
}

// Conn represents a socket connection.
// It implements net.Conn.
type Conn struct {
	ctx    context.Context
	desc   string
	offset int64

	prot          pb.CreateSocketRequest_SocketProtocol
	local, remote *pb.AddressPort

	readDeadline, writeDeadline time.Time // optional
}

// SetContext sets the context that is used by this Conn.
// It is usually used only when using a Conn that was created in a different context,
// such as when a connection is created during a warmup request but used while
// servicing a user request.
func (cn *Conn) SetContext(ctx context.Context) {
	cn.ctx = ctx
}

func (cn *Conn) Read(b []byte) (n int, err error) {
	const maxRead = 1 << 20
	if len(b) > maxRead {
		b = b[:maxRead]
	}

	req := &pb.ReceiveRequest{
		SocketDescriptor: &cn.desc,
		DataSize:         proto.Int32(int32(len(b))),
	}
	res := &pb.ReceiveReply{}
	if !cn.readDeadline.IsZero() {
		req.TimeoutSeconds = proto.Float64(cn.readDeadline.Sub(time.Now()).Seconds())
	}
	ctx, cancel := withDeadline(cn.ctx, cn.readDeadline)
	defer cancel()
	if err := internal.Call(ctx, "remote_socket", "Receive", req, res); err != nil {
		return 0, err
	}
	if len(res.Data) == 0 {
		return 0, io.EOF
	}
	if len(res.Data) > len(b) {
		return 0, fmt.Errorf("socket: internal error: read too much data: %d > %d", len(res.Data), len(b))
	}
	return copy(b, res.Data), nil
}

func (cn *Conn) Write(b []byte) (n int, err error) {
	const lim = 1 << 20 // max per chunk

	for n < len(b) {
		chunk := b[n:]
		if len(chunk) > lim {
			chunk = chunk[:lim]
		}

		req := &pb.SendRequest{
			SocketDescriptor: &cn.desc,
			Data:             chunk,
			StreamOffset:     &cn.offset,
		}
		res := &pb.SendReply{}
		if !cn.writeDeadline.IsZero() {
			req.TimeoutSeconds = proto.Float64(cn.writeDeadline.Sub(time.Now()).Seconds())
		}
		ctx, cancel := withDeadline(cn.ctx, cn.writeDeadline)
		defer cancel()
		if err = internal.Call(ctx, "remote_socket", "Send", req, res); err != nil {
			// assume zero bytes were sent in this RPC
			break
		}
		n += int(res.GetDataSent())
		cn.offset += int64(res.GetDataSent())
	}

	return
}

func (cn *Conn) Close() error {
	req := &pb.CloseRequest{
		SocketDescriptor: &cn.desc,
	}
	res := &pb.CloseReply{}
	if err := internal.Call(cn.ctx, "remote_socket", "Close", req, res); err != nil {
		return err
	}
	cn.desc = "CLOSED"
	return nil
}

func addr(prot pb.CreateSocketRequest_SocketProtocol, ap *pb.AddressPort) net.Addr {
	if ap == nil {
		return nil
	}
	switch prot {
	case pb.CreateSocketRequest_TCP:
		return &net.TCPAddr{
			IP:   net.IP(ap.PackedAddress),
			Port: int(*ap.Port),
		}
	case pb.CreateSocketRequest_UDP:
		return &net.UDPAddr{
			IP:   net.IP(ap.PackedAddress),
			Port: int(*ap.Port),
		}
	}
	panic("unknown protocol " + prot.String())
}

func (cn *Conn) LocalAddr() net.Addr  { return addr(cn.prot, cn.local) }
func (cn *Conn) RemoteAddr() net.Addr { return addr(cn.prot, cn.remote) }

func (cn *Conn) SetDeadline(t time.Time) error {
	cn.readDeadline = t
	cn.writeDeadline = t
	return nil
}

func (cn *Conn) SetReadDeadline(t time.Time) error {
	cn.readDeadline = t
	return nil
}

func (cn *Conn) SetWriteDeadline(t time.Time) error {
	cn.writeDeadline = t
	return nil
}

// KeepAlive signals that the connection is still in use.
// It may be called to prevent the socket being closed due to inactivity.
func (cn *Conn) KeepAlive() error {
	req := &pb.GetSocketNameRequest{
		SocketDescriptor: &cn.desc,
	}
	res := &pb.GetSocketNameReply{}
	return internal.Call(cn.ctx, "remote_socket", "GetSocketName", req, res)
}

func init() {
	internal.RegisterErrorCodeMap("remote_socket", pb.RemoteSocketServiceError_ErrorCode_name)
}
