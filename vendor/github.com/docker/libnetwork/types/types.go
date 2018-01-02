// Package types contains types that are common across libnetwork project
package types

import (
	"bytes"
	"fmt"
	"net"
	"strconv"
	"strings"
)

// constants for the IP address type
const (
	IP = iota // IPv4 and IPv6
	IPv4
	IPv6
)

// EncryptionKey is the libnetwork representation of the key distributed by the lead
// manager.
type EncryptionKey struct {
	Subsystem   string
	Algorithm   int32
	Key         []byte
	LamportTime uint64
}

// UUID represents a globally unique ID of various resources like network and endpoint
type UUID string

// QosPolicy represents a quality of service policy on an endpoint
type QosPolicy struct {
	MaxEgressBandwidth uint64
}

// TransportPort represents a local Layer 4 endpoint
type TransportPort struct {
	Proto Protocol
	Port  uint16
}

// Equal checks if this instance of Transportport is equal to the passed one
func (t *TransportPort) Equal(o *TransportPort) bool {
	if t == o {
		return true
	}

	if o == nil {
		return false
	}

	if t.Proto != o.Proto || t.Port != o.Port {
		return false
	}

	return true
}

// GetCopy returns a copy of this TransportPort structure instance
func (t *TransportPort) GetCopy() TransportPort {
	return TransportPort{Proto: t.Proto, Port: t.Port}
}

// String returns the TransportPort structure in string form
func (t *TransportPort) String() string {
	return fmt.Sprintf("%s/%d", t.Proto.String(), t.Port)
}

// FromString reads the TransportPort structure from string
func (t *TransportPort) FromString(s string) error {
	ps := strings.Split(s, "/")
	if len(ps) == 2 {
		t.Proto = ParseProtocol(ps[0])
		if p, err := strconv.ParseUint(ps[1], 10, 16); err == nil {
			t.Port = uint16(p)
			return nil
		}
	}
	return BadRequestErrorf("invalid format for transport port: %s", s)
}

// PortBinding represents a port binding between the container and the host
type PortBinding struct {
	Proto       Protocol
	IP          net.IP
	Port        uint16
	HostIP      net.IP
	HostPort    uint16
	HostPortEnd uint16
}

// HostAddr returns the host side transport address
func (p PortBinding) HostAddr() (net.Addr, error) {
	switch p.Proto {
	case UDP:
		return &net.UDPAddr{IP: p.HostIP, Port: int(p.HostPort)}, nil
	case TCP:
		return &net.TCPAddr{IP: p.HostIP, Port: int(p.HostPort)}, nil
	default:
		return nil, ErrInvalidProtocolBinding(p.Proto.String())
	}
}

// ContainerAddr returns the container side transport address
func (p PortBinding) ContainerAddr() (net.Addr, error) {
	switch p.Proto {
	case UDP:
		return &net.UDPAddr{IP: p.IP, Port: int(p.Port)}, nil
	case TCP:
		return &net.TCPAddr{IP: p.IP, Port: int(p.Port)}, nil
	default:
		return nil, ErrInvalidProtocolBinding(p.Proto.String())
	}
}

// GetCopy returns a copy of this PortBinding structure instance
func (p *PortBinding) GetCopy() PortBinding {
	return PortBinding{
		Proto:       p.Proto,
		IP:          GetIPCopy(p.IP),
		Port:        p.Port,
		HostIP:      GetIPCopy(p.HostIP),
		HostPort:    p.HostPort,
		HostPortEnd: p.HostPortEnd,
	}
}

// String returns the PortBinding structure in string form
func (p *PortBinding) String() string {
	ret := fmt.Sprintf("%s/", p.Proto)
	if p.IP != nil {
		ret += p.IP.String()
	}
	ret = fmt.Sprintf("%s:%d/", ret, p.Port)
	if p.HostIP != nil {
		ret += p.HostIP.String()
	}
	ret = fmt.Sprintf("%s:%d", ret, p.HostPort)
	return ret
}

// FromString reads the TransportPort structure from string
func (p *PortBinding) FromString(s string) error {
	ps := strings.Split(s, "/")
	if len(ps) != 3 {
		return BadRequestErrorf("invalid format for port binding: %s", s)
	}

	p.Proto = ParseProtocol(ps[0])

	var err error
	if p.IP, p.Port, err = parseIPPort(ps[1]); err != nil {
		return BadRequestErrorf("failed to parse Container IP/Port in port binding: %s", err.Error())
	}

	if p.HostIP, p.HostPort, err = parseIPPort(ps[2]); err != nil {
		return BadRequestErrorf("failed to parse Host IP/Port in port binding: %s", err.Error())
	}

	return nil
}

func parseIPPort(s string) (net.IP, uint16, error) {
	pp := strings.Split(s, ":")
	if len(pp) != 2 {
		return nil, 0, BadRequestErrorf("invalid format: %s", s)
	}

	var ip net.IP
	if pp[0] != "" {
		if ip = net.ParseIP(pp[0]); ip == nil {
			return nil, 0, BadRequestErrorf("invalid ip: %s", pp[0])
		}
	}

	port, err := strconv.ParseUint(pp[1], 10, 16)
	if err != nil {
		return nil, 0, BadRequestErrorf("invalid port: %s", pp[1])
	}

	return ip, uint16(port), nil
}

// Equal checks if this instance of PortBinding is equal to the passed one
func (p *PortBinding) Equal(o *PortBinding) bool {
	if p == o {
		return true
	}

	if o == nil {
		return false
	}

	if p.Proto != o.Proto || p.Port != o.Port ||
		p.HostPort != o.HostPort || p.HostPortEnd != o.HostPortEnd {
		return false
	}

	if p.IP != nil {
		if !p.IP.Equal(o.IP) {
			return false
		}
	} else {
		if o.IP != nil {
			return false
		}
	}

	if p.HostIP != nil {
		if !p.HostIP.Equal(o.HostIP) {
			return false
		}
	} else {
		if o.HostIP != nil {
			return false
		}
	}

	return true
}

// ErrInvalidProtocolBinding is returned when the port binding protocol is not valid.
type ErrInvalidProtocolBinding string

func (ipb ErrInvalidProtocolBinding) Error() string {
	return fmt.Sprintf("invalid transport protocol: %s", string(ipb))
}

const (
	// ICMP is for the ICMP ip protocol
	ICMP = 1
	// TCP is for the TCP ip protocol
	TCP = 6
	// UDP is for the UDP ip protocol
	UDP = 17
)

// Protocol represents an IP protocol number
type Protocol uint8

func (p Protocol) String() string {
	switch p {
	case ICMP:
		return "icmp"
	case TCP:
		return "tcp"
	case UDP:
		return "udp"
	default:
		return fmt.Sprintf("%d", p)
	}
}

// ParseProtocol returns the respective Protocol type for the passed string
func ParseProtocol(s string) Protocol {
	switch strings.ToLower(s) {
	case "icmp":
		return ICMP
	case "udp":
		return UDP
	case "tcp":
		return TCP
	default:
		return 0
	}
}

// GetMacCopy returns a copy of the passed MAC address
func GetMacCopy(from net.HardwareAddr) net.HardwareAddr {
	if from == nil {
		return nil
	}
	to := make(net.HardwareAddr, len(from))
	copy(to, from)
	return to
}

// GetIPCopy returns a copy of the passed IP address
func GetIPCopy(from net.IP) net.IP {
	if from == nil {
		return nil
	}
	to := make(net.IP, len(from))
	copy(to, from)
	return to
}

// GetIPNetCopy returns a copy of the passed IP Network
func GetIPNetCopy(from *net.IPNet) *net.IPNet {
	if from == nil {
		return nil
	}
	bm := make(net.IPMask, len(from.Mask))
	copy(bm, from.Mask)
	return &net.IPNet{IP: GetIPCopy(from.IP), Mask: bm}
}

// GetIPNetCanonical returns the canonical form for the passed network
func GetIPNetCanonical(nw *net.IPNet) *net.IPNet {
	if nw == nil {
		return nil
	}
	c := GetIPNetCopy(nw)
	c.IP = c.IP.Mask(nw.Mask)
	return c
}

// CompareIPNet returns equal if the two IP Networks are equal
func CompareIPNet(a, b *net.IPNet) bool {
	if a == b {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return a.IP.Equal(b.IP) && bytes.Equal(a.Mask, b.Mask)
}

// GetMinimalIP returns the address in its shortest form
func GetMinimalIP(ip net.IP) net.IP {
	if ip != nil && ip.To4() != nil {
		return ip.To4()
	}
	return ip
}

// GetMinimalIPNet returns a copy of the passed IP Network with congruent ip and mask notation
func GetMinimalIPNet(nw *net.IPNet) *net.IPNet {
	if nw == nil {
		return nil
	}
	if len(nw.IP) == 16 && nw.IP.To4() != nil {
		m := nw.Mask
		if len(m) == 16 {
			m = m[12:16]
		}
		return &net.IPNet{IP: nw.IP.To4(), Mask: m}
	}
	return nw
}

// IsIPNetValid returns true if the ipnet is a valid network/mask
// combination. Otherwise returns false.
func IsIPNetValid(nw *net.IPNet) bool {
	return nw.String() != "0.0.0.0/0"
}

var v4inV6MaskPrefix = []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}

// compareIPMask checks if the passed ip and mask are semantically compatible.
// It returns the byte indexes for the address and mask so that caller can
// do bitwise operations without modifying address representation.
func compareIPMask(ip net.IP, mask net.IPMask) (is int, ms int, err error) {
	// Find the effective starting of address and mask
	if len(ip) == net.IPv6len && ip.To4() != nil {
		is = 12
	}
	if len(ip[is:]) == net.IPv4len && len(mask) == net.IPv6len && bytes.Equal(mask[:12], v4inV6MaskPrefix) {
		ms = 12
	}
	// Check if address and mask are semantically compatible
	if len(ip[is:]) != len(mask[ms:]) {
		err = fmt.Errorf("ip and mask are not compatible: (%#v, %#v)", ip, mask)
	}
	return
}

// GetHostPartIP returns the host portion of the ip address identified by the mask.
// IP address representation is not modified. If address and mask are not compatible
// an error is returned.
func GetHostPartIP(ip net.IP, mask net.IPMask) (net.IP, error) {
	// Find the effective starting of address and mask
	is, ms, err := compareIPMask(ip, mask)
	if err != nil {
		return nil, fmt.Errorf("cannot compute host portion ip address because %s", err)
	}

	// Compute host portion
	out := GetIPCopy(ip)
	for i := 0; i < len(mask[ms:]); i++ {
		out[is+i] &= ^mask[ms+i]
	}

	return out, nil
}

// GetBroadcastIP returns the broadcast ip address for the passed network (ip and mask).
// IP address representation is not modified. If address and mask are not compatible
// an error is returned.
func GetBroadcastIP(ip net.IP, mask net.IPMask) (net.IP, error) {
	// Find the effective starting of address and mask
	is, ms, err := compareIPMask(ip, mask)
	if err != nil {
		return nil, fmt.Errorf("cannot compute broadcast ip address because %s", err)
	}

	// Compute broadcast address
	out := GetIPCopy(ip)
	for i := 0; i < len(mask[ms:]); i++ {
		out[is+i] |= ^mask[ms+i]
	}

	return out, nil
}

// ParseCIDR returns the *net.IPNet represented by the passed CIDR notation
func ParseCIDR(cidr string) (n *net.IPNet, e error) {
	var i net.IP
	if i, n, e = net.ParseCIDR(cidr); e == nil {
		n.IP = i
	}
	return
}

const (
	// NEXTHOP indicates a StaticRoute with an IP next hop.
	NEXTHOP = iota

	// CONNECTED indicates a StaticRoute with an interface for directly connected peers.
	CONNECTED
)

// StaticRoute is a statically-provisioned IP route.
type StaticRoute struct {
	Destination *net.IPNet

	RouteType int // NEXT_HOP or CONNECTED

	// NextHop will be resolved by the kernel (i.e. as a loose hop).
	NextHop net.IP
}

// GetCopy returns a copy of this StaticRoute structure
func (r *StaticRoute) GetCopy() *StaticRoute {
	d := GetIPNetCopy(r.Destination)
	nh := GetIPCopy(r.NextHop)
	return &StaticRoute{Destination: d,
		RouteType: r.RouteType,
		NextHop:   nh,
	}
}

// InterfaceStatistics represents the interface's statistics
type InterfaceStatistics struct {
	RxBytes   uint64
	RxPackets uint64
	RxErrors  uint64
	RxDropped uint64
	TxBytes   uint64
	TxPackets uint64
	TxErrors  uint64
	TxDropped uint64
}

func (is *InterfaceStatistics) String() string {
	return fmt.Sprintf("\nRxBytes: %d, RxPackets: %d, RxErrors: %d, RxDropped: %d, TxBytes: %d, TxPackets: %d, TxErrors: %d, TxDropped: %d",
		is.RxBytes, is.RxPackets, is.RxErrors, is.RxDropped, is.TxBytes, is.TxPackets, is.TxErrors, is.TxDropped)
}

/******************************
 * Well-known Error Interfaces
 ******************************/

// MaskableError is an interface for errors which can be ignored by caller
type MaskableError interface {
	// Maskable makes implementer into MaskableError type
	Maskable()
}

// RetryError is an interface for errors which might get resolved through retry
type RetryError interface {
	// Retry makes implementer into RetryError type
	Retry()
}

// BadRequestError is an interface for errors originated by a bad request
type BadRequestError interface {
	// BadRequest makes implementer into BadRequestError type
	BadRequest()
}

// NotFoundError is an interface for errors raised because a needed resource is not available
type NotFoundError interface {
	// NotFound makes implementer into NotFoundError type
	NotFound()
}

// ForbiddenError is an interface for errors which denote a valid request that cannot be honored
type ForbiddenError interface {
	// Forbidden makes implementer into ForbiddenError type
	Forbidden()
}

// NoServiceError is an interface for errors returned when the required service is not available
type NoServiceError interface {
	// NoService makes implementer into NoServiceError type
	NoService()
}

// TimeoutError is an interface for errors raised because of timeout
type TimeoutError interface {
	// Timeout makes implementer into TimeoutError type
	Timeout()
}

// NotImplementedError is an interface for errors raised because of requested functionality is not yet implemented
type NotImplementedError interface {
	// NotImplemented makes implementer into NotImplementedError type
	NotImplemented()
}

// InternalError is an interface for errors raised because of an internal error
type InternalError interface {
	// Internal makes implementer into InternalError type
	Internal()
}

/******************************
 * Well-known Error Formatters
 ******************************/

// BadRequestErrorf creates an instance of BadRequestError
func BadRequestErrorf(format string, params ...interface{}) error {
	return badRequest(fmt.Sprintf(format, params...))
}

// NotFoundErrorf creates an instance of NotFoundError
func NotFoundErrorf(format string, params ...interface{}) error {
	return notFound(fmt.Sprintf(format, params...))
}

// ForbiddenErrorf creates an instance of ForbiddenError
func ForbiddenErrorf(format string, params ...interface{}) error {
	return forbidden(fmt.Sprintf(format, params...))
}

// NoServiceErrorf creates an instance of NoServiceError
func NoServiceErrorf(format string, params ...interface{}) error {
	return noService(fmt.Sprintf(format, params...))
}

// NotImplementedErrorf creates an instance of NotImplementedError
func NotImplementedErrorf(format string, params ...interface{}) error {
	return notImpl(fmt.Sprintf(format, params...))
}

// TimeoutErrorf creates an instance of TimeoutError
func TimeoutErrorf(format string, params ...interface{}) error {
	return timeout(fmt.Sprintf(format, params...))
}

// InternalErrorf creates an instance of InternalError
func InternalErrorf(format string, params ...interface{}) error {
	return internal(fmt.Sprintf(format, params...))
}

// InternalMaskableErrorf creates an instance of InternalError and MaskableError
func InternalMaskableErrorf(format string, params ...interface{}) error {
	return maskInternal(fmt.Sprintf(format, params...))
}

// RetryErrorf creates an instance of RetryError
func RetryErrorf(format string, params ...interface{}) error {
	return retry(fmt.Sprintf(format, params...))
}

/***********************
 * Internal Error Types
 ***********************/
type badRequest string

func (br badRequest) Error() string {
	return string(br)
}
func (br badRequest) BadRequest() {}

type maskBadRequest string

type notFound string

func (nf notFound) Error() string {
	return string(nf)
}
func (nf notFound) NotFound() {}

type forbidden string

func (frb forbidden) Error() string {
	return string(frb)
}
func (frb forbidden) Forbidden() {}

type noService string

func (ns noService) Error() string {
	return string(ns)
}
func (ns noService) NoService() {}

type maskNoService string

type timeout string

func (to timeout) Error() string {
	return string(to)
}
func (to timeout) Timeout() {}

type notImpl string

func (ni notImpl) Error() string {
	return string(ni)
}
func (ni notImpl) NotImplemented() {}

type internal string

func (nt internal) Error() string {
	return string(nt)
}
func (nt internal) Internal() {}

type maskInternal string

func (mnt maskInternal) Error() string {
	return string(mnt)
}
func (mnt maskInternal) Internal() {}
func (mnt maskInternal) Maskable() {}

type retry string

func (r retry) Error() string {
	return string(r)
}
func (r retry) Retry() {}
