package ipvs

import (
	"fmt"
	"net"
	"time"

	"github.com/vishvananda/netlink/nl"
	"github.com/vishvananda/netns"
	"golang.org/x/sys/unix"
)

const (
	netlinkRecvSocketsTimeout = 3 * time.Second
	netlinkSendSocketTimeout  = 30 * time.Second
)

// Service defines an IPVS service in its entirety.
type Service struct {
	// Virtual service address.
	Address  net.IP
	Protocol uint16
	Port     uint16
	FWMark   uint32 // Firewall mark of the service.

	// Virtual service options.
	SchedName     string
	Flags         uint32
	Timeout       uint32
	Netmask       uint32
	AddressFamily uint16
	PEName        string
	Stats         SvcStats
}

// SvcStats defines an IPVS service statistics
type SvcStats struct {
	Connections uint32
	PacketsIn   uint32
	PacketsOut  uint32
	BytesIn     uint64
	BytesOut    uint64
	CPS         uint32
	BPSOut      uint32
	PPSIn       uint32
	PPSOut      uint32
	BPSIn       uint32
}

// Destination defines an IPVS destination (real server) in its
// entirety.
type Destination struct {
	Address             net.IP
	Port                uint16
	Weight              int
	ConnectionFlags     uint32
	AddressFamily       uint16
	UpperThreshold      uint32
	LowerThreshold      uint32
	ActiveConnections   int
	InactiveConnections int
	Stats               DstStats
}

// DstStats defines IPVS destination (real server) statistics
type DstStats SvcStats

// Config defines IPVS timeout configuration
type Config struct {
	TimeoutTCP    time.Duration
	TimeoutTCPFin time.Duration
	TimeoutUDP    time.Duration
}

// Handle provides a namespace specific ipvs handle to program ipvs
// rules.
type Handle struct {
	seq  uint32
	sock *nl.NetlinkSocket
}

// New provides a new ipvs handle in the namespace pointed to by the
// passed path. It will return a valid handle or an error in case an
// error occurred while creating the handle.
func New(path string) (*Handle, error) {
	setup()

	n := netns.None()
	if path != "" {
		var err error
		n, err = netns.GetFromPath(path)
		if err != nil {
			return nil, err
		}
	}
	defer n.Close()

	sock, err := nl.GetNetlinkSocketAt(n, netns.None(), unix.NETLINK_GENERIC)
	if err != nil {
		return nil, err
	}
	// Add operation timeout to avoid deadlocks
	tv := unix.NsecToTimeval(netlinkSendSocketTimeout.Nanoseconds())
	if err := sock.SetSendTimeout(&tv); err != nil {
		return nil, err
	}
	tv = unix.NsecToTimeval(netlinkRecvSocketsTimeout.Nanoseconds())
	if err := sock.SetReceiveTimeout(&tv); err != nil {
		return nil, err
	}

	return &Handle{sock: sock}, nil
}

// Close closes the ipvs handle. The handle is invalid after Close
// returns.
func (i *Handle) Close() {
	if i.sock != nil {
		i.sock.Close()
	}
}

// NewService creates a new ipvs service in the passed handle.
func (i *Handle) NewService(s *Service) error {
	return i.doCmd(s, nil, ipvsCmdNewService)
}

// IsServicePresent queries for the ipvs service in the passed handle.
func (i *Handle) IsServicePresent(s *Service) bool {
	return nil == i.doCmd(s, nil, ipvsCmdGetService)
}

// UpdateService updates an already existing service in the passed
// handle.
func (i *Handle) UpdateService(s *Service) error {
	return i.doCmd(s, nil, ipvsCmdSetService)
}

// DelService deletes an already existing service in the passed
// handle.
func (i *Handle) DelService(s *Service) error {
	return i.doCmd(s, nil, ipvsCmdDelService)
}

// Flush deletes all existing services in the passed
// handle.
func (i *Handle) Flush() error {
	_, err := i.doCmdWithoutAttr(ipvsCmdFlush)
	return err
}

// NewDestination creates a new real server in the passed ipvs
// service which should already be existing in the passed handle.
func (i *Handle) NewDestination(s *Service, d *Destination) error {
	return i.doCmd(s, d, ipvsCmdNewDest)
}

// UpdateDestination updates an already existing real server in the
// passed ipvs service in the passed handle.
func (i *Handle) UpdateDestination(s *Service, d *Destination) error {
	return i.doCmd(s, d, ipvsCmdSetDest)
}

// DelDestination deletes an already existing real server in the
// passed ipvs service in the passed handle.
func (i *Handle) DelDestination(s *Service, d *Destination) error {
	return i.doCmd(s, d, ipvsCmdDelDest)
}

// GetServices returns an array of services configured on the Node
func (i *Handle) GetServices() ([]*Service, error) {
	return i.doGetServicesCmd(nil)
}

// GetDestinations returns an array of Destinations configured for this Service
func (i *Handle) GetDestinations(s *Service) ([]*Destination, error) {
	return i.doGetDestinationsCmd(s, nil)
}

// GetService gets details of a specific IPVS services, useful in updating statisics etc.,
func (i *Handle) GetService(s *Service) (*Service, error) {
	res, err := i.doGetServicesCmd(s)
	if err != nil {
		return nil, err
	}

	// We are looking for exactly one service otherwise error out
	if len(res) != 1 {
		return nil, fmt.Errorf("Expected only one service obtained=%d", len(res))
	}

	return res[0], nil
}

// GetConfig returns the current timeout configuration
func (i *Handle) GetConfig() (*Config, error) {
	return i.doGetConfigCmd()
}

// SetConfig set the current timeout configuration. 0: no change
func (i *Handle) SetConfig(c *Config) error {
	return i.doSetConfigCmd(c)
}
