package portallocator

import (
	"errors"
	"fmt"
	"net"
	"sync"
)

const (
	// DefaultPortRangeStart indicates the first port in port range
	DefaultPortRangeStart = 49153
	// DefaultPortRangeEnd indicates the last port in port range
	DefaultPortRangeEnd = 65535
)

type ipMapping map[string]protoMap

var (
	// ErrAllPortsAllocated is returned when no more ports are available
	ErrAllPortsAllocated = errors.New("all ports are allocated")
	// ErrUnknownProtocol is returned when an unknown protocol was specified
	ErrUnknownProtocol = errors.New("unknown protocol")
	defaultIP          = net.ParseIP("0.0.0.0")
	once               sync.Once
	instance           *PortAllocator
	createInstance     = func() { instance = newInstance() }
)

// ErrPortAlreadyAllocated is the returned error information when a requested port is already being used
type ErrPortAlreadyAllocated struct {
	ip   string
	port int
}

func newErrPortAlreadyAllocated(ip string, port int) ErrPortAlreadyAllocated {
	return ErrPortAlreadyAllocated{
		ip:   ip,
		port: port,
	}
}

// IP returns the address to which the used port is associated
func (e ErrPortAlreadyAllocated) IP() string {
	return e.ip
}

// Port returns the value of the already used port
func (e ErrPortAlreadyAllocated) Port() int {
	return e.port
}

// IPPort returns the address and the port in the form ip:port
func (e ErrPortAlreadyAllocated) IPPort() string {
	return fmt.Sprintf("%s:%d", e.ip, e.port)
}

// Error is the implementation of error.Error interface
func (e ErrPortAlreadyAllocated) Error() string {
	return fmt.Sprintf("Bind for %s:%d failed: port is already allocated", e.ip, e.port)
}

type (
	// PortAllocator manages the transport ports database
	PortAllocator struct {
		mutex sync.Mutex
		ipMap ipMapping
		Begin int
		End   int
	}
	portRange struct {
		begin int
		end   int
		last  int
	}
	portMap struct {
		p            map[int]struct{}
		defaultRange string
		portRanges   map[string]*portRange
	}
	protoMap map[string]*portMap
)

// Get returns the default instance of PortAllocator
func Get() *PortAllocator {
	// Port Allocator is a singleton
	// Note: Long term solution will be each PortAllocator will have access to
	// the OS so that it can have up to date view of the OS port allocation.
	// When this happens singleton behavior will be removed. Clients do not
	// need to worry about this, they will not see a change in behavior.
	once.Do(createInstance)
	return instance
}

func newInstance() *PortAllocator {
	start, end, err := getDynamicPortRange()
	if err != nil {
		start, end = DefaultPortRangeStart, DefaultPortRangeEnd
	}
	return &PortAllocator{
		ipMap: ipMapping{},
		Begin: start,
		End:   end,
	}
}

// RequestPort requests new port from global ports pool for specified ip and proto.
// If port is 0 it returns first free port. Otherwise it checks port availability
// in proto's pool and returns that port or error if port is already busy.
func (p *PortAllocator) RequestPort(ip net.IP, proto string, port int) (int, error) {
	return p.RequestPortInRange(ip, proto, port, port)
}

// RequestPortInRange requests new port from global ports pool for specified ip and proto.
// If portStart and portEnd are 0 it returns the first free port in the default ephemeral range.
// If portStart != portEnd it returns the first free port in the requested range.
// Otherwise (portStart == portEnd) it checks port availability in the requested proto's port-pool
// and returns that port or error if port is already busy.
func (p *PortAllocator) RequestPortInRange(ip net.IP, proto string, portStart, portEnd int) (int, error) {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	if proto != "tcp" && proto != "udp" {
		return 0, ErrUnknownProtocol
	}

	if ip == nil {
		ip = defaultIP
	}
	ipstr := ip.String()
	protomap, ok := p.ipMap[ipstr]
	if !ok {
		protomap = protoMap{
			"tcp": p.newPortMap(),
			"udp": p.newPortMap(),
		}

		p.ipMap[ipstr] = protomap
	}
	mapping := protomap[proto]
	if portStart > 0 && portStart == portEnd {
		if _, ok := mapping.p[portStart]; !ok {
			mapping.p[portStart] = struct{}{}
			return portStart, nil
		}
		return 0, newErrPortAlreadyAllocated(ipstr, portStart)
	}

	port, err := mapping.findPort(portStart, portEnd)
	if err != nil {
		return 0, err
	}
	return port, nil
}

// ReleasePort releases port from global ports pool for specified ip and proto.
func (p *PortAllocator) ReleasePort(ip net.IP, proto string, port int) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	if ip == nil {
		ip = defaultIP
	}
	protomap, ok := p.ipMap[ip.String()]
	if !ok {
		return nil
	}
	delete(protomap[proto].p, port)
	return nil
}

func (p *PortAllocator) newPortMap() *portMap {
	defaultKey := getRangeKey(p.Begin, p.End)
	pm := &portMap{
		p:            map[int]struct{}{},
		defaultRange: defaultKey,
		portRanges: map[string]*portRange{
			defaultKey: newPortRange(p.Begin, p.End),
		},
	}
	return pm
}

// ReleaseAll releases all ports for all ips.
func (p *PortAllocator) ReleaseAll() error {
	p.mutex.Lock()
	p.ipMap = ipMapping{}
	p.mutex.Unlock()
	return nil
}

func getRangeKey(portStart, portEnd int) string {
	return fmt.Sprintf("%d-%d", portStart, portEnd)
}

func newPortRange(portStart, portEnd int) *portRange {
	return &portRange{
		begin: portStart,
		end:   portEnd,
		last:  portEnd,
	}
}

func (pm *portMap) getPortRange(portStart, portEnd int) (*portRange, error) {
	var key string
	if portStart == 0 && portEnd == 0 {
		key = pm.defaultRange
	} else {
		key = getRangeKey(portStart, portEnd)
		if portStart == portEnd ||
			portStart == 0 || portEnd == 0 ||
			portEnd < portStart {
			return nil, fmt.Errorf("invalid port range: %s", key)
		}
	}

	// Return existing port range, if already known.
	if pr, exists := pm.portRanges[key]; exists {
		return pr, nil
	}

	// Otherwise create a new port range.
	pr := newPortRange(portStart, portEnd)
	pm.portRanges[key] = pr
	return pr, nil
}

func (pm *portMap) findPort(portStart, portEnd int) (int, error) {
	pr, err := pm.getPortRange(portStart, portEnd)
	if err != nil {
		return 0, err
	}
	port := pr.last

	for i := 0; i <= pr.end-pr.begin; i++ {
		port++
		if port > pr.end {
			port = pr.begin
		}

		if _, ok := pm.p[port]; !ok {
			pm.p[port] = struct{}{}
			pr.last = port
			return port, nil
		}
	}
	return 0, ErrAllPortsAllocated
}
