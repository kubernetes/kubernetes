package discovery

import (
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

var (
	// Backends is a global map of discovery backends indexed by their
	// associated scheme.
	backends = make(map[string]Backend)
)

// Register makes a discovery backend available by the provided scheme.
// If Register is called twice with the same scheme an error is returned.
func Register(scheme string, d Backend) error {
	if _, exists := backends[scheme]; exists {
		return fmt.Errorf("scheme already registered %s", scheme)
	}
	logrus.WithField("name", scheme).Debugf("Registering discovery service")
	backends[scheme] = d
	return nil
}

func parse(rawurl string) (string, string) {
	parts := strings.SplitN(rawurl, "://", 2)

	// nodes:port,node2:port => nodes://node1:port,node2:port
	if len(parts) == 1 {
		return "nodes", parts[0]
	}
	return parts[0], parts[1]
}

// ParseAdvertise parses the --cluster-advertise daemon config which accepts
// <ip-address>:<port> or <interface-name>:<port>
func ParseAdvertise(advertise string) (string, error) {
	var (
		iface *net.Interface
		addrs []net.Addr
		err   error
	)

	addr, port, err := net.SplitHostPort(advertise)

	if err != nil {
		return "", fmt.Errorf("invalid --cluster-advertise configuration: %s: %v", advertise, err)
	}

	ip := net.ParseIP(addr)
	// If it is a valid ip-address, use it as is
	if ip != nil {
		return advertise, nil
	}

	// If advertise is a valid interface name, get the valid IPv4 address and use it to advertise
	ifaceName := addr
	iface, err = net.InterfaceByName(ifaceName)
	if err != nil {
		return "", fmt.Errorf("invalid cluster advertise IP address or interface name (%s) : %v", advertise, err)
	}

	addrs, err = iface.Addrs()
	if err != nil {
		return "", fmt.Errorf("unable to get advertise IP address from interface (%s) : %v", advertise, err)
	}

	if len(addrs) == 0 {
		return "", fmt.Errorf("no available advertise IP address in interface (%s)", advertise)
	}

	addr = ""
	for _, a := range addrs {
		ip, _, err := net.ParseCIDR(a.String())
		if err != nil {
			return "", fmt.Errorf("error deriving advertise ip-address in interface (%s) : %v", advertise, err)
		}
		if ip.To4() == nil || ip.IsLoopback() {
			continue
		}
		addr = ip.String()
		break
	}
	if addr == "" {
		return "", fmt.Errorf("could not find a valid ip-address in interface %s", advertise)
	}

	addr = net.JoinHostPort(addr, port)
	return addr, nil
}

// New returns a new Discovery given a URL, heartbeat and ttl settings.
// Returns an error if the URL scheme is not supported.
func New(rawurl string, heartbeat time.Duration, ttl time.Duration, clusterOpts map[string]string) (Backend, error) {
	scheme, uri := parse(rawurl)
	if backend, exists := backends[scheme]; exists {
		logrus.WithFields(logrus.Fields{"name": scheme, "uri": uri}).Debugf("Initializing discovery service")
		err := backend.Initialize(uri, heartbeat, ttl, clusterOpts)
		return backend, err
	}

	return nil, ErrNotSupported
}
