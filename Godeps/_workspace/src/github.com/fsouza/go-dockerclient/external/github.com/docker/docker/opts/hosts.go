package opts

import (
	"fmt"
	"net"
	"net/url"
	"runtime"
	"strconv"
	"strings"
)

var (
	// DefaultHTTPPort Default HTTP Port used if only the protocol is provided to -H flag e.g. docker daemon -H tcp://
	// TODO Windows. DefaultHTTPPort is only used on Windows if a -H parameter
	// is not supplied. A better longer term solution would be to use a named
	// pipe as the default on the Windows daemon.
	// These are the IANA registered port numbers for use with Docker
	// see http://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml?search=docker
	DefaultHTTPPort = 2375 // Default HTTP Port
	// DefaultTLSHTTPPort Default HTTP Port used when TLS enabled
	DefaultTLSHTTPPort = 2376 // Default TLS encrypted HTTP Port
	// DefaultUnixSocket Path for the unix socket.
	// Docker daemon by default always listens on the default unix socket
	DefaultUnixSocket = "/var/run/docker.sock"
	// DefaultTCPHost constant defines the default host string used by docker on Windows
	DefaultTCPHost = fmt.Sprintf("tcp://%s:%d", DefaultHTTPHost, DefaultHTTPPort)
	// DefaultTLSHost constant defines the default host string used by docker for TLS sockets
	DefaultTLSHost = fmt.Sprintf("tcp://%s:%d", DefaultHTTPHost, DefaultTLSHTTPPort)
)

// ValidateHost validates that the specified string is a valid host and returns it.
func ValidateHost(val string) (string, error) {
	_, err := parseDockerDaemonHost(DefaultTCPHost, DefaultTLSHost, DefaultUnixSocket, "", val)
	if err != nil {
		return val, err
	}
	// Note: unlike most flag validators, we don't return the mutated value here
	//       we need to know what the user entered later (using ParseHost) to adjust for tls
	return val, nil
}

// ParseHost and set defaults for a Daemon host string
func ParseHost(defaultHost, val string) (string, error) {
	host, err := parseDockerDaemonHost(DefaultTCPHost, DefaultTLSHost, DefaultUnixSocket, defaultHost, val)
	if err != nil {
		return val, err
	}
	return host, nil
}

// parseDockerDaemonHost parses the specified address and returns an address that will be used as the host.
// Depending of the address specified, will use the defaultTCPAddr or defaultUnixAddr
// defaultUnixAddr must be a absolute file path (no `unix://` prefix)
// defaultTCPAddr must be the full `tcp://host:port` form
func parseDockerDaemonHost(defaultTCPAddr, defaultTLSHost, defaultUnixAddr, defaultAddr, addr string) (string, error) {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		if defaultAddr == defaultTLSHost {
			return defaultTLSHost, nil
		}
		if runtime.GOOS != "windows" {
			return fmt.Sprintf("unix://%s", defaultUnixAddr), nil
		}
		return defaultTCPAddr, nil
	}
	addrParts := strings.Split(addr, "://")
	if len(addrParts) == 1 {
		addrParts = []string{"tcp", addrParts[0]}
	}

	switch addrParts[0] {
	case "tcp":
		return parseTCPAddr(addrParts[1], defaultTCPAddr)
	case "unix":
		return parseUnixAddr(addrParts[1], defaultUnixAddr)
	case "fd":
		return addr, nil
	default:
		return "", fmt.Errorf("Invalid bind address format: %s", addr)
	}
}

// parseUnixAddr parses and validates that the specified address is a valid UNIX
// socket address. It returns a formatted UNIX socket address, either using the
// address parsed from addr, or the contents of defaultAddr if addr is a blank
// string.
func parseUnixAddr(addr string, defaultAddr string) (string, error) {
	addr = strings.TrimPrefix(addr, "unix://")
	if strings.Contains(addr, "://") {
		return "", fmt.Errorf("Invalid proto, expected unix: %s", addr)
	}
	if addr == "" {
		addr = defaultAddr
	}
	return fmt.Sprintf("unix://%s", addr), nil
}

// parseTCPAddr parses and validates that the specified address is a valid TCP
// address. It returns a formatted TCP address, either using the address parsed
// from tryAddr, or the contents of defaultAddr if tryAddr is a blank string.
// tryAddr is expected to have already been Trim()'d
// defaultAddr must be in the full `tcp://host:port` form
func parseTCPAddr(tryAddr string, defaultAddr string) (string, error) {
	if tryAddr == "" || tryAddr == "tcp://" {
		return defaultAddr, nil
	}
	addr := strings.TrimPrefix(tryAddr, "tcp://")
	if strings.Contains(addr, "://") || addr == "" {
		return "", fmt.Errorf("Invalid proto, expected tcp: %s", tryAddr)
	}

	defaultAddr = strings.TrimPrefix(defaultAddr, "tcp://")
	defaultHost, defaultPort, err := net.SplitHostPort(defaultAddr)
	if err != nil {
		return "", err
	}
	// url.Parse fails for trailing colon on IPv6 brackets on Go 1.5, but
	// not 1.4. See https://github.com/golang/go/issues/12200 and
	// https://github.com/golang/go/issues/6530.
	if strings.HasSuffix(addr, "]:") {
		addr += defaultPort
	}

	u, err := url.Parse("tcp://" + addr)
	if err != nil {
		return "", err
	}

	host, port, err := net.SplitHostPort(u.Host)
	if err != nil {
		return "", fmt.Errorf("Invalid bind address format: %s", tryAddr)
	}

	if host == "" {
		host = defaultHost
	}
	if port == "" {
		port = defaultPort
	}
	p, err := strconv.Atoi(port)
	if err != nil && p == 0 {
		return "", fmt.Errorf("Invalid bind address format: %s", tryAddr)
	}

	return fmt.Sprintf("tcp://%s%s", net.JoinHostPort(host, port), u.Path), nil
}
