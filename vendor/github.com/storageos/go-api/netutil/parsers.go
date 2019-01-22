package netutil

import (
	"net"
	"net/url"
	"strconv"
	"strings"
)

// addrsFromNodes takes a list of node hosts and attempts to return a list of hosts in ip:port
// format along with any error encountered.
//
// The function accepts node hosts in URL, ip, ip:port, resolvable-name and resolvable-name:port
// formats and will append the default port value if needed.
func addrsFromNodes(nodes []string) ([]string, error) {
	var addrs []string

	for _, n := range nodes {
		switch {
		// Assume that the node is provided as a URL
		case strings.Contains(n, "://"):
			newAddrs, err := parseURL(n)
			if err != nil {
				return nil, newInvalidNodeError(err)
			}

			addrs = append(addrs, newAddrs...)

		// Assume the node is in hostname:port or ip:port format
		case strings.Contains(n, ":"):
			newAddrs, err := parseHostPort(n)
			if err != nil {
				return nil, newInvalidNodeError(err)
			}

			addrs = append(addrs, newAddrs...)

		// Assume hostname or ip
		default:
			newAddrs, err := parseHost(n)
			if err != nil {
				return nil, newInvalidNodeError(err)
			}

			addrs = append(addrs, newAddrs...)
		}
	}

	return addrs, nil
}

func validPort(port string) bool {
	intPort, err := strconv.Atoi(port)

	return (err == nil) &&
		(intPort > 0) &&
		(intPort <= 65535)
}

// parseURL takes a valid URL and verifies that it is using a correct scheme, has a resolvable
// address (or is an IP) and has a valid port (or adds the default if the port is omitted). The
// function then returns a list of addresses in ip:port format along with any error encountered.
//
// The function may return multiple addresses depending on the dns answer received when resolving
// the host.
func parseURL(node string) ([]string, error) {
	url, err := url.Parse(node)
	if err != nil {
		return nil, err
	}

	// Verify a valid scheme
	switch url.Scheme {
	case "tcp", "http", "https":
		host, port, err := net.SplitHostPort(url.Host)
		if err != nil {
			// We could be here as there is no port, lets try one last time with default port added
			host, port, err = net.SplitHostPort(url.Host + ":" + DefaultDialPort)
			if err != nil {
				return nil, err
			}
		}

		if !validPort(port) {
			return nil, errInvalidPortNumber
		}

		// LookupHost works for IP addr too
		addrs, err := net.LookupHost(host)
		if err != nil {
			return nil, err
		}

		for i, a := range addrs {
			addrs[i] = a + ":" + port
		}

		return addrs, nil

	default:
		return nil, errUnsupportedScheme
	}
}

// parseHostPort takes a string in host:port format and checks it has a resolvable address (or is
// an IP) and a valid port (or adds the default if the port is omitted). The function then returns
// a list of addresses in ip:port format along with any error encountered.
//
// The function may return multiple addresses depending on the dns answer received when resolving
// the host.
func parseHostPort(node string) ([]string, error) {
	host, port, err := net.SplitHostPort(node)
	if err != nil {
		return nil, err
	}

	if !validPort(port) {
		return nil, errInvalidPortNumber
	}

	// LookupHost works for IP addr too
	addrs, err := net.LookupHost(host)
	if err != nil {
		return nil, err
	}

	for i, a := range addrs {
		addrs[i] = a + ":" + port
	}

	return addrs, nil
}

// parseHostPort takes a hostname string and checks it is resolvable to an address (or is already
// an IP) The function then returns a list of addresses in ip:port format (where port is the
// default port) along with any error encountered.
//
// The function may return multiple addresses depending on the dns answer received when resolving
// the host.
func parseHost(node string) ([]string, error) {
	return parseHostPort(node + ":" + DefaultDialPort)
}
