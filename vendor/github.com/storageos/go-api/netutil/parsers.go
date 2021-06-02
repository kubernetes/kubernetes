package netutil

import (
	"net/url"
	"strconv"
	"strings"
)

const (
	httpScheme  = "http"
	httpsScheme = "https"
	tcpScheme   = "tcp"
)

// AddressesFromNodes takes a list of node hosts and attempts to return a list of hosts in host:port
// format along with any error encountered.
//
// The function accepts node hosts in URL, ip, ip:port, resolvable-name and resolvable-name:port
// formats and will append the default port value if needed. For hosts where the scheme has been omitted,
// the scheme for the first host will be used. If the first host has no scheme, it will default to http.
func AddressesFromNodes(nodes []string) ([]string, error) {
	var addresses []string

	var scheme string

	for _, node := range nodes {
		address := node
		// If no scheme present, set the first scheme
		if !strings.Contains(address, "://") {
			if scheme == "" {
				scheme = httpScheme
			}
			address = strings.Join([]string{scheme, address}, "://")
		}

		url, err := url.Parse(address)
		if err != nil {
			return nil, newInvalidNodeError(err)
		}

		switch url.Scheme {
		case tcpScheme:
			url.Scheme = httpScheme
			fallthrough
		case httpScheme, httpsScheme:
			if scheme == "" {
				scheme = url.Scheme
			}
		default:
			return nil, newInvalidNodeError(errUnsupportedScheme)
		}

		host := url.Hostname()
		if host == "" {
			return nil, newInvalidNodeError(errInvalidHostName)
		}

		// Given input like "http://localhost:8080:8383", url.Parse() will
		// return host as "localhost:8000", which isn't a vaild DNS name.
		if strings.Contains(host, ":") {
			return nil, newInvalidNodeError(errInvalidHostName)
		}

		port := url.Port()
		if port == "" {
			port = DefaultDialPort
		}
		if !validPort(port) {
			return nil, newInvalidNodeError(errInvalidPortNumber)
		}

		addresses = append(addresses, strings.TrimRight(url.String(), "/"))
	}

	return addresses, nil
}

func validPort(port string) bool {
	intPort, err := strconv.Atoi(port)

	return (err == nil) &&
		(intPort > 0) &&
		(intPort <= 65535)
}
