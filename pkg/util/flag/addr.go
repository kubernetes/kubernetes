/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package flag

import (
	"fmt"
	"net"
	"net/url"
	"strconv"
	"strings"
)

// urlPrefixes is the list of string prefix values that may indicate a URL
// is present.
var urlPrefixes = []string{"http://", "https://", "tcp://"}

// Addr is a flag type that attempts to load a host, IP, host:port, or
// URL value from a string argument. It tracks whether the value was set
// and allows the caller to provide defaults for the scheme and port.
type Addr struct {
	// Specified by the caller
	DefaultScheme string
	DefaultPort   int
	AllowPrefix   bool

	// Provided will be true if Set is invoked
	Provided bool
	// Value is the exact value provided on the flag
	Value string

	// URL represents the user input. The Host field is guaranteed
	// to be set if Provided is true
	URL *url.URL
	// Host is the hostname or IP portion of the user input
	Host string
	// IPv6Host is true if the hostname appears to be an IPv6 input
	IPv6Host bool
	// Port is the port portion of the user input. Will be 0 if no port was found
	// and no default port could be established.
	Port int
}

// Default creates a new Address with the value set
func (a Addr) Default() Addr {
	if err := a.Set(a.Value); err != nil {
		panic(err)
	}
	a.Provided = false
	return a
}

// String returns the string representation of the Addr
func (a *Addr) String() string {
	if a.URL == nil {
		return a.Value
	}
	return a.URL.String()
}

// Set attempts to set a string value to an address
func (a *Addr) Set(value string) error {
	scheme := a.DefaultScheme
	if len(scheme) == 0 {
		scheme = "tcp"
	}
	addr := &url.URL{
		Scheme: scheme,
	}

	switch {
	case a.isURL(value):
		parsed, err := url.Parse(value)
		if err != nil {
			return fmt.Errorf("not a valid URL: %v", err)
		}
		if !a.AllowPrefix {
			parsed.Path = ""
		}
		parsed.RawQuery = ""
		parsed.Fragment = ""

		if strings.Contains(parsed.Host, ":") {
			host, port, err := net.SplitHostPort(parsed.Host)
			if err != nil {
				return fmt.Errorf("not a valid host:port: %v", err)
			}
			portNum, err := strconv.ParseUint(port, 10, 64)
			if err != nil {
				return fmt.Errorf("not a valid port: %v", err)
			}
			a.Host = host
			a.Port = int(portNum)

		} else {
			port := 0
			switch parsed.Scheme {
			case "http":
				port = 80
			case "https":
				port = 443
			default:
				return fmt.Errorf("no port specified")
			}
			a.Host = parsed.Host
			a.Port = port
		}
		addr = parsed

	case isIPv6Host(value):
		a.Host = value
		a.Port = a.DefaultPort

	case strings.Contains(value, ":"):
		host, port, err := net.SplitHostPort(value)
		if err != nil {
			return fmt.Errorf("not a valid host:port: %v", err)
		}
		portNum, err := strconv.ParseUint(port, 10, 64)
		if err != nil {
			return fmt.Errorf("not a valid port: %v", err)
		}
		a.Host = host
		a.Port = int(portNum)

	default:
		port := a.DefaultPort
		if port == 0 {
			switch a.DefaultScheme {
			case "http":
				port = 80
			case "https":
				port = 443
			default:
				return fmt.Errorf("no port specified")
			}
		}
		a.Host = value
		a.Port = port
	}
	addr.Host = net.JoinHostPort(a.Host, strconv.FormatInt(int64(a.Port), 10))

	if value != a.Value {
		a.Provided = true
	}
	a.URL = addr
	a.IPv6Host = isIPv6Host(a.Host)
	a.Value = value

	return nil
}

// Type returns a string representation of what kind of value this is
func (a *Addr) Type() string {
	return "string"
}

// isURL returns true if the provided value appears to be a valid URL.
func (a *Addr) isURL(value string) bool {
	prefixes := urlPrefixes
	if a.DefaultScheme != "" {
		prefixes = append(prefixes, fmt.Sprintf("%s://", a.DefaultScheme))
	}
	for _, p := range prefixes {
		if strings.HasPrefix(value, p) {
			return true
		}
	}
	return false
}

// isIPv6Host returns true if the value appears to be an IPv6 host string (that does
// not include a port).
func isIPv6Host(value string) bool {
	if strings.HasPrefix(value, "[") {
		return false
	}
	return strings.Contains(value, "%") || strings.Count(value, ":") > 1
}
