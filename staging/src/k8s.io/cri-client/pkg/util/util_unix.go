//go:build freebsd || linux || darwin
// +build freebsd linux darwin

/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

const (
	// unixProtocol is the network protocol of unix socket.
	unixProtocol = "unix"
)

// CreateListener creates a listener on the specified endpoint.
func CreateListener(endpoint string) (net.Listener, error) {
	protocol, addr, err := parseEndpointWithFallbackProtocol(endpoint, unixProtocol)
	if err != nil {
		return nil, err
	}
	if protocol != unixProtocol {
		return nil, fmt.Errorf("only support unix socket endpoint")
	}

	// Unlink to cleanup the previous socket file.
	err = unix.Unlink(addr)
	if err != nil && !os.IsNotExist(err) {
		return nil, fmt.Errorf("failed to unlink socket file %q: %w", addr, err)
	}

	if err := os.MkdirAll(filepath.Dir(addr), 0750); err != nil {
		return nil, fmt.Errorf("error creating socket directory %q: %w", filepath.Dir(addr), err)
	}

	// Create the socket on a tempfile and move it to the destination socket to handle improper cleanup
	file, err := os.CreateTemp(filepath.Dir(addr), "")
	if err != nil {
		return nil, fmt.Errorf("failed to create temporary file: %w", err)
	}

	if err := os.Remove(file.Name()); err != nil {
		return nil, fmt.Errorf("failed to remove temporary file: %w", err)
	}

	l, err := net.Listen(protocol, file.Name())
	if err != nil {
		return nil, err
	}

	if err = os.Rename(file.Name(), addr); err != nil {
		return nil, fmt.Errorf("failed to move temporary file to addr %q: %w", addr, err)
	}

	return l, nil
}

// GetAddressAndDialer returns the address parsed from the given endpoint and a context dialer.
func GetAddressAndDialer(endpoint string) (string, func(ctx context.Context, addr string) (net.Conn, error), error) {
	protocol, addr, err := parseEndpointWithFallbackProtocol(endpoint, unixProtocol)
	if err != nil {
		return "", nil, err
	}
	if protocol != unixProtocol {
		return "", nil, fmt.Errorf("only support unix socket endpoint")
	}

	return addr, dial, nil
}

func dial(ctx context.Context, addr string) (net.Conn, error) {
	return (&net.Dialer{}).DialContext(ctx, unixProtocol, addr)
}

func parseEndpointWithFallbackProtocol(endpoint string, fallbackProtocol string) (protocol string, addr string, err error) {
	if protocol, addr, err = parseEndpoint(endpoint); err != nil && protocol == "" {
		fallbackEndpoint := fallbackProtocol + "://" + endpoint
		protocol, addr, err = parseEndpoint(fallbackEndpoint)
	}
	return
}

func parseEndpoint(endpoint string) (string, string, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", "", err
	}

	switch u.Scheme {
	case "tcp":
		return "tcp", u.Host, nil

	case "unix":
		return "unix", u.Path, nil

	case "":
		return "", "", fmt.Errorf("using %q as endpoint is deprecated, please consider using full url format", endpoint)

	default:
		return u.Scheme, "", fmt.Errorf("protocol %q not supported", u.Scheme)
	}
}
