// +build windows

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
	"fmt"
	"net"
	"net/url"
	"strings"
	"syscall"
	"time"

	"github.com/Microsoft/go-winio"
)

const (
	tcpProtocol   = "tcp"
	npipeProtocol = "npipe"
)

// CreateListener creates a listener on the specified endpoint.
func CreateListener(endpoint string) (net.Listener, error) {
	protocol, addr, err := parseEndpoint(endpoint)
	if err != nil {
		return nil, err
	}

	switch protocol {
	case tcpProtocol:
		return net.Listen(tcpProtocol, addr)

	case npipeProtocol:
		return winio.ListenPipe(addr, nil)

	default:
		return nil, fmt.Errorf("only support tcp and npipe endpoint")
	}
}

// GetAddressAndDialer returns the address parsed from the given endpoint and a dialer.
func GetAddressAndDialer(endpoint string) (string, func(addr string, timeout time.Duration) (net.Conn, error), error) {
	protocol, addr, err := parseEndpoint(endpoint)
	if err != nil {
		return "", nil, err
	}

	if protocol == tcpProtocol {
		return addr, tcpDial, nil
	}

	if protocol == npipeProtocol {
		return addr, npipeDial, nil
	}

	return "", nil, fmt.Errorf("only support tcp and npipe endpoint")
}

func tcpDial(addr string, timeout time.Duration) (net.Conn, error) {
	return net.DialTimeout(tcpProtocol, addr, timeout)
}

func npipeDial(addr string, timeout time.Duration) (net.Conn, error) {
	return winio.DialPipe(addr, &timeout)
}

func parseEndpoint(endpoint string) (string, string, error) {
	// url.Parse doesn't recognize \, so replace with / first.
	endpoint = strings.Replace(endpoint, "\\", "/", -1)
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", "", err
	}

	if u.Scheme == "tcp" {
		return "tcp", u.Host, nil
	} else if u.Scheme == "npipe" {
		if strings.HasPrefix(u.Path, "//./pipe") {
			return "npipe", u.Path, nil
		}

		// fallback host if not provided.
		host := u.Host
		if host == "" {
			host = "."
		}
		return "npipe", fmt.Sprintf("//%s%s", host, u.Path), nil
	} else if u.Scheme == "" {
		return "", "", fmt.Errorf("Using %q as endpoint is deprecated, please consider using full url format", endpoint)
	} else {
		return u.Scheme, "", fmt.Errorf("protocol %q not supported", u.Scheme)
	}
}

// LocalEndpoint empty implementation
func LocalEndpoint(path, file string) (string, error) {
	return "", fmt.Errorf("LocalEndpoints are unsupported in this build")
}

var tickCount = syscall.NewLazyDLL("kernel32.dll").NewProc("GetTickCount64")

// GetBootTime returns the time at which the machine was started, truncated to the nearest second
func GetBootTime() (time.Time, error) {
	currentTime := time.Now()
	output, _, err := tickCount.Call()
	if errno, ok := err.(syscall.Errno); !ok || errno != 0 {
		return time.Time{}, err
	}
	return currentTime.Add(-time.Duration(output) * time.Millisecond).Truncate(time.Second), nil
}
